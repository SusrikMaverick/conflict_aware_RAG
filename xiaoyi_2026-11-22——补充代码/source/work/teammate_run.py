"""Run a pinned teammate implementation without changing its prompts or decisions."""
import argparse
import ast
import hashlib
import importlib.util
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'materials/teammate_20261115/upstream'
PLAN = ROOT / 'planning/teammate_20261115'
RUNS = ROOT / 'runs/teammate_20261115'
METHODS = ('evidence_map', 'evidence_map_memory', 'teammate_plain')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def load_pipeline():
    spec = importlib.util.spec_from_file_location('teammate_pipeline', UPSTREAM / 'src/pipeline.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_prompt():
    # Read the literal prompt without importing the upstream 72B server launcher.
    tree = ast.parse((UPSTREAM / 'src/benchmark_ramdocs_baseline.py').read_text())
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PROMPT' for t in node.targets))


def parse_baseline(text):
    import re
    match = re.search(r'All Correct Answers\s*:', text, re.I)
    if not match:
        raise ValueError('missing_answer_list_marker')
    fragment = text[match.end():].lstrip()
    if fragment.lower().startswith(('unknown', "'unknown'", '"unknown"')):
        return []
    value, _ = json.JSONDecoder().raw_decode(fragment)
    if not isinstance(value, list) or any(not isinstance(item, (str, int, float)) for item in value):
        raise ValueError('invalid_answer_list')
    return [str(item) for item in value]


class Client:
    """Every call is fresh, saved once, and usable for a separate exact replay audit."""
    def __init__(self, port, directory, tokenizer, replay=False):
        self.port, self.directory, self.tokenizer = port, directory, tokenizer
        self.replay, self.records = replay, []
        self.directory.mkdir(parents=True, exist_ok=True)

    def call(self, messages, **kwargs):
        extra = kwargs.pop('extra_body', {})
        payload = dict(model='teammate-qwen25-7b', messages=messages, top_p=1.0, seed=576, **kwargs, **extra)
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(ids) + payload['max_tokens'] > 32768:
            raise ValueError('context_overflow: no input truncation permitted')
        index = len(self.records)
        path = self.directory / f'{index:04d}.json'
        if self.replay or path.exists():
            record = json.loads(path.read_text())
            assert record['request'] == payload and record['prompt_token_ids'] == ids
        else:
            request = urllib.request.Request(f'http://127.0.0.1:{self.port}/v1/chat/completions',
                data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            start = time.time()
            with urllib.request.urlopen(request, timeout=1800) as response:
                output = json.load(response)
            choice, usage = output['choices'][0], output['usage']
            result = dict(text=choice['message']['content'] or '', prompt_tokens=usage['prompt_tokens'],
                          completion_tokens=usage['completion_tokens'], finish_reason=choice['finish_reason'])
            record = dict(request=payload, response=output, prompt_token_ids=ids,
                          input_sha256=digest(payload), result=result, seconds=time.time()-start)
            save(path, record)
        self.records.append(record)
        return record['result']


def compute(pipeline, client, row, method):
    if method == 'teammate_plain':
        documents = '\n'.join(f"Document {i}: {d['text']}" for i, d in enumerate(row['documents'], 1))
        reply = client.call([dict(role='user', content=baseline_prompt().format(question=row['question'], documents=documents))],
                            temperature=0, max_tokens=128, extra_body={'repetition_penalty': 1.05})
        answers = parse_baseline(reply['text'])
        return dict(accepted_answers=answers, status='answered' if answers else 'abstained',
                    output=reply['text'], capped=int(reply['finish_reason']=='length'))
    # Public document IDs are deliberately omitted: upstream interprets any extra field as source metadata.
    documents = [dict(text=d['text']) for d in row['documents']]
    result = pipeline.run(client, row['question'], documents, mode='forced', memory=method=='evidence_map_memory')
    answers = [a for entry in result['entries'] for a in entry['answers']]
    assert all(isinstance(a, str) for a in answers)
    return dict(accepted_answers=answers, status='answered' if answers else 'abstained', native=result)


def smoke_rows(pipeline):
    synthetic = [
        dict(question_id='synthetic-avatar', **pipeline.EXAMPLE),
        dict(question_id='synthetic-namesakes', question='When was Riverlight Club founded?', documents=[
            dict(text='Riverlight Club in Alder was founded in 1982.'),
            dict(text='Riverlight Club in Birch was founded in 1990.')]),
        dict(question_id='synthetic-empty', question='What is the secret code?', documents=[dict(text='No secret code is given.')]),
    ]
    inputs = json.loads((PLAN/'inputs.json').read_text())
    # Fixed, uniformly spread cases; no outcomes or labels determine the smoke sample.
    for prefix in ['ramdocs-', 'rgb-', 'cb-']:
        group = [r for r in inputs if r['question_id'].startswith(prefix)]
        assert group, prefix
        synthetic += [group[i] for i in [0, len(group)//2, len(group)-1]]
    return synthetic


def readiness(records):
    import collections
    steps = collections.defaultdict(collections.Counter)
    kept = dropped = unread = text_group = groups = 0
    errors = []
    for r in records:
        if r['status']=='error':
            errors.append(dict(question_id=r['question_id'], error=r['error']));continue
        if 'native' not in r:
            continue
        trace = r['native']['trace']
        for c in trace['calls']:
            steps[c['step']].update(calls=1, capped=int(c['capped']), unreadable=int(not c['json']))
        for e in trace['extract']:
            kept += len(e['cards']);dropped += len(e['dropped']);unread += int('unread' in e)
        groups += 1;text_group += int(trace['map']['grouped_by']=='text match')
    calls=sum(v['calls'] for v in steps.values());capped=sum(v['capped'] for v in steps.values())
    invalid=sum(v['unreadable'] for v in steps.values())
    problems = []
    if errors:problems.append('runtime_errors')
    if calls and capped/calls>.03:problems.append('over_3pct_calls_capped')
    if calls and invalid/calls>.03:problems.append('over_3pct_replies_unreadable')
    if dropped/max(1,kept+dropped)>.15:problems.append('over_15pct_cards_dropped')
    if text_group/max(1,groups)>.10:problems.append('over_10pct_text_grouping_fallback')
    return dict(ready=not problems, problems=problems, steps=dict(steps), calls=calls,
                capped=capped, unreadable=invalid, kept_cards=kept, dropped_cards=dropped,
                unread_documents=unread, text_grouping_fallback=text_group, questions=groups, errors=errors)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=METHODS,required=True)
    parser.add_argument('--port',type=int,required=True);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--replay',action='store_true');parser.add_argument('--repeat-check',action='store_true')
    args=parser.parse_args();pipeline=load_pipeline()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen2.5-7B-Instruct',local_files_only=True)
    rows=smoke_rows(pipeline) if args.smoke else json.loads((PLAN/'inputs.json').read_text())
    stage='smoke' if args.smoke else 'full'
    if args.repeat_check:
        assert args.smoke
        rows=rows[:1];stage='repeat'
    out=RUNS/stage/args.method;out.mkdir(parents=True,exist_ok=True)
    predictions=[]
    for index,row in enumerate(rows):
        dest=out/'questions'/(row['question_id']+'.json')
        if dest.exists() and not args.replay:
            record=json.loads(dest.read_text());assert record['input_sha256']==digest(row)
        else:
            client=Client(args.port,out/'requests'/row['question_id'],tokenizer,args.replay)
            record=dict(question_id=row['question_id'],method=args.method,input_sha256=digest(row))
            try:
                record.update(compute(pipeline,client,row,args.method))
            except Exception as exc:
                record.update(status='error',accepted_answers=[],error=repr(exc))
            record['request_count']=len(client.records)
            if args.replay:
                assert record==json.loads(dest.read_text()),('replay_mismatch',row['question_id'])
            else:
                save(dest,record)
        predictions.append(record)
        print(f'{stage} {args.method}: {index+1}/{len(rows)} | {record["status"]}',flush=True)
    receipt=readiness(predictions)
    receipt.update(predictions=len(predictions),complete=True,replay=args.replay)
    if args.repeat_check:
        original=json.loads((RUNS/'smoke'/args.method/'questions'/f'{rows[0]["question_id"]}.json').read_text())
        receipt['identical_repeat']=original==predictions[0]
        if not receipt['identical_repeat']:
            receipt['ready']=False;receipt['problems'].append('repeat_changed')
    save(out/('replay.json' if args.replay else 'finished.json'),receipt)
    if args.smoke and not receipt['ready']:
        raise SystemExit(2)


if __name__=='__main__':main()

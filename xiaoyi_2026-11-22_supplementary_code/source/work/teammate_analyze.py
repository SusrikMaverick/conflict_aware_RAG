"""Score only complete frozen runs, with failures retained in every denominator."""
import collections
import csv
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/'deliverables/Conflict_QA_Submission_v7/code'
sys.path.insert(0,str(PKG))
from evaluation import score


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    out=ROOT/'analysis/teammate_20261115';out.mkdir(parents=True,exist_ok=True)
    runs=ROOT/'runs/teammate_20261115/full'
    labels=json.loads((PKG/'data/evaluation.json').read_text())
    inputs=json.loads((ROOT/'planning/teammate_20261115/inputs.json').read_text())
    indexed={r['question_id']:r for r in inputs};ids=set(indexed)
    spec=importlib.util.spec_from_file_location('teammate_adapter',ROOT/'work/teammate_run.py')
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    pipeline=adapter.load_pipeline()
    predictions=[];audit=collections.Counter();cost=[];limitations=[]
    for method in adapter.METHODS:
        directory=runs/method
        assert json.loads((directory/'finished.json').read_text())['predictions']==1100
        assert json.loads((directory/'replay.json').read_text())['replay']
        records=[json.loads(p.read_text()) for p in sorted((directory/'questions').glob('*.json'))]
        assert len(records)==1100 and {r['question_id'] for r in records}==ids
        for record in records:
            q=record['question_id'];row=indexed[q]
            assert record['input_sha256']==adapter.digest(row)
            requests=[json.loads(p.read_text()) for p in sorted((directory/'requests'/q).glob('*.json'))]
            assert len(requests)==record['request_count']
            for request in requests:
                assert request['request']['model']=='teammate-qwen25-7b'
                assert request['input_sha256']==adapter.digest(request['request'])
                audit['stored_model_calls']+=1
            predictions.append(dict(question_id=q,method=method,status=record['status'],
                                    accepted_answers=record['accepted_answers'],origin='Fresh 7B inference'))
            if 'native' in record:
                result=record['native'];trace=result['trace'];audit['native_results']+=1
                assert result['calls']==len(requests)==len(trace['calls'])
                for extraction in trace['extract']:
                    audit['fresh_document_extractions']+=1
                    document=row['documents'][extraction['doc']-1]['text']
                    for card in extraction['cards']:
                        assert pipeline.find_quote(card['evidence'],document)==card['quote']
                        audit['normalized_exact_quotes' if card['quote']=='exact' else 'approximate_quotes']+=1
                        audit['literal_quotes']+=int(card['evidence'] in document)
                for cluster in trace['map']['clusters']:
                    if len(cluster['answer'])==0:limitations.append(dict(question_id=q,method=method,issue='empty_cluster_answer'))
                if len(trace['map']['clusters'])>26:
                    limitations.append(dict(question_id=q,method=method,issue='over_26_clusters_letter_limit'))
        diagnosis=adapter.readiness(records)
        request_rows=[json.loads(p.read_text()) for p in directory.glob('requests/*/*.json')]
        cost.append(dict(method=method,questions=1100,calls=len(request_rows),
                         input_tokens=sum(r['result']['prompt_tokens'] for r in request_rows),
                         output_tokens=sum(r['result']['completion_tokens'] for r in request_rows),
                         capped_calls=sum(r['result']['finish_reason']=='length' for r in request_rows),
                         failed_questions=sum(r['status']=='error' for r in records),
                         unreadable_step_replies=diagnosis['unreadable'],dropped_cards=diagnosis['dropped_cards']))
    historical=[]
    for filename in ['predictions.jsonl.gz','baselines/predictions.jsonl.gz']:
        with gzip.open(PKG/'data'/filename,'rt') as f:
            historical += [json.loads(s)|{'origin':'Historical 7B inference; rescored, not rerun'} for s in f]
    predictions+=historical
    names=dict(ours_1024='Docs + Cards (Ours)',FR4_M03='Evidence Pruning (Ours)',WR5_M02='Memory + Cards (Ours)',
               WR3_M01='Context Review (Ours)',WR2_M08='Answer Agreement (Ours)',WR1_M05='Dual Review (Ours)',
               plain_1024='Plain RAG',madam='MADAM-RAG',self_reflection='Self-reflection',
               astute_structured1024='Astute RAG (structured)',faithfulrag_adaptation='FaithfulRAG (adapted)',
               trust='TrustRAG (adapted)',robust_keyword='RobustRAG (keyword)',
               evidence_map='Teammate Evidence Map',evidence_map_memory='Teammate Evidence Map + Memory',
               teammate_plain='Teammate Plain RAG (128 tokens)')
    totals=collections.defaultdict(collections.Counter);per=[];seen=set()
    for p in predictions:
        q=p['question_id'];m=p['method'];assert (q,m) not in seen;seen.add((q,m))
        lab=labels[q];values=score(p['accepted_answers'],p['status'],lab['groups'],lab['wrong_groups'])
        per.append(dict(dataset=lab['dataset'],question_id=q,method=names[m],origin=p['origin'],**values))
        totals[lab['dataset'],m].update(n=1,**values)
    rows=[]
    for (dataset,method),counts in sorted(totals.items()):
        rows.append(dict(dataset=dataset,method=names[method],**counts,
                         answer_match_pct=100*counts['answer_match']/counts['n'],
                         strict_set_pct=100*counts['strict_set']/counts['n'],
                         labeled_wrong_pct=100*counts['labeled_wrong']/counts['n']))
    assert len(rows)==48 and len(per)==17600
    write_csv(out/'summary.csv',rows);write_csv(out/'per_question.csv',per);write_csv(out/'cost.csv',cost)
    (out/'audit.json').write_text(json.dumps(dict(checks=dict(audit),limitations=limitations,
        gold_loaded_only_after_generation=True,complete=True),indent=2)+'\n')
    (out/'READ_ME.txt').write_text(
        'Qwen2.5-7B-Instruct BF16, fixed upstream commit 6a4c742f4d925363f2cf7e6bdc5987c82bba2912. '
        'Three fresh conditions across RAMDocs500, RGB100 and ConflictBank500. Both Evidence Map modes use '
        'the original forced policy, prompts, thresholds and per-step budgets, with memory off/on. '
        'Fresh per-document cards in both modes. No guided JSON or added answer-generation prompt. '
        'The runtime is local vLLM0.17.1 rather than upstream vLLM0.32.0; the model is 7B BF16 rather than72B AWQ. '
        'The teammate plain baseline retains128tokens. Its answer list is parsed separately; explanations are excluded. '
        'Malformed answer lists are retained as parse/runtime failures. Historical13methods are rescored, not rerun. '
        'Budgets and contracts differ; this is not a compute-matched experiment or an independent test. '
        'All1100questions were previously observed. All three metrics, failures and abstentions are reported. '
        'Approximate quote matching and transitive relation merging are retained upstream behavior, not truth checks.\n')
    print(str(out),flush=True)


if __name__=='__main__':main()

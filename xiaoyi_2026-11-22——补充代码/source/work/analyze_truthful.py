"""Audit completed generation, then score with the frozen delivery evaluator."""
import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import truthful_run as tr

sys.path.insert(0,str(tr.PACKAGE))
from evaluation import score, METRIC_VERSION

OUT=tr.ROOT/'analysis/external_20261112'
COUNTS=['answer_match','strict_set','labeled_wrong','failed','abstained']


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    finished=json.loads((tr.RUN/'execution_finished.json').read_text())
    assert finished['count']==1100
    protocol=json.loads((tr.PLAN/'protocol.json').read_text())
    assert protocol['hashes']==tr.source_hashes()
    retained=json.loads((tr.PLAN/'retained_outputs_before_parallel.json').read_text())
    assert all(hashlib.sha256((tr.ROOT/p).read_bytes()).hexdigest()==h for p,h in retained.items())
    inputs={r['question_id']:r for r in json.loads((tr.PACKAGE/'data/inputs.json').read_text())}
    paths=list((tr.RUN/'outputs').glob('*/prediction.json'))
    assert len(paths)==1100
    outputs={}
    requested=set()
    audit=collections.Counter()
    cost=[]
    for path in sorted(paths):
        out=json.loads(path.read_text());qid=out['question_id'];row=inputs[qid]
        assert qid not in outputs and out['input_hash']==tr.digest(row)
        outputs[qid]=out
        assert out['status'] in {'answered','abstain','error','parse_error'}
        if out['status'] in {'error','parse_error','abstain'}:assert out['accepted_answers']==[]
        counts=collections.Counter()
        entropies=[]
        for req in out['requests']:
            key=req['key'];requested.add(key)
            with gzip.open(tr.RUN/'requests'/(key+'.json.gz'),'rt') as f:saved=json.load(f)
            payload=saved['request']
            assert key==tr.digest({'revision':tr.MODEL_REV,'runtime':'vllm0.8.5.post1-bf16','payload':payload})
            assert payload['temperature']==0 and payload['seed']==576
            assert saved['preflight_tokens']+payload['max_tokens']<=32768
            assert req['usage']==saved['response'].get('usage',{})
            counts['calls']+=1
            counts['input_tokens']+=req['usage'].get('prompt_tokens',0)
            counts['output_tokens']+=req['usage'].get('completion_tokens',0)
            counts['truncated_calls']+=req['finish_reason']=='length'
            counts[req['stage']+'_calls']+=int(req['stage'] not in {'entropy','final'})
            counts[req['stage']+'_truncations']+=req['finish_reason']=='length'
            counts['cached_calls']+=req['cache_hit']
            if req['stage']=='entropy':
                assert payload['max_tokens']==32 and payload['top_logprobs']==10
                counts['entropy_calls']+=1
                token_entropies=[]
                for token in saved['response']['choices'][0]['logprobs']['content']:
                    weights=[math.exp(v['logprob']) for v in token['top_logprobs']]
                    z=sum(weights)+1e-10
                    probs=[p/z for p in weights]
                    token_entropies.append(-sum(p*math.log(p+1e-10) for p in probs))
                entropies.append(sum(token_entropies)/len(token_entropies))
            if req['stage']=='final':
                counts['final_calls']+=1
                sample_context='\n'.join(f'Document {i}: {d["text"]}' for i,d in enumerate(row['documents'],1))
                pg=tr.PromptGenerator(llm_type='openai',task='qa-cot')
                expected=pg.generate_qa_prompt_normal_cot(context=sample_context,question=row['question'],facts='\n\n'.join(out['filtered_elements']))+tr.CONTRACT
                assert payload['messages']==[{'role':'system','content':pg.system_prompt},{'role':'user','content':expected}]
                assert out['raw_output']==saved['response']['choices'][0]['message']['content'].strip()
                assert payload['max_tokens']==1024
                if out['status']!='parse_error':assert out['accepted_answers']==tr.parse(out['raw_output'])
                audit['final_payloads_verified']+=1
        if out['status']!='error':
            assert counts['final_calls']==1
            assert counts['entropy_calls']==1+min(30,len(out['retrieved_elements']))
            candidates=out['retrieved_elements'][:30]
            base=entropies[0]
            eligible=[(e,h-base) for e,h in zip(candidates,entropies[1:]) if h-base>=0 or (h!=0 and base/h<=10)]
            if not eligible and candidates:
                byelement=dict(zip(candidates,entropies[1:]))
                largest=max(byelement.values())
                eligible=[(e,h) for e,h in byelement.items() if h==largest]
            expected=[e for e,delta in sorted(eligible,key=lambda pair:pair[1],reverse=True)[:10]]
            assert expected==out['filtered_elements'],qid
            audit['entropy_selections_verified']+=1
        audit['outputs_verified']+=1
        audit['logical_calls']+=counts['calls']
        cost.append(dict(question_id=qid,status=out['status'],calls=counts['calls'],graph_calls=counts['graph_calls'],retrieve_calls=counts['retrieve_calls'],entropy_calls=counts['entropy_calls'],final_calls=counts['final_calls'],input_tokens=counts['input_tokens'],output_tokens=counts['output_tokens'],truncated_calls=counts['truncated_calls'],graph_truncations=counts['graph_truncations'],retrieve_truncations=counts['retrieve_truncations'],entropy_truncations=counts['entropy_truncations'],final_truncations=counts['final_truncations'],cached_calls=counts['cached_calls'],retrieved=len(out.get('retrieved_elements',[])),filtered=len(out.get('filtered_elements',[])),wall_seconds=out['wall_seconds']))
    assert set(inputs)==set(outputs)
    OUT.mkdir(parents=True,exist_ok=True)
    audit['unique_requests']=len(requested)
    audit['retained_predictions_unchanged']=len(retained)
    tr.write(OUT/'execution_audit.json',dict(audit))
    csv_write(OUT/'cost_per_question.csv',cost)

    # Labels are first loaded after the complete execution and payload audit.
    gold=json.loads((tr.PACKAGE/'data/evaluation.json').read_text())
    cost_summary=[]
    for dataset in ['RAMDocs','RGB','ConflictBank-500']:
        group=[r for r in cost if gold[r['question_id']]['dataset']==dataset]
        totals_cost=dict(dataset=dataset,n=len(group))
        for key in ['calls','graph_calls','retrieve_calls','entropy_calls','final_calls','input_tokens','output_tokens','graph_truncations','retrieve_truncations','entropy_truncations','final_truncations','cached_calls']:
            totals_cost[key]=sum(r[key] for r in group)
        for key in ['calls','input_tokens','output_tokens']:
            totals_cost[key+'_per_question']=totals_cost[key]/len(group)
        totals_cost['questions_with_graph_paths']=sum(r['retrieved']>0 for r in group)
        cost_summary.append(totals_cost)
    csv_write(OUT/'cost_summary.csv',cost_summary)
    with (tr.PACKAGE.parent/'results/per_question.csv').open() as f:historic=list(csv.DictReader(f))
    rows=[]
    for h in historic:
        g=gold[h['question_id']]
        calculated=score(json.loads(h['answers']),h['status'],g['groups'],g['wrong_groups'])
        assert all(int(h[k])==v for k,v in calculated.items()),h['question_id']
        rows.append({**h,**calculated})
    for qid,out in outputs.items():
        g=gold[qid];v=score(out['accepted_answers'],out['status'],g['groups'],g['wrong_groups'])
        rows.append(dict(dataset=g['dataset'],method='truthful',method_name='TruthfulRAG (label-free)',legacy_method='truthfulrag_label_free',question_id=qid,status=out['status'],metric=METRIC_VERSION,**v,answers=json.dumps(out['accepted_answers'],ensure_ascii=False)))
    assert len(rows)==15400
    totals={}
    for r in rows:
        k=(r['dataset'],r['method'])
        t=totals.setdefault(k,dict(dataset=r['dataset'],method=r['method'],method_name=r['method_name'],metric=METRIC_VERSION,n=0,**{c:0 for c in COUNTS}))
        t['n']+=1
        for c in COUNTS:t[c]+=int(r[c])
    for t in totals.values():
        for c in COUNTS[:3]:t[c+'_pct']=100*t[c]/t['n']
    order=['plain','madam','reflection','astute','faithful','trust','robust','truthful','cards','evidence_prune','memory_cards','context_review','answer_agreement','dual_review']
    summary=sorted(totals.values(),key=lambda x:({'RAMDocs':0,'RGB':1,'ConflictBank':2}.get(x['dataset'],3),order.index(x['method'])))
    assert len(summary)==42
    csv_write(OUT/'summary.csv',summary)
    csv_write(OUT/'per_question.csv',rows)
    csv_write(OUT/'new_method_summary.csv',[r for r in summary if r['method']=='truthful'])
    tr.write(OUT/'failures.json',[o for o in outputs.values() if o['status'] in {'error','parse_error'}])

    # Paired question bootstrap, descriptive only (already observed benchmarks).
    rng=np.random.default_rng(576)
    comparisons=[]
    for dataset in sorted({r['dataset'] for r in rows}):
        bymethod={m:{r['question_id']:r for r in rows if r['dataset']==dataset and r['method']==m} for m in order}
        ids=sorted(bymethod['truthful'])
        sample=rng.integers(0,len(ids),(5000,len(ids)))
        for method in order:
            if method=='truthful':continue
            for metric in COUNTS[:3]:
                delta=np.array([int(bymethod['truthful'][q][metric])-int(bymethod[method][q][metric]) for q in ids])
                lo,hi=np.quantile(delta[sample].mean(1)*100,[.025,.975])
                comparisons.append(dict(dataset=dataset,reference=method,metric=metric,delta_count=int(delta.sum()),delta_pp=float(delta.mean()*100),ci_low_pp=float(lo),ci_high_pp=float(hi)))
    csv_write(OUT/'paired_comparisons.csv',comparisons)
    tr.write(OUT/'provenance.json',{'new_predictions':1100,'historical_predictions_reused':14300,'comparison_rows':42,'metric':METRIC_VERSION,'protocol':str(tr.PLAN/'protocol.json'),'original_code':str(tr.ROOT/'materials/external_20261112/TruthfulRAG'),'label_free_patch':str(tr.PLAN/'upstream.patch'),'scope':'Adapted pipeline under common Qwen2.5-7B BF16 backbone; not original paper scores; total budgets differ','source_hashes':protocol['hashes']})
    print(json.dumps([r for r in summary if r['method']=='truthful'],indent=2))


if __name__=='__main__':main()

"""Audit completed fresh runs, then score with the unchanged project metric."""
import json,csv,hashlib,sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PKG=ROOT/'deliverables/Conflict_QA_Submission_v7/code';PLAN=ROOT/'planning/backbones_20261113';OUT=ROOT/'analysis/backbones_20261113'
sys.path[:0]=[str(PKG),str(PKG/'src')]
from evaluation import score,METRIC_VERSION
from source_cards import segments,source_span
from real_runtime import digest
from answer_format import parse

def save_csv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 protocol=json.loads((PLAN/'protocol.json').read_text())
 for name,sha in protocol['hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
 inputs=json.loads((PLAN/'inputs.json').read_text());byid={r['question_id']:r for r in inputs};audits={};allpred=[]
 for model,meta in protocol['models'].items():
  root=ROOT/'runs/backbones_20261113'/model/'full';done=json.loads((root/'finished.json').read_text());assert done['n']==1100 and done['predictions']==4400 and done['protocol']==digest(protocol)
  predictions=json.loads((root/'all_predictions.json').read_text());assert len(predictions)==4400
  keys={(p['question_id'],p['method']) for p in predictions};assert len(keys)==4400 and keys=={(q,m) for q in byid for m in protocol['methods']}
  requests={};counts=collections.Counter()
  for p in (root/'runs').rglob('request_cache/*.json'):
   d=json.loads(p.read_text());assert d['model_revision']==meta['revision'] and d['request']['model']==model
   assert d['request']['chat_template_kwargs']=={'enable_thinking':False}
   assert d['preflight_tokens']+d['request']['max_tokens']<=32768
   assert d['preflight_tokens']==d['usage']['prompt_tokens']
   assert digest({'revision':meta['revision'],'protocol':digest(protocol),'payload':d['request']})==d['key']
   requests[d['key']]=d
  for p in (root/'request_cache').glob('*.json'):
   d=json.loads(p.read_text());assert d['model_revision']==meta['revision'] and d['request']['model']==model;requests[d['key']]=d
  for d in requests.values():
   assert d['request']['chat_template_kwargs']=={'enable_thinking':False}
   assert d['preflight_tokens']==d['usage']['prompt_tokens']
   assert d['preflight_tokens']+d['request']['max_tokens']<=32768
   assert digest({'revision':meta['revision'],'protocol':digest(protocol),'payload':d['request']})==d['key']
   counts['calls']+=1;counts['input_tokens']+=d['usage']['prompt_tokens'];counts['output_tokens']+=d['usage']['completion_tokens'];counts['cap_hits']+=d['finish_reason']=='length'
  for p in (root/'document_cards').rglob('*.json'):
   d=json.loads(p.read_text());assert d['backbone']==model and d['revision']==meta['revision'];assert digest(d['input'])==d['input_sha256'];row=byid[d['input']['question_id']];doc=d['input']['documents'][0];assert doc in row['documents'] and d['input']['question']==row['question'];counts['document_extractions']+=1
   for c in d['output']['cards']:
    span=source_span(c['source_ids'],segments(doc['text']),doc['text']);assert all(c[k]==v for k,v in span.items());counts['cards']+=1
   counts['extraction_issues']+=len(d['output']['extraction_issues'])
  source_paths={'cards':'runs/cross_dataset_v1/outputs/ours_1024','evidence_prune':'runs/cross_dataset_v1/outputs/FR_4_M03','memory_cards':'runs/wrong_reduction_v1/round5/outputs/M02','plain':'plain'}
  for p in predictions:
   original=json.loads((root/source_paths[p['method']]/(p['question_id']+'.json')).read_text())
   assert original['status']==p['status'] and original['accepted_answers']==p['accepted_answers']
   p['backbone']=model
  allpred+=predictions;audits[model]=dict(counts)
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'audit.json').write_text(json.dumps(audits,indent=2)+'\n')
 # First label read occurs after both completed runs pass the audits above.
 labels=json.loads((PKG/'data/evaluation.json').read_text());per=[];totals={}
 for p in allpred:
  lab=labels[p['question_id']];v=score(p['accepted_answers'],p['status'],lab['groups'],lab['wrong_groups']);row=dict(backbone=p['backbone'],dataset=lab['dataset'],method=p['method'],question_id=p['question_id'],status=p['status'],**v,answers=json.dumps(p['accepted_answers'],ensure_ascii=False));per.append(row)
  k=(row['backbone'],row['dataset'],row['method']);t=totals.setdefault(k,dict(backbone=k[0],dataset=k[1],method=k[2],n=0,**{metric:0 for metric in v}));t['n']+=1
  for metric in v:t[metric]+=v[metric]
 for t in totals.values():
  for m in ['answer_match','strict_set','labeled_wrong']:t[m+'_pct']=100*t[m]/t['n']
 save_csv(OUT/'per_question.csv',per);save_csv(OUT/'summary.csv',list(totals.values()))
 comparison=[]
 for r in csv.DictReader((ROOT/'analysis/external_20261112/summary.csv').open()):
  if r['method'] in protocol['methods']:
   comparison.append(dict(backbone='Qwen2.5-7B-Instruct BF16 (historical)',dataset=r['dataset'],method=r['method'],n=r['n'],answer_match_pct=r['answer_match_pct'],strict_set_pct=r['strict_set_pct'],labeled_wrong=r['labeled_wrong'],abstained=r['abstained'],failed=r['failed']))
 for r in totals.values():comparison.append({k:r[k] for k in comparison[0]})
 save_csv(OUT/'with_historical_7b.csv',comparison)
 print(json.dumps(list(totals.values()),indent=2))
if __name__=='__main__':main()

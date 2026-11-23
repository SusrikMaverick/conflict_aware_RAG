"""Run frozen existing methods with one new backbone and freshly extracted cards."""
import os,json,hashlib,argparse,sys,time
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];PKG=ROOT/'deliverables/Conflict_QA_Submission_v7/code';PLAN=ROOT/'planning/backbones_20261113'
sys.path[:0]=[str(PKG),str(PKG/'src')]
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen3_32b','qwen35_35b'],required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=12);a=p.parse_args()
 protocol=json.loads((PLAN/'protocol.json').read_text());m=protocol['models'][a.model];out=ROOT/'runs/backbones_20261113'/a.model/('smoke' if a.smoke else 'full')
 digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
 for name,sha in protocol['hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
 os.environ.update(SIX_METHOD_ROOT=str(out),BACKBONE_MODEL_PATH=str(ROOT/'models'/m['name']),BACKBONE_ALIAS=a.model,BACKBONE_REVISION=m['revision'],BACKBONE_PROTOCOL=digest(protocol),BACKBONE_URL='http://127.0.0.1:'+str(m['port'])+'/v1')
 import run,card_pipeline as X,answer_review as W
 from backbone_client import BackboneClient,BackboneCompactClient
 from source_cards import SourceCards
 from real_runtime import write_json,load_plain_prompt
 from prompt_parts import docs,CONTRACT
 from answer_format import SCHEMA,parse
 # Keep method computations unchanged. Only substitute the generation client.
 X.Client=W.Client=BackboneClient
 X.CompactClient=W.CompactClient=BackboneCompactClient
 class SavedSourceCards(SourceCards):
  def run(self,row,method=None):
   value=super().run(row,method);doc=row['documents'][0]
   write_json(out/'document_cards'/row['question_id']/(digest(doc['doc_id'])+'.json'),dict(backbone=a.model,revision=m['revision'],input_sha256=digest(row),input=row,output=value));return value
 X.SourceCards=SavedSourceCards
 def check_server():
  import urllib.request
  with urllib.request.urlopen(os.environ['BACKBONE_URL']+'/models',timeout=15) as f:data=json.load(f)
  assert any(x['id']==a.model for x in data['data']),data
 run.check_servers=check_server
 run.METHODS=['ours_1024','FR4_M03','WR5_M02']
 input_path=PLAN/('synthetic_inputs.json' if a.smoke else 'inputs.json')
 data=run.validate_inputs(json.loads(input_path.read_text()))
 def identity(data,replay,metric):
  v=dict(mode='fresh',metric=metric,n=len(data),methods=run.METHODS,backbone=a.model,revision=m['revision'],protocol=digest(protocol),input_sha256=digest(data))
  return v|{'fingerprint':digest(v)}
 run.generation_identity=identity
 args=SimpleNamespace(workers=a.workers,input=str(input_path),dataset='all',limit=None,output=str(out),replay=False,metric='answer_boundary_v2')
 # Passing label-free custom inputs prevents the package from scoring mid-run.
 run.generate(args)
 client=BackboneClient()
 def plain(row):
  dest=out/'plain'/f"{row['question_id']}.json"
  if dest.exists():return json.loads(dest.read_text())
  result=dict(question_id=row['question_id'],method='plain',method_name='Plain RAG',accepted_answers=[])
  try:
   r=client.chat(load_plain_prompt().format(question=row['question'],documents=docs(row))+CONTRACT,1024,SCHEMA,1.05)
   answers=parse(r['text']);result.update(status='answered' if answers else 'abstain',accepted_answers=answers,request_key=r['key'],finish_reason=r['finish_reason'])
  except Exception as exc:result.update(status='error',error=str(exc))
  write_json(dest,result);return result
 with ThreadPoolExecutor(max_workers=a.workers) as pool:baseline=list(pool.map(plain,data))
 combined=json.loads((out/'predictions.json').read_text())+baseline
 write_json(out/'all_predictions.json',combined)
 write_json(out/'finished.json',dict(n=len(data),predictions=len(combined),protocol=digest(protocol),finished_at=time.time()))
 print('Finished',a.model,len(combined),flush=True)
if __name__=='__main__':main()

"""Model-specific local client; never read historical response/card caches."""
import os,json,threading,time,urllib.request,urllib.error
from pathlib import Path
from real_runtime import digest,write_json

class BackboneClient:
    compact=False
    tokenizer=None
    token_lock=threading.Lock()
    locks={}
    guard=threading.Lock()
    def __init__(self):
        from transformers import AutoTokenizer
        with self.guard:
            if BackboneClient.tokenizer is None:
                BackboneClient.tokenizer=AutoTokenizer.from_pretrained(os.environ['BACKBONE_MODEL_PATH'],local_files_only=True)
        self.cache=Path(os.environ['SIX_METHOD_ROOT'])/'request_cache'
        self.cache.mkdir(parents=True,exist_ok=True)
    def chat(self,prompt,max_tokens=768,json_mode=False,penalty=1.0):
        messages=[{'role':'user','content':prompt}]
        with self.token_lock:
            ids=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if len(ids)+max_tokens>32768:raise ValueError(f'context_overflow:{len(ids)}+{max_tokens}; no truncation')
        payload=dict(model=os.environ['BACKBONE_ALIAS'],messages=messages,temperature=0,top_p=1,max_tokens=max_tokens,seed=576,repetition_penalty=penalty,chat_template_kwargs={'enable_thinking':False})
        if isinstance(json_mode,dict):payload['response_format']={'type':'json_schema','json_schema':{'name':'decision','schema':json_mode}}
        elif json_mode:payload['response_format']={'type':'json_object'}
        from answer_format import SCHEMA
        if self.compact and isinstance(json_mode,dict) and json_mode!=SCHEMA:
            payload.pop('response_format')
            payload['structured_outputs']={'json':json_mode,'disable_any_whitespace':True}
        revision=os.environ['BACKBONE_REVISION'];key=digest({'revision':revision,'protocol':os.environ['BACKBONE_PROTOCOL'],'payload':payload})
        path=self.cache/(key+'.json')
        with self.guard:lock=self.locks.setdefault(key,threading.Lock())
        with lock:
            if path.exists():return json.loads(path.read_text())|{'cache_hit':True}
            start=time.monotonic();errors=[]
            for attempt in range(3):
                try:
                    request=urllib.request.Request(os.environ['BACKBONE_URL']+'/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(request,timeout=1800) as f:out=json.load(f)
                    choice=out['choices'][0];text=choice['message']['content']
                    if not isinstance(text,str):raise ValueError('Missing assistant content')
                    result=dict(key=key,model_revision=revision,request=payload,response=out,text=text,finish_reason=choice['finish_reason'],usage=out.get('usage',{}),preflight_tokens=len(ids),prompt_token_ids=ids,latency_seconds=time.monotonic()-start,attempts=attempt+1,retry_errors=errors,cache_hit=False,is_model_inference=True)
                    write_json(path,result);return result
                except urllib.error.HTTPError as exc:
                    detail=exc.read().decode()[:1200]
                    if exc.code<500:raise RuntimeError(f'HTTP {exc.code}: {detail}') from exc
                    errors.append(detail)
                except (urllib.error.URLError,TimeoutError) as exc:errors.append(str(exc))
                time.sleep(2*(attempt+1))
            raise RuntimeError('Backend failed: '+str(errors))

class BackboneCompactClient(BackboneClient):
    compact=True

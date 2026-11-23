"""TruthfulRAG on the frozen three-dataset inputs; no gold labels are loaded."""
import argparse
import asyncio
import contextvars
import functools
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'planning/external_20261112'
RUN = ROOT / 'runs/truthful_20261112'
UPSTREAM = ROOT / 'work/external_20261112/upstream'
PACKAGE = ROOT / 'deliverables/Conflict_QA_Submission_v7/code'
os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(ROOT/'.cache/tiktoken'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('NLTK_DATA', str(PACKAGE/'assets/nltk'))
os.environ['COSINE_THRESHOLD'] = '0.2'
os.environ['TOP_K'] = '10'
sys.path[:0] = [str(UPSTREAM), str(UPSTREAM/'truthfulrag'), str(PACKAGE/'src')]

import httpx
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from truthfulrag import TruthfulRAG
import truthfulrag.modules as modules
from truthfulrag.llm.backend import LLMBackend
from truthfulrag.llm.hf import hf_embed
from truthfulrag.utils import EmbeddingFunc
from truthfulrag.prompts import PromptGenerator
from answer_format import SCHEMA, parse
from prompt_parts import CONTRACT

torch.set_num_threads(2)
CURRENT = contextvars.ContextVar('current')
STAGE = contextvars.ContextVar('stage', default='unknown')
MODEL_REV = 'a09a35458c702b33eeacc393d103063234e8bc28'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def source_hashes():
    files = sorted(UPSTREAM.rglob('*.py')) + [Path(__file__), PLAN/'requirements.lock', PACKAGE/'data/inputs.json', PACKAGE/'src/answer_format.py', PACKAGE/'src/prompt_parts.py', PACKAGE/'evaluation.py', PACKAGE/'src/metrics.py', ROOT/'models/Qwen2.5-7B-Instruct/config.json', ROOT/'models/all-MiniLM-L6-v2/config.json']
    return {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


class Transport:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=600)
        self.semaphore = asyncio.Semaphore(40)
        self.locks = {}
        self.tokenizer = AutoTokenizer.from_pretrained(ROOT/'models/Qwen2.5-7B-Instruct', local_files_only=True)
        self.endpoints = ['http://127.0.0.1:8976/v1/chat/completions', 'http://127.0.0.1:8978/v1/chat/completions']
        self.next_endpoint = 0

    async def call(self, prompt, system_prompt=None, history_messages=None, **kwargs):
        messages = ([{'role':'system','content':system_prompt}] if system_prompt else []) + (history_messages or []) + [{'role':'user','content':prompt}]
        payload = {'model':'nlp576-qwen7b','messages':messages,'temperature':0,'top_p':1.0,'seed':576,'max_tokens':kwargs.get('max_tokens',1000),'repetition_penalty':1.05}
        for key in ('logprobs','top_logprobs','response_format'):
            if key in kwargs:
                payload[key] = kwargs[key]
        n = len(self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
        if n + payload['max_tokens'] > 32768:
            raise ValueError(f'context_overflow:{n}+{payload["max_tokens"]}')
        key = digest({'revision':MODEL_REV,'runtime':'vllm0.8.5.post1-bf16','payload':payload})
        path = RUN/'requests'/(key+'.json.gz')
        lock = self.locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = path.exists()
            if cached:
                with gzip.open(path,'rt') as f:
                    record = json.load(f)
            else:
                start = time.time()
                async with self.semaphore:
                    endpoint = self.endpoints[self.next_endpoint % len(self.endpoints)]
                    self.next_endpoint += 1
                    response = await self.client.post(endpoint,json=payload)
                    response.raise_for_status()
                    result = response.json()
                record = {'key':key,'request':payload,'response':result,'preflight_tokens':n,'seconds':time.time()-start,'model_revision':MODEL_REV,'endpoint':endpoint}
                path.parent.mkdir(parents=True,exist_ok=True)
                tmp=path.with_suffix('.tmp')
                with gzip.open(tmp,'wt',encoding='utf-8') as f:
                    json.dump(record,f,ensure_ascii=False)
                tmp.replace(path)
        choice = record['response']['choices'][0]
        CURRENT.get()['requests'].append({'key':key,'stage':STAGE.get(),'cache_hit':cached,'finish_reason':choice['finish_reason'],'usage':record['response'].get('usage',{}),'seconds':record['seconds']})
        if kwargs.get('logprobs'):
            content = choice.get('logprobs',{}).get('content')
            if not content:
                raise ValueError('Missing token probabilities for entropy filtering')
            return [x['top_logprobs'] for x in content]
        return (choice['message']['content'] or '').strip()


def backend_class(transport):
    class AuditedBackend(LLMBackend):
        def __init__(self, backend_type, model_name, **config):
            assert backend_type == 'openai'
            self.backend_type = backend_type
            self.model_name = model_name
            self.backend_config = config
            self.default_sampling_params = {'max_tokens':1000,'top_p':1.0}
            self.complete_fn = transport.call
    return AuditedBackend


class Experiment:
    def __init__(self):
        self.transport = Transport()
        modules.LLMBackend = backend_class(self.transport)
        # The upstream constructors here are unused, but cache them without changing their behavior.
        modules.SentenceTransformer = functools.lru_cache(maxsize=1)(functools.partial(modules.SentenceTransformer, device='cpu'))
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(ROOT/'models/all-MiniLM-L6-v2',local_files_only=True)
        self.encoder = AutoModel.from_pretrained(ROOT/'models/all-MiniLM-L6-v2',local_files_only=True).eval()
        # A closure survives upstream dataclasses.asdict without deep-copying HTTP locks.
        self.embedding = EmbeddingFunc(384,1000,lambda texts: self.embed(texts))

    async def embed(self,texts):
        # Match upstream unmasked mean pooling exactly; do not substitute SentenceTransformer pooling.
        return await hf_embed(texts,self.encoder_tokenizer,self.encoder)

    async def one(self,row,pilot=False):
        assert set(row)=={'question_id','question','documents'}
        assert all(set(d)=={'doc_id','text'} for d in row['documents'])
        qid=row['question_id']
        directory=RUN/('pilot' if pilot else 'outputs')/qid
        out_path=directory/'prediction.json'
        if out_path.exists():
            old=json.loads(out_path.read_text())
            assert old['input_hash']==digest(row)
            return old
        output={'question_id':qid,'method':'truthfulrag_label_free','input_hash':digest(row),'status':'pending','accepted_answers':[],'requests':[]}
        CURRENT.set(output)
        start=time.time()
        directory.mkdir(parents=True,exist_ok=True)
        sample={'id':qid,'question':row['question'],'context':'\n'.join(f'Document {i}: {d["text"]}' for i,d in enumerate(row['documents'],1))}
        # A separate graph for every question prevents cross-question contamination.
        try:
            rag=TruthfulRAG(dataset=[],backend_type='openai',model_name='nlp576-qwen7b',similarity_model=str(ROOT/'models/all-MiniLM-L6-v2'),threshold=1,embedding_func=self.embedding,working_dir=str(directory/'graph'),output_dir=str(directory))
            STAGE.set('graph')
            await rag.make_knowledge_graph(sample)
            STAGE.set('retrieve')
            elements=await rag.knowledge_graph_retrieve(sample)
            if not isinstance(elements,list) or any(not isinstance(e,str) for e in elements):
                raise ValueError('Invalid graph retrieval output')
            output['retrieved_elements']=elements
            STAGE.set('entropy')
            filtered=await rag.entropy_based_filter(sample=sample,elements=elements[:30],threshold=1)
            filtered=filtered[:10]
            output['filtered_elements']=filtered
            pg=PromptGenerator(llm_type='openai',task='qa-cot')
            prompt=pg.generate_qa_prompt_normal_cot(context=sample['context'],question=sample['question'],facts='\n\n'.join(filtered)) + CONTRACT
            STAGE.set('final')
            raw=await self.transport.call(prompt,system_prompt=pg.system_prompt,max_tokens=1024,response_format={'type':'json_schema','json_schema':{'name':'decision','schema':SCHEMA}})
            output['raw_output']=raw
            try:
                output['accepted_answers']=parse(raw)
                output['status']='answered' if output['accepted_answers'] else 'abstain'
            except (ValueError,TypeError):
                output['status']='parse_error'
        except Exception as exc:
            output.update(status='error',error=str(exc),traceback=traceback.format_exc())
        output['wall_seconds']=time.time()-start
        write(out_path,output)
        return output


def pilot_rows():
    specs=[
        ('agreement','When did the Arbor Observatory open?',['The Arbor Observatory opened in 1982.','The Arbor Observatory began public operations in 1982.']),
        ('conflict','When did the Arbor Observatory open?',['The Arbor Observatory opened in 1982.','The Arbor Observatory opened in 1991.']),
        ('multiple','Which cities have an Arbor Museum?',['The Arbor Museum in Lanton opened in 1990.','Another Arbor Museum is located in Westmere.']),
        ('unsupported','When did the Arbor Observatory open?',['The Arbor Observatory has a copper roof.'])]
    return [{'question_id':'synthetic-'+name,'question':q,'documents':[{'doc_id':f'd{i}','text':d} for i,d in enumerate(docs)]} for name,q,docs in specs]


async def execute(pilot):
    assert os.environ.get('PYTHONHASHSEED')=='576', 'Use the frozen Python hash seed for upstream set iteration'
    if not pilot:
        protocol=json.loads((PLAN/'protocol.json').read_text())
        assert protocol['hashes']==source_hashes(),'Frozen sources changed'
    rows=pilot_rows() if pilot else json.loads((PACKAGE/'data/inputs.json').read_text())
    if not pilot:
        assert len(rows)==1100 and len({r['question_id'] for r in rows})==1100
    exp=Experiment()
    semaphore=asyncio.Semaphore(32)
    completed=0
    async def one(row):
        nonlocal completed
        async with semaphore:
            result=await exp.one(row,pilot)
            completed+=1
            write(RUN/'progress.json',{'pilot':pilot,'completed':completed,'total':len(rows),'last':row['question_id'],'last_status':result['status'],'timestamp':time.time()})
            print('PROGRESS',completed,len(rows),result['status'],result.get('error',''),flush=True)
            return result
    results=await asyncio.gather(*(one(row) for row in rows))
    await exp.transport.client.aclose()
    write(RUN/('pilot_finished.json' if pilot else 'execution_finished.json'),{'count':len(results),'statuses':{s:sum(r['status']==s for r in results) for s in sorted({r['status'] for r in results})},'timestamp':time.time()})


def freeze():
    assert not (PLAN/'protocol.json').exists()
    pilot=json.loads((RUN/'pilot_finished.json').read_text())
    assert pilot['count']==4 and not any(pilot['statuses'].get(s,0) for s in ['error','parse_error'])
    write(PLAN/'protocol.json',{'method':'TruthfulRAG (label-free adaptation)','model_revision':MODEL_REV,'datasets':{'RAMDocs':500,'RGB':100,'ConflictBank':500},'entropy_tokens':32,'threshold':1,'retrieved_limit':30,'filtered_limit':10,'kg_query_tokens':1000,'final_tokens':1024,'temperature':0,'repetition_penalty':1.05,'seed':576,'question_workers':8,'http_concurrency':12,'hashes':source_hashes(),'evaluation':'answer_boundary_v2; all failures included; scoring only after execution','status':'Frozen before benchmark inference','comparison':'Exploratory; historical 13-method outputs reused, not relabeled as new inference'})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['pilot','freeze','run'])
    args=parser.parse_args()
    if args.command=='freeze':freeze()
    else:asyncio.run(execute(args.command=='pilot'))

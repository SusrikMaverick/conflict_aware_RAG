"""Recompute the published table from saved answers; no model service required."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
CORE = SOURCE / 'deliverables/Conflict_QA_Submission_v7/code'
sys.path.insert(0, str(CORE))
from evaluation import score, METRIC_VERSION

NAMES = {
    'ours_1024': 'Docs + Cards (Ours)', 'FR4_M03': 'Evidence Pruning (Ours)',
    'WR5_M02': 'Memory + Cards (Ours)', 'WR3_M01': 'Context Review (Ours)',
    'WR2_M08': 'Answer Agreement (Ours)', 'WR1_M05': 'Dual Review (Ours)',
    'plain_1024': 'Plain RAG', 'madam': 'MADAM-RAG', 'self_reflection': 'Self-reflection',
    'astute_structured1024': 'Astute RAG (structured)', 'faithfulrag_adaptation': 'FaithfulRAG (adapted)',
    'trust': 'TrustRAG (adapted)', 'robust_keyword': 'RobustRAG (keyword)',
    'evidence_map': 'Teammate Evidence Map', 'evidence_map_memory': 'Teammate Evidence Map + Memory',
    'teammate_plain': 'Teammate Plain RAG (128 tokens)',
    'truthfulrag_label_free': 'TruthfulRAG (label-free)',
    'cards': 'Docs + Cards (Ours)', 'memory_cards': 'Memory + Cards (Ours)',
    'evidence_prune': 'Evidence Pruning (Ours)', 'plain': 'Plain RAG',
}
COUNTS = ['n', 'answer_match', 'strict_set', 'labeled_wrong', 'failed', 'abstained']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def predictions():
    for name in ['predictions.jsonl.gz', 'baselines/predictions.jsonl.gz']:
        with gzip.open(CORE / 'data' / name, 'rt', encoding='utf-8') as stream:
            for line in stream:
                yield 'Qwen2.5-7B-Instruct', json.loads(line)
    for path in sorted((SOURCE/'runs/teammate_20261115/full').glob('*/questions/*.json')):
        record = read_json(path)
        # The method directory is the stable method identifier.
        record['method'] = path.parents[1].name
        yield 'Qwen2.5-7B-Instruct', record
    for path in sorted((SOURCE/'runs/truthful_20261112/outputs').glob('*/prediction.json')):
        yield 'Qwen2.5-7B-Instruct', read_json(path)
    for model, label in [('qwen3_32b', 'Qwen3-32B'), ('qwen35_35b', 'Qwen3.5-35B-A3B')]:
        for record in read_json(SOURCE/'runs/backbones_20261113'/model/'full/all_predictions.json'):
            yield label, record


def rescore(output):
    labels = read_json(CORE/'data/evaluation.json')
    totals = defaultdict(lambda: defaultdict(int))
    seen = set()
    covered_ids = defaultdict(set)
    per_question = []
    for model, record in predictions():
        qid, method = record['question_id'], NAMES[record['method']]
        identity = model, method, qid
        if identity in seen:
            raise ValueError('Duplicate prediction: ' + str(identity))
        seen.add(identity)
        covered_ids[model, method].add(qid)
        label = labels[qid]
        dataset = {'ConflictBank': 'ConflictBank-500'}.get(label['dataset'], label['dataset'])
        values = score(record['accepted_answers'], record['status'], label['groups'], label['wrong_groups'])
        total = totals[model, dataset, method]
        total['n'] += 1
        for name, value in values.items():
            total[name] += value
        per_question.append(dict(model=model, dataset=dataset, method=method, question_id=qid,
                                 status=record['status'], accepted_answers=json.dumps(record['accepted_answers'], ensure_ascii=False), **values))
    with (ROOT/'results/All_Full_Benchmark_Results.csv').open(encoding='utf-8', newline='') as stream:
        reference = list(csv.DictReader(stream))
    expected = {(r['model'], r['dataset'], r['method']): r for r in reference}
    if len(expected) != len(reference):
        raise ValueError('Duplicate summary row in the reference table')
    if set(totals) != set(expected):
        raise ValueError('Missing or extra model/dataset/method rows')
    for key, ids in covered_ids.items():
        if ids != set(labels):
            raise ValueError('Incomplete question coverage: ' + str(key))
    rows = []
    for ref in reference:
        key = ref['model'], ref['dataset'], ref['method']
        actual = totals[key]
        if ref['metric'] != METRIC_VERSION:
            raise ValueError('Reference metric does not match the evaluator')
        for name in COUNTS:
            if actual[name] != int(ref[name]):
                raise ValueError(f'Mismatch: {key} {name}: {actual[name]} != {ref[name]}')
        row = dict(ref)
        row.update({name: actual[name] for name in COUNTS})
        for name in ['answer_match', 'strict_set', 'labeled_wrong']:
            row[name+'_pct'] = 100 * actual[name] / actual['n']
            if not math.isclose(float(ref[name+'_pct']), row[name+'_pct'], abs_tol=1e-10):
                raise ValueError('Reference percentage does not match its count: ' + str(key))
        rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in [('All_Full_Benchmark_Results.csv', rows), ('Per_Question.csv', per_question)]:
        with (output/name).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(f'All {len(rows)} summary rows matched; {len(per_question)} predictions rescored.')


def verify():
    manifest = read_json(ROOT/'manifest.json')
    for name, digest in manifest.items():
        path = ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Missing or changed file: '+name)
    print(f'All {len(manifest)} packaged files verified.')


def make_pdf(output):
    # Keep the archived layout, using the scores recomputed in this run.
    script = SOURCE/'work/build_selected_results_tables.py'
    code = runpy.run_path(str(script))
    destination = output/'Current_Results_Tables.pdf'
    output.mkdir(parents=True, exist_ok=True)
    code['main'].__globals__['OUT'] = destination
    code['main'].__globals__['FULL'] = output/'All_Full_Benchmark_Results.csv'
    code['main']()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'score', 'pdf'])
    parser.add_argument('--output', type=Path, default=ROOT/'outputs')
    args = parser.parse_args()
    target = args.output.resolve()
    if args.command != 'verify' and (target == ROOT or any(target.is_relative_to(ROOT/p) for p in ['source', 'results'])):
        parser.error('Choose an output directory outside source/ and results/.')
    if args.command == 'verify':
        verify()
    else:
        rescore(target)
        if args.command == 'pdf':
            make_pdf(target)


if __name__ == '__main__':
    main()

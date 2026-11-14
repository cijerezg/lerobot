"""Merge one component's v3 batch verdicts, then union them with the v2 verdicts for finalize.

    uv run python lerobot/examples/dataset/diverse_robot_dataset/molmoact_review/union_verdicts.py household ['{"<episode>": {<patch>}}']

Writes outputs/_annotation/diverse_v3_review/<component>/verdicts_v3.json (the v3 round alone, same checks as
merge_verdicts.py: full coverage of the v3 candidates, unsure -> reject, events only with quality <= 2) and
verdicts_union.json (every v2 record verbatim plus the v3 records, each stamped `round`), which is what
`prepare_molmoact.py finalize --verdicts` takes. Refuses when any union candidate lacks a verdict.
"""
import glob, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
REVIEW = ROOT / 'outputs/_annotation/diverse_v3_review'
V2_VERDICTS = {'household': ROOT / 'outputs/_annotation/diverse_v2_expansion_review/household/verdicts.json',
               'tabletop': ROOT / 'outputs/diverse_robot_dataset_v3_build/molmoact/review/tabletop/verdicts.json'}
KINDS = {'failed_close', 'slip', 'drop', 'knock', 'wrong_target'}
REVIEW_DATE = '2026-09-19'
BASIS = 'three 12-tile contact sheets per episode (two external + wrist), model review under the ReBot rubric, unclear = reject'

component = sys.argv[1]; overrides = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
root = REVIEW / component
candidates = json.load(open(root / 'candidates.json'))['candidates']
expected = {str(x['episode_index']) for x in candidates if x['round'] == 'v3'}
episodes = {}; reviewers = set()
for path in sorted(glob.glob(str(root / 'verdicts_batch*.json'))):
    d = json.load(open(path)); reviewers.add(d.get('reviewer'))
    for k, v in d['episodes'].items():
        assert k not in episodes, f'duplicate verdict {k}'; v['batch_file'] = Path(path).name; v.setdefault('reviewer', d.get('reviewer')); episodes[k] = v
missing = sorted(expected - set(episodes), key=int); extra = sorted(set(episodes) - expected, key=int)
if missing or extra:
    raise SystemExit(f'Incomplete v3 coverage: missing={missing}, extra={extra}; nothing written')
for k, patch in overrides.items():
    episodes[k].update(patch); episodes[k]['override'] = True
for k, v in episodes.items():
    assert isinstance(v['accept'], bool), k
    if v['accept'] or v.get('quality') is not None:
        assert type(v.get('quality')) is int and 1 <= v['quality'] <= 5, k
    for e in v.get('mistake_events', []): assert e['kind'] in KINDS, k
    if v['accept'] and v.get('confidence') == 'unsure': v['accept'] = False; v['notes'] = 'unsure -> reject; ' + v.get('notes', '')
    if v.get('mistake_events') and int(v['quality']) > 2: raise SystemExit(f"{k}: events with quality {v['quality']}")
v3 = {'reviewer': sorted(reviewers), 'review_date': REVIEW_DATE, 'round': 'v3', 'review_basis': BASIS,
      'episodes': dict(sorted(episodes.items(), key=lambda kv: int(kv[0])))}
json.dump(v3, open(root / 'verdicts_v3.json', 'w'), indent=1)

v2 = json.load(open(V2_VERDICTS[component]))
union = {k: {**v, 'round': 'v2'} for k, v in v2['episodes'].items()}
for k, v in v3['episodes'].items():
    assert k not in union, f'episode {k} has a verdict in both rounds'; union[k] = {**v, 'round': 'v3'}
expected_union = {str(x['episode_index']) for x in candidates}
missing = sorted(expected_union - set(union), key=int); extra = sorted(set(union) - expected_union, key=int)
if missing or extra:
    raise SystemExit(f'Incomplete union coverage: missing={missing}, extra={extra}; verdicts_union.json not written')
out = {'reviewer': sorted(set(v2['reviewer']) | set(v3['reviewer'])), 'review_date': REVIEW_DATE, 'review_basis': BASIS,
       'rounds': {'v2': {'verdicts': str(V2_VERDICTS[component].relative_to(ROOT)), 'review_date': v2['review_date'], 'episodes': len(v2['episodes'])},
                  'v3': {'verdicts': str((root / 'verdicts_v3.json').relative_to(ROOT)), 'review_date': REVIEW_DATE, 'episodes': len(v3['episodes'])}},
       'episodes': dict(sorted(union.items(), key=lambda kv: int(kv[0])))}
json.dump(out, open(root / 'verdicts_union.json', 'w'), indent=1)
for name, table in (('v3', v3['episodes']), ('union', union)):
    acc = [k for k, v in table.items() if v['accept']]
    print(name, 'episodes', len(table), 'accepted', len(acc), 'quality', dict(sorted(Counter(int(v['quality']) for v in table.values() if v['accept']).items())),
          'events', sum(len(v.get('mistake_events', [])) for v in table.values() if v['accept']))
print('v3 rejected', {k: v.get('notes', '')[:80] for k, v in v3['episodes'].items() if not v['accept']})

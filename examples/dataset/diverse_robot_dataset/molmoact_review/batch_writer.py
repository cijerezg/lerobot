"""Shared writer for v3 batch verdict files (household | tabletop); each batch script supplies explicit decisions.

    from batch_writer import write_batch
    write_batch('household', 0, D)   # D = {episode_index: dict(quality=..., notes=..., evidence=..., ...)}
"""
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
REVIEW = ROOT / 'outputs/_annotation/diverse_v3_review'
KINDS = {'failed_close', 'slip', 'drop', 'knock', 'wrong_target'}


def write_batch(component, batch_index, decisions, *, reviewer='Claude', review_date='2026-09-19'):
    """decisions: {episode_index: dict(quality, notes, evidence, accept=True, outcome='success',
    confidence='sure', task=None, events=None, status='reviewed', extra_sheets=None)}. Coverage against batches.json is enforced."""
    cache = REVIEW / component
    batches = json.loads((cache / 'batches.json').read_text())
    expected = batches[batch_index]
    assert sorted(decisions) == sorted(expected), (sorted(set(expected) - set(decisions)), sorted(set(decisions) - set(expected)))
    rows = {}
    for ep in expected:
        d = dict(decisions[ep])
        files = [cache / f'episode_{ep:06d}.{cam}.jpg' for cam in ('primary', 'secondary', 'wrist')]
        assert all(p.is_file() for p in files), ep
        extra = d.pop('extra_sheets', None) or []   # supplemental sheets (terminal/dense) that the decision cites
        extra = [cache / e for e in extra]
        assert all(e.is_file() for e in extra), (ep, extra)
        events = d.pop('events', None) or []
        for e in events:
            assert e['kind'] in KINDS and 0 <= e['start_s'] < e['end_s'], (ep, e)
        accept = d.pop('accept', True); quality = d.pop('quality'); confidence = d.pop('confidence', 'sure')
        outcome = d.pop('outcome', 'success'); status = d.pop('status', 'reviewed'); task = d.pop('task', None)
        if accept:
            assert type(quality) is int and 1 <= quality <= 5 and confidence == 'sure', ep
            assert (quality <= 2) == bool(events) or (quality == 1 and events), (ep, quality, events)
            assert not (events and quality > 2), ep
        row = dict(accept=accept, outcome=outcome, quality=quality, mistake_events=events,
                   notes=d.pop('notes'), evidence=d.pop('evidence'), confidence=confidence,
                   review_status=status, reviewer=reviewer, quality_provenance='model_reviewed_rebot_rubric',
                   sheets_read=[str(p.relative_to(ROOT)) for p in files],
                   sheet_sha256={str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in files})
        if task:
            row['task'] = task
        if extra:
            row['additional_evidence_read'] = [{'path': str(e.relative_to(ROOT)), 'sha256': sha256(e.read_bytes()).hexdigest()} for e in extra]
        assert not d, (ep, d)
        rows[str(ep)] = row
    out = {'reviewer': reviewer, 'review_date': review_date, 'source': f'molmoact_{component}', 'round': 'v3', 'batch_index': batch_index,
           'review_provenance': 'model_reviewed', 'quality_provenance': 'model_reviewed_rebot_rubric',
           'status': f'batch_{batch_index}_of_{len(batches)}_complete',
           'limitations': 'Three 12-tile contact sheets per episode; events between tiles cannot be ruled out. Outcomes are judged visually from the last tiles.',
           'episodes': rows}
    path = cache / f'verdicts_batch{batch_index}.json'
    assert not path.exists(), path
    path.write_text(json.dumps(out, indent=1) + '\n')
    acc = [k for k, v in rows.items() if v['accept']]
    print(json.dumps({'component': component, 'batch': batch_index, 'episodes': len(rows), 'accepted': len(acc),
                      'quality': dict(sorted(Counter(rows[k]['quality'] for k in acc).items())),
                      'events': sum(len(rows[k]['mistake_events']) for k in acc),
                      'rejected_or_withheld': {k: rows[k]['review_status'] for k in rows if not rows[k]['accept']}}))
    return path

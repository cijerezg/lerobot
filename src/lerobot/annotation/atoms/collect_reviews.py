"""Collect the visual reviews into verdicts: manual overrides (Codex visual corrections) win,
then Claude agent reviews (agent_reviews/). The Gemma drafts in local_vlm_reviews/ are
not consulted (user decision 2026-09-09: no machine drafts in the verdicts)."""
import argparse,collections,json
from pathlib import Path
from lerobot.annotation.atoms.atoms_common import REVIEW,WORK,load_corpus,read_jsonl
from lerobot.annotation.atoms.verdict import VERDICTS,check_verdict,validator


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');args=ap.parse_args()
    episodes,parents=load_corpus();proposals=read_jsonl(WORK/'proposals.jsonl')
    by_ep=collections.defaultdict(dict)
    for p in proposals:by_ep[p['episode_id']][p['parent_interval_index']]=p
    report={'complete_episodes':0,'reviewed_parents':0,'missing':[],'errors':{},'unsure':[],'source_counts':{},'manual_parents':0,'agent_parents':0}
    for eid,ep in episodes.items():
        override_path=REVIEW/'manual_overrides'/f'{eid}.json'
        overrides=json.loads(override_path.read_text()) if override_path.exists() else {'parents':[]}
        overrides_by_parent={r['parent_interval_index']:r for r in overrides['parents']}
        agent_path=REVIEW/'agent_reviews'/f'{eid}.json'
        agent=json.loads(agent_path.read_text()) if agent_path.exists() else {'parents':[]}
        agent_by_parent={r['parent_interval_index']:r for r in agent['parents']}
        verdict={'episode_id':eid,'reviewer':'Claude visual review'+(' + Codex visual corrections' if overrides_by_parent else ''),'parents':[],'review_provenance':[]}
        for pidx,proposal in by_ep[eid].items():
            if pidx in overrides_by_parent:
                pv=overrides_by_parent[pidx];evidence={'parent_interval_index':pidx,'reviewer':overrides['reviewer'],'path':str(override_path),'images':overrides['images']};report['manual_parents']+=1
            elif pidx in agent_by_parent:
                pv=agent_by_parent[pidx];evidence={'parent_interval_index':pidx,'reviewer':agent['reviewer'],'path':str(agent_path),'images':agent['images']};report['agent_parents']+=1
            else:report['missing'].append(f'{eid} P{pidx}');continue
            report['reviewed_parents']+=1
            if pv['confidence']=='unsure':report['unsure'].append({'episode_id':eid,'parent_interval_index':pidx,'note':pv['note']})
            verdict['parents'].append(pv);verdict['review_provenance'].append(evidence)
        if len(verdict['parents'])!=len(by_ep[eid]):continue
        errors=check_verdict(verdict,by_ep[eid],ep,{r['interval_index']:r for r in parents[eid]})
        for pv in verdict['parents']:
            for a in pv['atoms']:errors+=validator.subtask_errors(dict(a,subtask=validator.render_subtask(a)),eid)
        if errors:report['errors'][eid]=errors;continue
        report['complete_episodes']+=1
        report['source_counts'][ep['source']]=report['source_counts'].get(ep['source'],0)+1
        if args.write:
            VERDICTS.mkdir(parents=True,exist_ok=True);(VERDICTS/f'{eid}.json').write_text(json.dumps(verdict,indent=1))
    (REVIEW/'review_progress.json').write_text(json.dumps(report,indent=1))
    print(json.dumps({k:v for k,v in report.items() if k not in ('missing','errors','unsure')},indent=1))
    print('missing',len(report['missing']),'errors',len(report['errors']),'unsure',len(report['unsure']))
if __name__=='__main__':main()

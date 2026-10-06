"""Apply the authoritative parent-quality ceiling to machine review suggestions.

This does not infer visual quality. Where a model proposed an impermissible higher
grade, retain the already reviewed parent grade and record the rejected suggestion.
"""
import json
from pathlib import Path
from lerobot.annotation.atoms.atoms_common import REVIEW,WORK,load_corpus,read_jsonl
from lerobot.annotation.atoms.verdict import check_verdict,validator


def main():
    episodes,parents=load_corpus();proposals={(p['episode_id'],p['parent_interval_index']):p for p in read_jsonl(WORK/'proposals.jsonl')}
    parent_rows={(eid,r['interval_index']):r for eid,rows in parents.items() for r in rows};count=0
    for path in (REVIEW/'local_vlm_reviews').glob('*_P*.json'):
        r=json.loads(path.read_text())
        if not r.get('verdict'):continue
        key=(r['episode_id'],r['parent_interval_index']);p=proposals[key];parent=parent_rows[key];q=int(parent['quality']);pv=r['verdict'];changes=[]
        event_driven=q<=2 and bool(parent['mistake_events'])
        for atom in pv['atoms']:
            proposed=atom.get('quality')
            if proposed is not None and proposed>q and not event_driven:
                changes.append({'span':[atom['start_timestep'],atom['end_timestep_exclusive']],'rejected_quality':proposed,'retained_parent_quality':q})
                atom['quality']=None;atom['quality_note']=''
        if not changes:continue
        r.setdefault('parent_quality_ceiling_enforced',[]).extend(changes)
        errors=check_verdict({'episode_id':key[0],'parents':[pv]},{key[1]:p},episodes[key[0]],{key[1]:parent})
        for a in pv['atoms']:errors+=validator.subtask_errors(dict(a,subtask=validator.render_subtask(a)),key[0])
        r['errors']=errors;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=1));tmp.replace(path);count+=1
    print('Retained authoritative parent quality in',count,'reviews')
if __name__=='__main__':main()

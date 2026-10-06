"""Build a local atomic annotation browser over the existing corpus videos."""
import json,os
from pathlib import Path
from lerobot.annotation.atoms.atoms_common import CORPUS,REVIEW,read_jsonl


def main():
    episodes={r['episode_id']:r for r in read_jsonl(CORPUS/'episodes.jsonl')}
    rows=read_jsonl(CORPUS/'subtask_atoms.jsonl')
    by_episode={}
    for r in rows:by_episode.setdefault(r['episode_id'],[]).append(r)
    data=[]
    speed_path=CORPUS/'speed_atoms.jsonl'
    speeds={(r['episode_id'],r['parent_interval_index'],r['atom_index']):r['speed'] for r in read_jsonl(speed_path)} if speed_path.exists() else {}
    for eid,atoms in by_episode.items():
        ep=episodes[eid];videos=CORPUS/ep['directory']/'videos'
        ext=next(p for p in sorted(videos.glob('*.mp4')) if 'wrist' not in p.name)
        wrist=next(p for p in sorted(videos.glob('*.mp4')) if 'wrist' in p.name)
        for atom in atoms:atom['speed']=speeds.get((eid,atom['parent_interval_index'],atom['atom_index']))
        data.append({'id':eid,'source':ep['source'],'task':ep['task'],'external':os.path.relpath(ext,REVIEW),'wrist':os.path.relpath(wrist,REVIEW),'atoms':atoms})
    payload=json.dumps(data).replace('</','<\\/')
    html='''<!doctype html><meta charset="utf-8"><title>Atomic subtask review</title>
<style>body{font:16px system-ui;background:#15181d;color:#e8edf4;margin:24px}h1{font-size:26px}select,input,button{font:inherit;padding:8px;background:#242b34;color:inherit;border:1px solid #596573;border-radius:5px}#videos{display:flex;gap:8px;margin-top:18px}video{width:49.5%;background:black}table{width:100%;border-collapse:collapse;margin-top:20px}td,th{text-align:left;padding:8px;border-bottom:1px solid #3c4552}tr.active{background:#30435b}tr{cursor:pointer}#description{line-height:1.6}#note{color:#c1c9d5}a{color:#9dcaff}.filters{display:flex;gap:12px;flex-wrap:wrap}</style>
<h1>Corpus atomic subtask review</h1><p>One action and one object per annotation. Select an episode, then an atom to play its interval in real time. External camera left; wrist right.</p>
<div class="filters"><select id="source"><option value="">All sources</option></select><input id="search" placeholder="Search task or episode"><select id="episode"></select><label><input type="checkbox" id="uncertain">Uncertain reviews only</label></div>
<div id="videos"><video id="external" controls preload="metadata"></video><video id="wrist" controls preload="metadata" muted></video></div>
<p><button id="previous">Previous atom</button> <button id="replay">Replay atom</button> <button id="next">Next atom</button></p><div id="description"></div><p id="note"></p>
<table><thead><tr><th>Parent / atom</th><th>Subtask</th><th>Start</th><th>Duration</th><th>Quality</th><th>Speed</th><th>Confidence</th></tr></thead><tbody id="atoms"></tbody></table>
<p>Machine visual annotations with recorded inspection and corrections. Parent quality and events are preserved under the annotation rules. <a href="../../../migration/subtask_atoms_2026-09-08/COMPLETION.md">Dataset and validation details</a></p>
<script>const data=PAYLOAD;const $=id=>document.getElementById(id);let current=null,index=0;
for(const source of [...new Set(data.map(e=>e.source))]){let o=new Option(source,source);$('source').add(o)}
function filter(){let q=$('search').value.toLowerCase();let selected=data.filter(e=>(!$('source').value||e.source===$('source').value)&&(!q||(e.id+' '+e.task).toLowerCase().includes(q))&&(!$('uncertain').checked||e.atoms.some(a=>a.confidence==='unsure')));$('episode').replaceChildren(...selected.map(e=>new Option(e.id,e.id)));load()}
function load(){current=data.find(e=>e.id===$('episode').value);$('atoms').replaceChildren();if(!current){$('description').textContent='No matching episodes';$('external').pause();$('wrist').pause();return}for(const key of ['external','wrist'])$(key).src=current[key];for(const [i,a]of current.atoms.entries()){let tr=document.createElement('tr');for(const v of [a.parent_interval_index+' / '+a.atom_index,a.subtask,a.start_s.toFixed(2)+' s',a.duration_s.toFixed(2)+' s',a.quality,a.speed??'—',a.confidence]){let td=document.createElement('td');td.textContent=v;tr.append(td)}tr.onclick=()=>select(i,true);$('atoms').append(tr)}select(0,false)}
function select(i,play){if(!current)return;index=Math.max(0,Math.min(current.atoms.length-1,i));let a=current.atoms[index];$('description').textContent=current.task+' — '+a.subtask+' ['+a.start_timestep+', '+a.end_timestep_exclusive+')';$('note').textContent=a.parent_note+(a.note?' '+a.note:'');[...$('atoms').children].forEach((r,k)=>r.classList.toggle('active',k===index));for(const key of ['external','wrist']){let v=$(key);v.pause();v.currentTime=a.start_s;if(play)v.play().catch(()=>{})}}
$('external').addEventListener('timeupdate',()=>{if(!current)return;const e=$('external'),w=$('wrist');if(e.currentTime>=current.atoms[index].end_s_exclusive){e.pause();w.pause()}else if(Math.abs(e.currentTime-w.currentTime)>.25)w.currentTime=e.currentTime});
$('external').addEventListener('play',()=>{$('wrist').currentTime=$('external').currentTime;$('wrist').play().catch(()=>{})});$('external').addEventListener('pause',()=>$('wrist').pause());
$('previous').onclick=()=>select(index-1,true);$('next').onclick=()=>select(index+1,true);$('replay').onclick=()=>select(index,true);$('source').onchange=filter;$('search').oninput=filter;$('uncertain').onchange=filter;$('episode').onchange=load;filter();</script>'''.replace('PAYLOAD',payload)
    (REVIEW/'index.html').write_text(html)
    print('wrote',REVIEW/'index.html',len(data),'episodes',len(rows),'atoms')
if __name__=='__main__':main()

"""Promote the approved render experiment into portable packaged scenes; retain backups."""
from pathlib import Path
import shutil,json,hashlib,xml.etree.ElementTree as E
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[3]
DEST=WS/'lerobot/src/lerobot/robots/rebot_b601_sim/scenes'
SOURCE=ROOT.parent/'photoreal_headphone_hinge'
BACKUP=ROOT/'before';BACKUP.mkdir(exist_ok=True)
FILES=[DEST/f'nail_ep{ep}/scene.xml' for ep in [5,1,2,3]]+[DEST/'README.md',WS/'lerobot/pyproject.toml',WS/'config_rl_sim.yaml']
for f in FILES:
 target=BACKUP/f.relative_to(WS);target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():shutil.copy2(f,target)
roots={ROOT.parent/'photoreal_clippers/assets':'clippers',ROOT.parent/'photoreal_objects/assets':'objects',SOURCE/'assets':'headphones'}
manifest={};scenes={}
for ep in [5,1,2,3]:
 layout=f'nail_ep{ep}';tree=E.parse(SOURCE/f'{layout}.xml');root=tree.getroot();root.find('compiler').set('assetdir','..')
 for el in root.iter():
  name=el.get('file')
  if not name:continue
  p=Path(name)
  if not p.is_absolute():continue
  if el.tag=='include':
   assert p==DEST/'rebot_b601_dm_photo.xml';el.set('file','../rebot_b601_dm_photo.xml');continue
  if p.is_relative_to(DEST):el.set('file',p.relative_to(DEST).as_posix());continue
  assert p.parent in roots,p
  out=DEST/'assets/photoreal'/roots[p.parent]/p.name;out.parent.mkdir(parents=True,exist_ok=True)
  if out.exists():assert out.read_bytes()==p.read_bytes(),f'Unexpected destination content: {out}'
  else:shutil.copy2(p,out)
  el.set('file',out.relative_to(DEST).as_posix());manifest[str(out.relative_to(WS))]={'source':str(p.relative_to(WS)),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
 E.indent(tree);out=DEST/layout/'scene.xml';tree.write(out,encoding='utf-8',xml_declaration=True);scenes[str(out.relative_to(WS))]=str((SOURCE/f'{layout}.xml').relative_to(WS))
(ROOT/'promotion_manifest.json').write_text(json.dumps({'scenes':scenes,'assets':manifest,'backup':str(BACKUP.relative_to(WS))},indent=2))
print(f'Promoted {len(scenes)} scenes and {len(manifest)} assets ({sum((WS/p).stat().st_size for p in manifest)/1e6:.1f} MB).')

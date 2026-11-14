"""Verify promoted physics, approved observations, runtime naming, and relocated assets."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import sys,json,copy,shutil,tempfile,tomllib,xml.etree.ElementTree as E
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[3];SRC=WS/'lerobot/src';SCENES=SRC/'lerobot/robots/rebot_b601_sim/scenes';EXP=ROOT.parent/'photoreal_headphone_hinge'
sys.path.insert(0,str(SRC))
import numpy as np,mujoco,yaml,cv2
from setuptools import Distribution
from setuptools.command.build_py import build_py
from lerobot.robots.rebot_b601_sim import RebotB601Sim
from lerobot.robots.rebot_b601_sim.config_rebot_b601_sim import RebotB601SimRobotConfig,MujocoCameraConfig

def robot(raw):
 raw=copy.deepcopy(raw);raw.pop('type');raw['episodes']=None
 cams={}
 for name,c in raw.pop('cameras').items():c.pop('type');cams[name]=MujocoCameraConfig(**c)
 o=RebotB601Sim(RebotB601SimRobotConfig(cameras=cams,**raw));o.connect();return o

report={'layouts':{}}
base=yaml.safe_load((WS/'config_rl_sim.yaml').read_text())['env']['robot']
old_config=yaml.safe_load((ROOT/'before/config_rl_sim.yaml').read_text())
assert yaml.safe_load((WS/'config_rl_sim.yaml').read_text())==old_config
for ep in [5,1,2,3]:
 name=f'nail_ep{ep}';result={};orig=E.parse(ROOT/'before/lerobot/src/lerobot/robots/rebot_b601_sim/scenes'/name/'scene.xml').getroot();orig.find('compiler').set('assetdir',str(SCENES));orig.find('include').set('file',str(SCENES/'rebot_b601_dm_photo.xml'))
 a=mujoco.MjModel.from_xml_string(E.tostring(orig,encoding='unicode'));b=mujoco.MjModel.from_xml_path(str(SCENES/name/'scene.xml'))
 for key in ['nq','nv','nu','nbody','njnt','body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat','qpos0','jnt_type','jnt_pos','jnt_axis','jnt_range','dof_damping','dof_frictionloss','dof_armature','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','eq_data','exclude_signature']:
  assert np.array_equal(getattr(a,key),getattr(b,key)),(name,key)
 ai=np.flatnonzero((a.geom_contype!=0)|(a.geom_conaffinity!=0));bi=np.flatnonzero((b.geom_contype!=0)|(b.geom_conaffinity!=0));assert len(ai)==len(bi)
 for key in ['geom_bodyid','geom_contype','geom_conaffinity','geom_condim','geom_type','geom_size','geom_pos','geom_quat','geom_friction','geom_solref','geom_solimp','geom_margin','geom_gap','geom_priority','geom_solmix']:
  assert np.array_equal(getattr(a,key)[ai],getattr(b,key)[bi]),(name,key)
 for ga,gb in zip(ai,bi):
  if a.geom_type[ga]!=mujoco.mjtGeom.mjGEOM_MESH:continue
  ma,mb=a.geom_dataid[ga],b.geom_dataid[gb]
  for key,adr,count in [('mesh_vert','mesh_vertadr','mesh_vertnum'),('mesh_face','mesh_faceadr','mesh_facenum')]:
   av=getattr(a,key)[getattr(a,adr)[ma]:getattr(a,adr)[ma]+getattr(a,count)[ma]];bv=getattr(b,key)[getattr(b,adr)[mb]:getattr(b,adr)[mb]+getattr(b,count)[mb]];assert np.array_equal(av,bv)
 result['original_physics_equal']=True
 raw=copy.deepcopy(base);raw['scene']=name;prod=robot(raw);approved=robot(yaml.safe_load((EXP/f'config_{name}.yaml').read_text())['env']['robot'])

 for field in ['tex_data','mat_texid','geom_rgba','geom_matid','light_pos','mesh_vert','mesh_normal','mesh_texcoord','geom_pos','geom_quat','mat_specular','mat_shininess','mat_rgba']:
  assert np.array_equal(getattr(prod.model,field),getattr(approved.model,field)),(name,'render model',field)
 result['approved_render_arrays_identical']=True
 obs=prod.get_observation();ref=approved.get_observation();result['observations']={}
 for cam in ['top','wrist','wrist.depth']:
  shape=(480,640) if cam.endswith('.depth') else (480,640,3);dtype=np.uint16 if cam.endswith('.depth') else np.uint8
  assert obs[cam].shape==shape and obs[cam].dtype==dtype
  diff=np.abs(obs[cam].astype(np.int32)-ref[cam].astype(np.int32));delta=int(diff.max());assert delta<=(0 if cam.endswith('.depth') else 3),(name,cam,delta)
  result['observations'][cam]={'shape':list(shape),'dtype':str(obs[cam].dtype),'max_difference_from_approved':delta,'mean_abs_difference':float(diff.mean()),'changed_fraction':float(np.mean(diff!=0))}
 cv2.imwrite(str(ROOT/f'{name}_packaged_top.jpg'),obs['top'][:,:,::-1])
 for i in range(30):
  action={f'{n}.pos':float(.5*np.sin(i/10)) for n in prod.motor_names};prod.send_action(action);approved.send_action(action)
  assert np.array_equal(prod.data.qpos,approved.data.qpos),(name,'qpos')
 result['matched_control_steps']=30;prod.disconnect();approved.disconnect();report['layouts'][name]=result;print(name,'physics, observations and stepping passed',flush=True)

# Use setuptools' own package-data expansion, then compile solely from copied package assets.
config=tomllib.loads((WS/'lerobot/pyproject.toml').read_text());patterns=config['tool']['setuptools']['package-data']
cmd=build_py(Distribution({'package_data':patterns}));cmd.finalize_options();cmd.manifest_files={}
files={Path(f).resolve() for f in cmd.find_data_files('lerobot',str(SRC/'lerobot'))};dependencies=set()
for ep in [5,1,2,3]:
 scene=SCENES/f'nail_ep{ep}/scene.xml';dependencies.add(scene);dependencies.add(scene.parent/'background_top.png')
 for doc in [scene,SCENES/'rebot_b601_dm_photo.xml']:
  dependencies.add(doc)
  for e in E.parse(doc).iter():
   f=e.get('file')
   if not f:continue
   assert not Path(f).is_absolute() and 'migration/' not in f,(doc,f)
   resolved=((doc.parent if e.tag=='include' else SCENES)/f).resolve();assert resolved.is_file(),resolved;dependencies.add(resolved)
missing=dependencies-files;assert not missing,sorted(map(str,missing))
with tempfile.TemporaryDirectory(prefix='lerobot-promoted-scenes-',dir='/tmp') as d:
 root=Path(d)
 for f in dependencies:
  out=root/f.relative_to(SRC);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,out)
 for ep in [5,1,2,3]:mujoco.MjModel.from_xml_path(str(root/'lerobot/robots/rebot_b601_sim/scenes'/f'nail_ep{ep}'/'scene.xml'))
report['portability']={'package_data_dependencies':len(dependencies),'missing_package_data':[],'relocated_four_scenes_compile':True,'runtime_migration_dependencies':False}
(ROOT/'validation.json').write_text(json.dumps(report,indent=2));print('All named-scene, physics, appearance-equivalence and relocated-package checks passed.',flush=True)

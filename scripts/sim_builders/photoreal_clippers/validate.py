"""Check original-file hashes, physical model equivalence, and RGB/depth observation contracts."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[3]
sys.path.insert(0,str(WS/'lerobot/src'))
import mujoco,numpy as np,yaml
from lerobot.robots.rebot_b601_sim import RebotB601Sim
from lerobot.robots.rebot_b601_sim.config_rebot_b601_sim import RebotB601SimRobotConfig,MujocoCameraConfig
SOURCE=WS/'lerobot/src/lerobot/robots/rebot_b601_sim/scenes'

report={'original_files':{},'layouts':{}}
original=json.loads((ROOT/'original_sha256.json').read_text())
changed=[name for name,sha in original.items() if not (WS/name).exists() or hashlib.sha256((WS/name).read_bytes()).hexdigest()!=sha]
report['original_files']={'checked':len(original),'changed':changed}
assert not changed,changed
for layout in ['nail_ep5','nail_ep1','nail_ep2','nail_ep3']:
    a=mujoco.MjModel.from_xml_path(str(SOURCE/layout/'scene.xml'))
    b=mujoco.MjModel.from_xml_path(str(ROOT/f'{layout}.xml'))
    result={'equal_model_fields':[],'equal_collision_fields':[]}
    for key in ['nq','nv','nu','nbody','njnt','body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat',
                'qpos0','jnt_type','jnt_pos','jnt_axis','jnt_range','dof_damping','dof_frictionloss','dof_armature',
                'actuator_gainprm','actuator_biasprm','actuator_ctrlrange','eq_data','exclude_signature']:
        assert np.array_equal(getattr(a,key),getattr(b,key)),(layout,key)
        result['equal_model_fields'].append(key)
    ai=np.flatnonzero((a.geom_contype!=0)|(a.geom_conaffinity!=0))
    bi=np.flatnonzero((b.geom_contype!=0)|(b.geom_conaffinity!=0))
    assert len(ai)==len(bi)
    for key in ['geom_bodyid','geom_contype','geom_conaffinity','geom_condim','geom_type','geom_size','geom_pos','geom_quat',
                'geom_friction','geom_solref','geom_solimp','geom_margin','geom_gap','geom_priority','geom_solmix']:
        assert np.array_equal(getattr(a,key)[ai],getattr(b,key)[bi]),(layout,key)
        result['equal_collision_fields'].append(key)
    for ga,gb in zip(ai,bi):
        if a.geom_type[ga]!=mujoco.mjtGeom.mjGEOM_MESH:continue
        ma,mb=a.geom_dataid[ga],b.geom_dataid[gb]
        for key,adr,count in [('mesh_vert','mesh_vertadr','mesh_vertnum'),('mesh_face','mesh_faceadr','mesh_facenum')]:
            av=getattr(a,key)[getattr(a,adr)[ma]:getattr(a,adr)[ma]+getattr(a,count)[ma]]
            bv=getattr(b,key)[getattr(b,adr)[mb]:getattr(b,adr)[mb]+getattr(b,count)[mb]]
            assert np.array_equal(av,bv),(layout,key,ga)
    raw=yaml.safe_load((ROOT/f'config_{layout}.yaml').read_text())['env']['robot'];raw.pop('type');raw['auto_subtasks']=False;raw['episodes']=None
    cams={}
    for name,config in raw.pop('cameras').items():
        config.pop('type');cams[name]=MujocoCameraConfig(**config)
    obj=RebotB601Sim(RebotB601SimRobotConfig(cameras=cams,**raw));obj.connect()
    obs=obj.get_observation();result['observations']={}
    for cam in ['top','wrist','wrist.depth']:
        expected=(480,640) if cam.endswith('.depth') else (480,640,3)
        dtype=np.uint16 if cam.endswith('.depth') else np.uint8
        assert obs[cam].shape==expected and obs[cam].dtype==dtype,(layout,cam)
        assert np.isfinite(obs[cam]).all()
        result['observations'][cam]={'shape':list(obs[cam].shape),'dtype':str(obs[cam].dtype)}
    result['collision_geoms']=len(ai);result['physics_equal']=True
    obj.disconnect();report['layouts'][layout]=result;print(layout,'passed',flush=True)
(ROOT/'validation.json').write_text(json.dumps(report,indent=2))
print('All four layouts passed; all original hashes unchanged.')

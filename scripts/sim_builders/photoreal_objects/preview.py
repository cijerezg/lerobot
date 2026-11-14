import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import sys,cv2,numpy as np,mujoco
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[3];PREV=ROOT.parent/'photoreal_clippers'
sys.path.insert(0,str(PREV))
from render import robot

def label(im,text):
    cv2.putText(im,text,(12,28),0,.7,(0,0,0),3);cv2.putText(im,text,(12,28),0,.7,(240,240,240),1);return im

for layout in ['nail_ep5','nail_ep1','nail_ep2','nail_ep3']:
    objs=[robot(PREV/f'config_{layout}.yaml'),robot(ROOT/f'config_{layout}.yaml')]
    top=[x.get_observation()['top'].copy() for x in objs]
    cv2.imwrite(str(ROOT/f'{layout}_top.jpg'),np.hstack([label(top[0],'Previous clipper pass'),label(top[1],'Updated objects')])[:,:,::-1])
    names={'nail_ep5':['bottle_yellow_cap_standing','spray_bottle_lying','blue_pouch','headphone_case_closed','bits_box'], 'nail_ep1':['headphone_half_a','bottle_orange_lying'], 'nail_ep2':[], 'nail_ep3':['paper_cup','tape_roll','headphone_half_b']}[layout]
    for name in names:
        ims=[]
        for j,obj in enumerate(objs):
            rend=mujoco.Renderer(obj.model,480,640);bid=obj.model.body(name).id
            cam=mujoco.MjvCamera();cam.lookat[:]=obj.data.xpos[bid];cam.azimuth=125;cam.elevation=-40
            if name=='bottle_yellow_cap_standing':
                v=obj.data.cam_xpos[obj.model.camera('ext').id]-obj.data.xpos[bid];cam.azimuth=np.rad2deg(np.arctan2(v[1],v[0]))
            g=next(g for g in range(obj.model.ngeom) if obj.model.geom_bodyid[g]==bid and obj.model.geom_contype[g]!=0)
            size=obj.model.geom_size[g];extent=max(size[:2]) if obj.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_CYLINDER else max(size)
            cam.distance=max(.21,extent*4.1);rend.update_scene(obj.data,cam);ims.append(label(rend.render().copy(),'Previous' if j==0 else 'Updated'));rend.close()
        cv2.imwrite(str(ROOT/f'preview_{name}.jpg'),np.hstack(ims)[:,:,::-1])
    for obj in objs:obj.disconnect()
    print('Previewed',layout,flush=True)

"""Matched native-MuJoCo views and an explicitly visual-only hinge demonstration."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import sys,json,subprocess,argparse,importlib.util
import numpy as np,cv2,mujoco,rerun as rr,rerun.blueprint as rrb
ROOT=Path(__file__).resolve().parent;PREV=ROOT.parent/'photoreal_objects'
spec=importlib.util.spec_from_file_location('previous_render',PREV/'render.py');prev=importlib.util.module_from_spec(spec);spec.loader.exec_module(prev)

def save(name,rgb):cv2.imwrite(str(ROOT/name),rgb[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,92])
def banner(im,text):
 out=np.full((im.shape[0]+90,im.shape[1],3),20,np.uint8);out[55:55+im.shape[0]]=im
 for x,s in [(16,'PREVIOUS OBJECT PASS'),(im.shape[1]//2+16,'CONNECTED CLAMSHELL')]:cv2.putText(out,s,(x,33),0,.65,(235,235,235),2,cv2.LINE_AA)
 cv2.putText(out,text,(16,len(out)-12),0,.45,(205,205,205),1,cv2.LINE_AA);return out
class Video:
 def __init__(self,name,w,h):self.p=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/name)],stdin=subprocess.PIPE)
 def frame(self,a):self.p.stdin.write(np.ascontiguousarray(a).tobytes())
 def close(self):self.p.stdin.close();assert self.p.wait()==0

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true');args=ap.parse_args();meta=json.loads((ROOT/'hinge_metadata.json').read_text());n=15 if args.quick else 150
 rr.init('Headphone case: connected clamshell',spawn=False);rr.save(str(ROOT/'headphone_case.rrd'));rr.send_blueprint(rrb.Blueprint(rrb.Grid(rrb.Spatial2DView(origin='before/detail',name='Previous detail'),rrb.Spatial2DView(origin='after/detail',name='Hinged case detail'),rrb.Spatial2DView(origin='before/top',name='Previous scene'),rrb.Spatial2DView(origin='after/top',name='Updated scene'),grid_columns=2)))
 rr.log('description',rr.TextDocument('Native simulator camera orbits; static poses, no policy. New halves and fabric hinge are visual geometry; original collision physics is unchanged.'),static=True)
 vid=Video('headphone_comparison.mp4',1280,570);sheets=[];tops=[];clock=0
 for ep in [5,1,2,3]:
  layout=f'nail_ep{ep}';m=meta[layout];pair=[prev.robot(PREV/f'config_{layout}.yaml'),prev.robot(ROOT/f'config_{layout}.yaml')];renderers=[mujoco.Renderer(o.model,480,640) for o in pair]
  o=pair[1];bid=o.model.body(m['body']).id;look=o.data.xpos[bid]+o.data.xmat[bid].reshape(3,3)@m['visual_center_local'];distance=max(m['size'][:2])*(4.2 if ep==5 else 6.4)
  top=[o.get_observation()['top'] for o in pair];topframe=banner(np.hstack(top),f'{layout} | Same table, other objects, cameras and physics');save(f'{layout}_top.jpg',topframe);tops.append(topframe)
  for i in range(n):
   ims=[]
   for rend,o in zip(renderers,pair):
    cam=mujoco.MjvCamera();cam.lookat[:]=look;cam.distance=distance;cam.azimuth=115+100*i/max(n-1,1);cam.elevation=-48+8*np.sin(i/max(n-1,1)*np.pi);rend.update_scene(o.data,cam);ims.append(rend.render().copy())
   frame=banner(np.hstack(ims),f'{layout} | '+('Closed shells, zipper and cloth hinge' if ep==5 else 'Headphones in one half; molded lid joined by a fabric hinge'))
   vid.frame(frame);rr.set_time('time',duration=clock/30);clock+=1
   for tag,im,t in zip(['before','after'],ims,top):rr.log(f'{tag}/detail',rr.Image(im).compress(jpeg_quality=85));rr.log(f'{tag}/top',rr.Image(t).compress(jpeg_quality=80))
   if i==0:save(f'{layout}_detail.jpg',frame);sheets.append(frame)
  for r in renderers:r.close()
  for o in pair:o.disconnect()
  print('Rendered',layout,flush=True)
 vid.close();rr.disconnect();save('comparison_sheet.jpg',np.vstack(sheets));save('layout_sheet.jpg',np.vstack(tops))
 # Animate only lid geom transforms to show the fold axis; no articulated dynamics are added.
 m=meta['nail_ep1'];o=prev.robot(ROOT/'config_nail_ep1.yaml');bid=o.model.body(m['body']).id;rend=mujoco.Renderer(o.model,640,960);video=Video('hinge_motion.mp4',960,720)
 ids=[o.model.geom(name).id for name in m['lid_geoms']];hids=[o.model.geom(name).id for name in m['hinge_geoms']];hp=o.model.geom_pos[hids].copy();hq=o.model.geom_quat[hids].copy();positions=o.model.geom_pos[ids].copy();quats=o.model.geom_quat[ids].copy();axis=np.array(m['axis_local']);pivot=np.array(m['pivot_local']);frames=30 if args.quick else 300
 for i in range(frames):
  t=i/(frames-1);phase=np.clip((t-.15)/.7,0,1);angle=np.pi*np.sin(np.pi*phase)**2;q=np.empty(4);mujoco.mju_axisAngle2Quat(q,axis,angle);R=np.empty(9);mujoco.mju_quat2Mat(R,q);R=R.reshape(3,3)
  for j,g in enumerate(ids):o.model.geom_pos[g]=pivot+R@(positions[j]-pivot);mujoco.mju_mulQuat(o.model.geom_quat[g],q,quats[j])
  halfq=np.empty(4);mujoco.mju_axisAngle2Quat(halfq,axis,angle/2);halfR=np.empty(9);mujoco.mju_quat2Mat(halfR,halfq);halfR=halfR.reshape(3,3)
  for j,g in enumerate(hids):o.model.geom_pos[g]=pivot+halfR@(hp[j]-pivot);mujoco.mju_mulQuat(o.model.geom_quat[g],halfq,hq[j])
  mujoco.mj_forward(o.model,o.data)
  cam=mujoco.MjvCamera();center=np.array(m['visual_center_local'])*(1-.5*(1-np.cos(angle)));cam.lookat[:]=o.data.xpos[bid]+o.data.xmat[bid].reshape(3,3)@center;cam.distance=.70;cam.elevation=-35;cam.azimuth=145
  rend.update_scene(o.data,cam);im=rend.render().copy();out=np.full((720,960,3),20,np.uint8);out[45:685]=im
  cv2.putText(out,'FABRIC HINGE: OPEN / CLOSE',(16,30),0,.70,(235,235,235),2,cv2.LINE_AA);cv2.putText(out,'Visual articulation demo | Original collision model retained | No policy',(16,709),0,.49,(210,210,210),1,cv2.LINE_AA);video.frame(out)
  if i in [0,frames//4,frames//2]:save(f'hinge_motion_{i:03d}.jpg',out)
 video.close();rend.close();o.disconnect();print('Hinge demonstration rendered',flush=True)
if __name__=='__main__':main()

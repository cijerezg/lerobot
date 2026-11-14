"""Render native MuJoCo before/after comparisons and replay real joint targets.
No policy is run. Closeups orbit a camera; replay uses real measured joints as actuator targets.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import sys,json,time,subprocess,argparse
ROOT=Path(__file__).resolve().parent; WS=ROOT.parents[3]
sys.path.insert(0,str(WS/'lerobot/src'))
import cv2,numpy as np,mujoco,yaml,pandas as pd
import rerun as rr
import rerun.blueprint as rrb
from lerobot.robots.rebot_b601_sim import RebotB601Sim
from lerobot.robots.rebot_b601_sim.config_rebot_b601_sim import RebotB601SimRobotConfig,MujocoCameraConfig
SOURCE=WS/'lerobot/src/lerobot/robots/rebot_b601_sim/scenes'

def robot(path):
    raw=yaml.safe_load(Path(path).read_text())['env']['robot'];raw.pop('type');raw['auto_subtasks']=False;raw['episodes']=None
    cams={}
    for key,cam in raw.pop('cameras').items():
        cam.pop('type');cam['use_depth']=False;cams[key]=MujocoCameraConfig(**cam)
    obj=RebotB601Sim(RebotB601SimRobotConfig(cameras=cams,**raw));obj.connect();return obj

def banner(im,left,right,subtitle):
    canvas=np.full((im.shape[0]+100,im.shape[1],3),20,np.uint8);canvas[60:60+im.shape[0]]=im
    for x,s in [(16,left),(im.shape[1]//2+16,right)]:cv2.putText(canvas,s,(x,36),0,.75,(235,235,235),2,cv2.LINE_AA)
    cv2.putText(canvas,subtitle,(16,canvas.shape[0]-16),0,.48,(200,200,200),1,cv2.LINE_AA)
    return canvas

class Video:
    def __init__(self,name,size,fps=30):
        self.path=ROOT/name
        self.p=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(self.path)],stdin=subprocess.PIPE)
    def frame(self,im):self.p.stdin.write(np.ascontiguousarray(im).tobytes())
    def close(self):
        self.p.stdin.close()
        if self.p.wait()!=0:raise RuntimeError('ffmpeg failed')

def image_out(name,rgb):cv2.imwrite(str(ROOT/name),rgb[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,94])

def main():
    p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');args=p.parse_args()
    before=robot(WS/'config_rl_sim.yaml');after=robot(ROOT/'config_nail_ep5.yaml');pair=[before,after]
    renderers=[mujoco.Renderer(x.model,480,640) for x in pair]
    fps=30;frames=36 if args.quick else 210
    vid=Video('comparison.mp4',(1280,580),fps)
    previews=[]
    for subject,distance in [('clipper_large',.145),('clipper_small',.125),('clipper_case',.27)]:
        for i in range(frames):
            u=i/max(frames-1,1);tiles=[]
            for r,obj in zip(renderers,pair):
                cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=obj.data.xpos[obj.model.body(subject).id]+[0,0,.004]
                cam.azimuth=125+170*u;cam.elevation=-32-23*np.sin(np.pi*u);cam.distance=distance
                r.update_scene(obj.data,cam);tiles.append(r.render().copy())
            frame=banner(np.hstack(tiles),'ORIGINAL','UPDATED',f'{subject.replace("_"," ").title()} | Same camera and body poses | Native MuJoCo render')
            vid.frame(frame)
            if i==0:
                image_out(f'comparison_{subject}.jpg',frame);previews.append(frame)
            if i%60==0:print('orbit',subject,i,'/',frames,flush=True)
    vid.close()
    image_out('comparison_sheet.jpg',np.vstack(previews))
    for r in renderers:r.close()

    # Actuator replay: both simulations get exactly the same targets at the original 30 Hz.
    df=pd.read_parquet(WS/'outputs/rebot_2026-10-16-annotated-v1/data/chunk-000/file-005.parquet')
    states=np.stack(df[df.episode_index==5].sort_values('frame_index')['observation.state'].values)
    rr.init('Clipper appearance comparison',spawn=False);rr.save(str(ROOT/'comparison.rrd'))
    rr.send_blueprint(rrb.Blueprint(rrb.Grid(rrb.Spatial2DView(origin='before/top',name='Original top'),rrb.Spatial2DView(origin='after/top',name='Updated top'),rrb.Spatial2DView(origin='before/wrist',name='Original wrist'),rrb.Spatial2DView(origin='after/wrist',name='Updated wrist'),grid_columns=2)))
    rr.log('description',rr.TextDocument('Recorded joint-target replay from 2026-10-16 episode 5. Identical actuator targets and collision physics. This is an appearance comparison, not a policy evaluation.'),static=True)
    vid=Video('camera_replay.mp4',(1280,1060),fps)
    start=260;end=320 if args.quick else 710
    render_ms=[[],[]];max_qpos_error=0
    for i,state in enumerate(states[:end]):
        observations=[]
        for j,obj in enumerate(pair):
            action={f'{name}.pos':float(v) for name,v in zip(obj.motor_names,state)};obj.send_action(action)
            if i>=start:
                t=time.perf_counter();observations.append(obj.get_observation());render_ms[j].append((time.perf_counter()-t)*1000)
        max_qpos_error=max(max_qpos_error,float(np.max(np.abs(before.data.qpos-after.data.qpos))))
        if i<start:continue
        rr.set_time('time',duration=(i-start)/fps)
        for tag,obs in zip(['before','after'],observations):
            for cam in ['top','wrist']:rr.log(f'{tag}/{cam}',rr.Image(obs[cam]).compress(jpeg_quality=85))
        top=np.hstack([x['top'] for x in observations]);wrist=np.hstack([x['wrist'] for x in observations])
        frame=banner(np.vstack([top,wrist]),'ORIGINAL','UPDATED',f'Real joint targets, ep5 f{i} | top / wrist | Identical physics; no policy evaluation')
        vid.frame(frame)
        if i in [start,334,500,668]:image_out(f'camera_replay_f{i:04d}.jpg',frame)
        if i%60==0:print('replay',i,'/',end,flush=True)
    vid.close();rr.disconnect()
    result=dict(replay_frames=[start,end-1],fps=fps,source='outputs/rebot_2026-10-16-annotated-v1 episode 5 observation.state',max_qpos_difference=max_qpos_error,rgb_observation_ms_mean=[float(np.mean(x)) for x in render_ms],rgb_observation_ms_p95=[float(np.percentile(x,95)) for x in render_ms],note='Two 640x480 RGB cameras, including top composite; excludes depth and policy. No performance claim for concurrent training.')
    (ROOT/'replay_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    for obj in pair:obj.disconnect()

if __name__=='__main__':main()

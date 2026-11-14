"""Native MuJoCo object gallery and identical real-target replay; no policy is run."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import sys,json,time,subprocess,argparse
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[3];PREV=ROOT.parent/'photoreal_clippers'
sys.path.insert(0,str(WS/'lerobot/src'))
import cv2,numpy as np,mujoco,yaml,pandas as pd
import rerun as rr
import rerun.blueprint as rrb
from lerobot.robots.rebot_b601_sim import RebotB601Sim
from lerobot.robots.rebot_b601_sim.config_rebot_b601_sim import RebotB601SimRobotConfig,MujocoCameraConfig

def robot(path):
    raw=yaml.safe_load(Path(path).read_text())['env']['robot'];raw.pop('type');raw['auto_subtasks']=False;raw['episodes']=None;cams={}
    for key,cam in raw.pop('cameras').items():cam.pop('type');cam['use_depth']=False;cams[key]=MujocoCameraConfig(**cam)
    obj=RebotB601Sim(RebotB601SimRobotConfig(cameras=cams,**raw));obj.connect();return obj

def banner(im,subtitle):
    canvas=np.full((im.shape[0]+100,im.shape[1],3),20,np.uint8);canvas[60:60+im.shape[0]]=im
    for x,s in [(16,'PREVIOUS CLIPPER PASS'),(im.shape[1]//2+16,'UPDATED OBJECTS')]:cv2.putText(canvas,s,(x,36),0,.67,(235,235,235),2,cv2.LINE_AA)
    cv2.putText(canvas,subtitle,(16,canvas.shape[0]-16),0,.46,(200,200,200),1,cv2.LINE_AA);return canvas

class Video:
    def __init__(self,name,size):
        self.p=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/name)],stdin=subprocess.PIPE)
    def frame(self,im):self.p.stdin.write(np.ascontiguousarray(im).tobytes())
    def close(self):
        self.p.stdin.close()
        if self.p.wait()!=0:raise RuntimeError('ffmpeg failed')

def image_out(name,rgb):cv2.imwrite(str(ROOT/name),rgb[:,:,::-1],[cv2.IMWRITE_JPEG_QUALITY,94])

def main():
    p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');args=p.parse_args()
    pairs={layout:[robot(PREV/f'config_{layout}.yaml'),robot(ROOT/f'config_{layout}.yaml')] for layout in ['nail_ep5','nail_ep1','nail_ep2','nail_ep3']}
    gallery=Video('objects_comparison.mp4',(1280,580));previews=[]
    for layout,pair in pairs.items():
        frame=banner(np.hstack([o.get_observation()['top'] for o in pair]),f'{layout} | Matching cameras and body poses | Native simulator observations')
        image_out(f'{layout}_comparison.jpg',frame);previews.append(frame)
        for _ in range(12 if args.quick else 60):gallery.frame(frame)
    image_out('layouts_comparison.jpg',np.vstack(previews));previews=[]
    shots=[('nail_ep5','bottle_yellow_cap_standing','Vitamin bottle: real label and ribbed cap'),('nail_ep1','bottle_orange_lying','Orange bottle: real label and shaped shoulders'),('nail_ep5','spray_bottle_lying','Spray bottle: label, neck and nozzle'),('nail_ep5','blue_pouch','Tool pouch: fabric, blue binding and zipper'),('nail_ep5','bits_box','Open organizer: clear lid, dividers, plugs and screws'),('nail_ep5','headphone_case_closed','Closed headphone case: fabric, zipper and handle'),('nail_ep1','headphone_half_a','Open headphone case: earcups, headband and cable'),('nail_ep3','headphone_half_b','Case lid: liner, rim and molded earcup impressions'),('nail_ep3','paper_cup','Paper cup: real print, tapered walls and open interior'),('nail_ep3','tape_roll','Tape roll: layered film and hollow cardboard core')]
    for layout,name,title in shots:
        pair=pairs[layout];renderers=[mujoco.Renderer(o.model,480,640) for o in pair]
        obj=pair[0];bid=obj.model.body(name).id;g=next(g for g in range(obj.model.ngeom) if obj.model.geom_bodyid[g]==bid and obj.model.geom_contype[g]!=0)
        size=obj.model.geom_size[g];extent=max(size[:2]) if obj.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_CYLINDER else max(size)
        distance=max(.21,extent*4.1);look_offset=np.zeros(3);start_az=125
        if name=='bottle_yellow_cap_standing':
            v=obj.data.cam_xpos[obj.model.camera('ext').id]-obj.data.xpos[bid];start_az=np.rad2deg(np.arctan2(v[1],v[0]))+180
        if name=='bits_box':distance*=1.15;look_offset[2]=.026
        for i in range(24 if args.quick else 90):
            u=i/(23 if args.quick else 89);tiles=[]
            for rend,obj in zip(renderers,pair):
                cam=mujoco.MjvCamera();cam.lookat[:]=obj.data.xpos[obj.model.body(name).id]+look_offset;cam.azimuth=start_az+110*u;cam.elevation=-36-15*np.sin(np.pi*u);cam.distance=distance
                rend.update_scene(obj.data,cam);tiles.append(rend.render().copy())
            frame=banner(np.hstack(tiles),title+' | Same physics')
            gallery.frame(frame)
            if i==0:image_out(f'comparison_{name}.jpg',frame);previews.append(frame)
        for rend in renderers:rend.close()
        print('Rendered',name,flush=True)
    gallery.close();image_out('objects_comparison_sheet.jpg',np.vstack(previews))

    before,after=pairs['nail_ep5'];pair=[before,after]
    states_df=pd.read_parquet(WS/'outputs/rebot_2026-10-16-annotated-v1/data/chunk-000/file-005.parquet')
    states=np.stack(states_df[states_df.episode_index==5].sort_values('frame_index')['observation.state'].values)
    rr.init('Surrounding object appearance comparison',spawn=False);rr.save(str(ROOT/'objects_comparison.rrd'))
    rr.send_blueprint(rrb.Blueprint(rrb.Grid(rrb.Spatial2DView(origin='before/top',name='Previous top'),rrb.Spatial2DView(origin='after/top',name='Updated top'),rrb.Spatial2DView(origin='before/wrist',name='Previous wrist'),rrb.Spatial2DView(origin='after/wrist',name='Updated wrist'),grid_columns=2)))
    rr.log('description',rr.TextDocument('Real measured joint-target replay, 10-16 episode 5. Previous clipper pass versus additional clutter visuals. Identical collision physics; not a policy evaluation.'),static=True)
    video=Video('camera_replay.mp4',(1280,1060));start=260;end=320 if args.quick else 710;max_error=0;times=[[],[]]
    for i,state in enumerate(states[:end]):
        observations=[]
        for j,obj in enumerate(pair):
            obj.send_action({f'{n}.pos':float(v) for n,v in zip(obj.motor_names,state)})
            if i>=start:
                t=time.perf_counter();observations.append(obj.get_observation());times[j].append((time.perf_counter()-t)*1000)
        max_error=max(max_error,float(np.max(np.abs(before.data.qpos-after.data.qpos))))
        if i<start:continue
        rr.set_time('time',duration=(i-start)/30)
        for tag,obs in zip(['before','after'],observations):
            for cam in ['top','wrist']:rr.log(f'{tag}/{cam}',rr.Image(obs[cam]).compress(jpeg_quality=85))
        frame=banner(np.vstack([np.hstack([o['top'] for o in observations]),np.hstack([o['wrist'] for o in observations])]),f'Real joint targets, ep5 f{i} | top / wrist | No policy evaluation')
        video.frame(frame)
        if i in [260,334,500,668]:image_out(f'camera_replay_f{i:04d}.jpg',frame)
        if i%90==0:print('Replay frame',i,flush=True)
    video.close();rr.disconnect()
    result={'replay_frames':[start,end-1],'fps':30,'max_qpos_difference':max_error,'rgb_observation_ms_mean':[float(np.mean(t)) for t in times],'rgb_observation_ms_p95':[float(np.percentile(t,95)) for t in times],'note':'Two native 640x480 RGB observations, including top composite; excludes depth and policy.'}
    (ROOT/'replay_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    for pair in pairs.values():
        for obj in pair:obj.disconnect()

if __name__=='__main__':main()

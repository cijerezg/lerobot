"""Real photograph crops and restrained material detail; no fabricated label text."""
from pathlib import Path
import cv2,numpy as np,json
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[3];A=ROOT/'assets';A.mkdir(exist_ok=True)
META={}

def crop(name,path,quad,size=(320,512),gain=None):
    im=cv2.imread(str(WS/path));assert im is not None,path
    dst=np.float32([[0,0],[size[0]-1,0],[size[0]-1,size[1]-1],[0,size[1]-1]])
    H=cv2.getPerspectiveTransform(np.float32(quad),dst)
    out=cv2.warpPerspective(im,H,size,flags=cv2.INTER_CUBIC)
    if gain is not None:out=np.clip(out*np.array(gain),0,255).astype(np.uint8)
    cv2.imwrite(str(A/f'{name}_crop.png'),out)
    META[name]={'source':str(path),'quad':quad,'homography':H.tolist(),'color_gain_bgr':gain}
    return out

def wrap(name,im,bg):
    # Front uses observed pixels. Unseen back is plain label paper, not invented text.
    h=512;w=1024;tex=np.full((h,w,3),bg,np.uint8)
    tex[:,320:704]=cv2.resize(im,(384,h))
    for k in range(12):
        alpha=(k+1)/13;tex[:,308+k]=np.clip(np.array(bg)*(1-alpha)+tex[:,320]*alpha,0,255)
        tex[:,704+k]=np.clip(tex[:,703]*(1-alpha)+np.array(bg)*alpha,0,255)
    cv2.imwrite(str(A/f'{name}.png'),tex)

# real2sim work/ frames were deleted before the 2026-11-13 move; the crops they produced are in assets/.
base='lerobot/scripts/sim_builders/real2sim/work/layouts'
yellow=crop('yellow',f'{base}/ep1/frames/wrist_01001.jpg',[[497,252],[588,277],[553,393],[454,357]],gain=[.86,.94,1.0]);wrap('yellow_label',yellow,[75,195,205])
orange=crop('orange','lerobot/scripts/sim_builders/photoreal_objects/references/ep2_wrist_2650.jpg',[[365,303],[414,315],[395,377],[344,361]]);wrap('orange_label',orange,[30,75,170])
spray=crop('spray','lerobot/scripts/sim_builders/real2sim/work/scene_nail/frames/r1016_ep5/top_00000.jpg',[[278,334],[272,355],[201,337],[207,316]]);wrap('spray_label',spray,[45,30,20])
cup=crop('cup',f'{base}/ep1/frames/top_00000.jpg',[[164,173],[192,159],[208,188],[185,195]],size=(256,384));wrap('cup_print',cup,[155,176,186])
# High-resolution actual textile weave, stripped of broad light/shadow before tiling.
im=cv2.imread(str(ROOT/'references/headphones_030.jpg'));h,w=im.shape[:2]
patch=im[int(.45*h):int(.55*h),int(.36*w):int(.55*w)]
gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY).astype(float);ratio=gray/np.maximum(cv2.GaussianBlur(gray,(0,0),11),10)
ratio=np.clip(ratio,.65,1.35);tile=cv2.resize(ratio,(512,128));tile=np.hstack([tile,tile[:,::-1]]);tile=np.vstack([tile,tile[::-1]]);tile=np.tile(tile,(4,1))
for name,col in [('fabric',[43,40,38]),('lining',[75,70,66]),('pouch',[28,26,23])]:
    tex=np.clip(tile[:,:,None]*np.array(col),0,255).astype(np.uint8);cv2.imwrite(str(A/f'{name}.png'),tex)
META['fabric']={'source':'sim/headphones.mp4','time_s':30,'normalized_crop':[.36,.45,.55,.55],'method':'photographic high-frequency weave, mirror tiling; mean color matched to dark case surfaces'}
(A/'texture_sources.json').write_text(json.dumps(META,indent=2))
print('Real-data textures written')

"""Extract the real frames used to author the clipper visuals; source video remains untouched."""
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[3]
out=ROOT/'references';out.mkdir(exist_ok=True)
cap=cv2.VideoCapture(str(WS/'sim/nail.mp4'))
assert cap.isOpened(),'Cannot open source video'
rows=[]
for t in [13,18.6,27,41,62,83]:
    cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,im=cap.read()
    assert ok,f'Failed to decode {t}s'
    cv2.imwrite(str(out/f'scan_{t:05.1f}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,97])
    small=cv2.resize(im,(768,432));cv2.putText(small,f'{t}s',(10,28),0,.8,(0,0,255),2);rows.append(small)
cap.release()
cv2.imwrite(str(out/'scan_views.jpg'),np.vstack([np.hstack(rows[i:i+2]) for i in range(0,6,2)]))
for t,name,box in [(18.6,'clipper_top_detail',(.31,.16,.64,.60)),(41,'clipper_side_detail',(.40,.55,.75,.95)),(62,'clipper_pivot_detail',(.05,.10,.45,.60))]:
    im=cv2.imread(str(out/f'scan_{t:05.1f}.jpg'));h,w=im.shape[:2];x0,y0,x1,y1=box
    cv2.imwrite(str(out/f'{name}.jpg'),im[int(y0*h):int(y1*h),int(x0*w):int(x1*w)])
print('Reference frames written to',out)

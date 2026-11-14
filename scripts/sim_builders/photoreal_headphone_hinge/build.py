"""Connected visual clamshells; preserve all original dynamics and other assets."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json,sys,numpy as np,trimesh,mujoco,yaml
from geometry import mesh,shaped,outline,tube,box,ellipsoid,ring
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[3]
PREV=ROOT.parent/'photoreal_objects';A=ROOT/'assets';A.mkdir(exist_ok=True)

def move(m,p):m.apply_translation(p);return m

def rim(x,y,z,r=.0015):
 p=outline(x,y,n=160,power=3.2);return tube(np.c_[p,np.full(len(p),z)],r,n=8,closed=True)

def shell(x,y,z,lid=False):
 # Separate textile outer wall and gray molded inner liner.
 outer=[(-z,.76),(-.9*z,.85),(-.65*z,.94),(-.2*z,.99),(.58*z,1)]
 inner=[(.58*z,.965),(.2*z,.94),(-.35*z,.86),(-.58*z,.75)]
 parts=[(shaped(x,y,outer,power=3.2,caps=False),'fabric'),(shaped(x,y,inner,power=3.2,caps=False),'lining'),(shaped(x*.76,y*.76,[(-z,1),(-.58*z,.99)],power=3.2),'fabric'),(shaped(x*.752,y*.752,[(-.579*z,1),(-.575*z,1)],power=3.2),'lining'),(rim(x,y,.60*z),'black'),(rim(x*1.002,y*1.002,.36*z,.0006),'zipper')]
 # Small, spaced stitches follow the perimeter; cloth hinge occupies one side.
 p=outline(x*1.005,y*1.005,n=180,power=3.2)
 stitches=[]
 for i in range(0,len(p),2):
  a=p[i];b=p[(i+1)%len(p)];stitches.append(tube([[*a,.15*z],[*b,.15*z]],.00018,n=5))
 parts.append((trimesh.util.concatenate(stitches),'gray'))
 if lid:
  # Broad molded contours, rather than two raised rings on a flat tray.
  for s in [-1,1]:
   m=shaped(.31*x,.45*y,[(-.60*z,.80),(-.48*z,.95),(-.40*z,1),(-.34*z,.92)],power=2.4)
   parts.append((move(m,[s*.36*x,0,0]),'lining'))
  path=np.array([[-.60*x,-.52*y,-.41*z],[-.30*x,-.61*y,-.40*z],[0,-.66*y,-.37*z],[.30*x,-.61*y,-.40*z],[.60*x,-.52*y,-.41*z]])
  parts.append((tube(path,.003,n=12),'lining'))
 return parts

def headband(x,y,z):
 # Wide ribbon-shaped padded band following the edge of the folded headphones.
 t=np.linspace(.04*np.pi,.96*np.pi,90);v=[];f=[]
 for a in t:
  c=np.array([.76*x*np.cos(a),-.76*y*np.sin(a),z*.50])
  n=np.array([np.cos(a),-np.sin(a),0.]);n/=np.linalg.norm(n)
  for w,h in [(-.008,-.003),(.008,-.003),(.008,.003),(-.008,.003)]:v.append(c+w*n+[0,0,h])
 for i in range(len(t)-1):
  for j in range(4):
   a=4*i+j;b=4*i+(j+1)%4;c=a+4;d=b+4;f.extend([(a,b,c),(b,d,c)])
 return mesh(v,f)

def headphones(x,y,z):
 parts=[(headband(x,y,z),'rubber')]
 for s in [-1,1]:
  center=np.array([s*.37*x,-.02*y,.16*z])
  cushion=shaped(.295*x,.41*y,[(-.48*z,.75),(-.30*z,1),(.18*z,1),(.30*z,.91)],power=2.5)
  parts.append((move(cushion,center),'rubber'))
  housing=shaped(.28*x,.39*y,[(.10*z,.95),(.35*z,1),(.65*z,.91),(.77*z,.70)],power=2.7)
  parts.append((move(housing,center),'black'))
  # Flattened outer earcup face and visible swivel arm.
  face=shaped(.195*x,.27*y,[(.76*z,1),(.79*z,.94)],power=2.8)
  parts.append((move(face,center),'gray'))
  arm=np.array([[s*.76*x,-.025*y,z*.50],[s*.70*x,.13*y,z*.52],[s*.59*x,.20*y,z*.63]])
  parts.append((tube(arm,.004,n=12),'black'))
  parts.append((ellipsoid([.008,.006,.004],[s*.59*x,.20*y,z*.63]),'zipper'))
 # Real case has a cable pocket beside the folded headphones.
 cable=[]
 for k in range(5):
  t=np.linspace(0,2*np.pi,60);path=np.c_[(.025+k*.0015)*np.cos(t),.56*y+(.009+k*.001)*np.sin(t),np.full(len(t),-.30*z+k*.0007)]
  cable.append(tube(path,.00065,n=6,closed=True))
 parts.append((trimesh.util.concatenate(cable),'black'))
 pocket=shaped(.34*x,.18*y,[(-.36*z,1),(.05*z,.90)],power=5)
 parts.append((move(pocket,[0,.73*y,0]),'lining'))
 parts.append((tube([[-.30*x,.57*y,.04*z],[0,.56*y,.045*z],[.30*x,.57*y,.04*z]],.0007),'gray'))
 return parts

def handle(x,y,z):
 t=np.linspace(0,np.pi,45);path=np.c_[.029*np.cos(t),-y-.036*np.sin(t),z+.012*np.sin(t)]
 # Flat tape handle, not a circular rubber tube.
 v=[];f=[]
 for p in path:
  for dz in [-.001,.001]:
   for dx in [-.005,.005]:v.append(p+[dx,0,dz])
 for i in range(len(path)-1):
  for j,k in [(0,1),(1,3),(3,2),(2,0)]:a=4*i+j;b=4*i+k;f.extend([(a,b,a+4),(b,b+4,a+4)])
 return mesh(v,f)

def hinge(x,y,z,gap):
 v=[];uv=[];f=[];nx=40;ny=10
 for i in range(nx+1):
  xx=(-.56+1.12*i/nx)*x;edge=y*(1-abs(xx/x)**3.2)**(1/3.2)
  for j in range(ny+1):
   u=j/ny;yy=edge+(2*y+gap-2*edge)*u
   v.append([xx,yy,.58*z-.0018*np.sin(np.pi*u)]);uv.append([i/nx*3,j/ny])
 for i in range(nx):
  for j in range(ny):a=i*(ny+1)+j;f.extend([(a,a+ny+1,a+1),(a+1,a+ny+1,a+ny+2)])
 return mesh(v,f,uv)

def pixel_direction(model,data,bid,target):
 cid=model.camera('ext').id;R=data.cam_xmat[cid].reshape(3,3);origin=data.cam_xpos[cid]
 focal=240/np.tan(np.deg2rad(model.cam_fovy[cid])/2)
 ray=R@np.array([(target[0]-.5)*640,-(target[1]-.5)*480,-focal])
 hit=origin+ray*((data.xpos[bid,2]-origin[2])/ray[2]);local=data.xmat[bid].reshape(3,3).T@(hit-data.xpos[bid])
 return np.arctan2(local[1],local[0])-np.pi/2

metadata={}
for ep in [5,1,2,3]:
 layout=f'nail_ep{ep}';tree=ET.parse(PREV/f'{layout}.xml');root=tree.getroot();asset=root.find('asset')
 model=mujoco.MjModel.from_xml_path(str(PREV/f'{layout}.xml'));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
 body=next(b for b in root.findall('.//body') if b.get('name','').startswith('headphone'))
 name=body.get('name');bid=model.body(name).id;collider=next(g for g in body.findall('geom') if g.get('group')=='3');x,y,z=np.fromstring(collider.get('size'),sep=' ')
 for g in list(body.findall('geom')):
  if g.get('group')=='2' and g.get('mass')=='0':body.remove(g)
 
 if ep!=5:x*=.97;y*=.88
 parts=[];gap=.007;sep=2*y+gap;angle=0.;pivot=np.array([0,y+gap/2,.58*z]);lid_names=[];hinge_names=[]
 if ep==5:
  # A real closed clamshell: two gently rounded shells separated by a zipper seam.
  for s in [-1,1]:
   profile=[(0,1),(.45*z,.99),(.75*z,.94),(.94*z,.80),(z,.63)]
   m=shaped(x,y,profile,power=3.2)
   if s<0:m.apply_transform(trimesh.transformations.rotation_matrix(np.pi,[1,0,0]))
   parts.append((m,'fabric','fixed'))
  parts.extend([(rim(x*1.004,y*1.004,0,.0015),'black','fixed'),(rim(x*1.008,y*1.008,.001,.00065),'zipper','fixed'),(handle(x,y,0),'fabric','fixed')])
  parts.append((box([x*.8,.009,.012],[0,y-.002,0]),'fabric','fixed'))
  pull=box([.015,.008,.002],[x*.60,-y*.91,.001]);parts.append((pull,'zipper','fixed'))
 else:
  # In ep3 the pre-existing body is the empty lid; in ep1/2 it is the loaded base.
  anchor_lid=ep==3;target={1:(.84,.34),2:(.435,.65),3:(.595,.486)}[ep]
  angle=pixel_direction(model,data,bid,target)
  for m,mat in shell(x,y,z,lid=anchor_lid):parts.append((m,mat,'fixed' if not anchor_lid else 'lid'))
  if not anchor_lid:
   for m,mat in headphones(x,y,z):parts.append((m,mat,'fixed'))
   parts.append((handle(x,y,0),'fabric','fixed'))
  # Mate is mirrored in-plane across the hinge, with the same outer silhouette.
  T=trimesh.transformations.rotation_matrix(np.pi,[0,0,1]);T[:3,3]=[0,sep,0]
  mate=shell(x,y,z,lid=not anchor_lid)
  if anchor_lid:mate+=headphones(x,y,z)+[(handle(x,y,0),'fabric')]
  for m,mat in mate:m.apply_transform(T);parts.append((m,mat,'fixed' if anchor_lid else 'lid'))
  parts.append((hinge(x,y,z,gap),'fabric','hinge'))
 rotation=trimesh.transformations.rotation_matrix(angle,[0,0,1]);pivot=(rotation@np.r_[pivot,1])[:3]
 for i,(m,mat,role) in enumerate(parts):
  m.apply_transform(rotation);key=f'hinged_{layout}_{i:02d}';path=A/f'{key}.obj'
  uv=np.asarray(m.visual.uv) if hasattr(m.visual,'uv') and m.visual.uv is not None else np.full((len(m.vertices),2),.5)
  normals=m.vertex_normals
  with path.open('w') as f:
   f.writelines('v %.8f %.8f %.8f\n'%tuple(v) for v in m.vertices);f.writelines('vt %.7f %.7f\n'%tuple(v) for v in uv);f.writelines('vn %.7f %.7f %.7f\n'%tuple(v) for v in normals)
   f.writelines('f '+' '.join(f'{j+1}/{j+1}/{j+1}' for j in face)+'\n' for face in m.faces)
  ET.SubElement(asset,'mesh',name=key,file=str(path),inertia='shell')
  ET.SubElement(body,'geom',name=key,type='mesh',mesh=key,material='clutter_'+mat,group='2',contype='0',conaffinity='0',mass='0')
  if role=='lid':lid_names.append(key)
  if role=='hinge':hinge_names.append(key)
 scene=ROOT/f'{layout}.xml';ET.indent(tree);tree.write(scene,encoding='utf-8',xml_declaration=True)
 cfg=yaml.safe_load((PREV/f'config_{layout}.yaml').read_text());cfg['env']['robot']['scene']=str(scene);(ROOT/f'config_{layout}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 metadata[layout]={'body':name,'pivot_local':pivot.tolist(),'axis_local':rotation[:3,0].tolist(),'lid_geoms':lid_names,'hinge_geoms':hinge_names,'visual_center_local':(rotation@np.array([0,sep/2 if ep!=5 else 0,0,1]))[:3].tolist(),'size':[x,y,z],'parts':len(parts),'angle':float(angle),'note':'Visual hinge; existing simplified collision geometry retained.'}
 print(layout,len(parts),'parts',flush=True)
(ROOT/'hinge_metadata.json').write_text(json.dumps(metadata,indent=2))

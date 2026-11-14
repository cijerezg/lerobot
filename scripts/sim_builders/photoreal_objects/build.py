"""Add reference-guided clutter visuals to the accepted clipper scenes. Originals are read-only."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json,numpy as np,trimesh,mujoco,yaml
from geometry import lathe,shaped,outline,tube,box,ellipsoid,disk,ring
ROOT=Path(__file__).resolve().parent;WS=ROOT.parents[3];PREV=ROOT.parent/'photoreal_clippers';A=ROOT/'assets';A.mkdir(exist_ok=True)
MATERIALS={
 'white':dict(rgba='.80 .81 .79 1',specular='.23',shininess='.35'),
 'cap':dict(rgba='.88 .89 .87 1',specular='.28',shininess='.45'),
 'black':dict(rgba='.045 .050 .058 1',specular='.3',shininess='.4'),
 'rubber':dict(rgba='.045 .048 .052 1',specular='.08',shininess='.12'),
 'gray':dict(rgba='.15 .17 .19 1',specular='.35',shininess='.55'),
 'blue':dict(rgba='.025 .105 .44 1',specular='.17',shininess='.25'),
 'zipper':dict(rgba='.18 .20 .23 1',specular='.4',shininess='.5'),
 'silver':dict(rgba='.56 .59 .62 1',specular='.85',shininess='.75'),
 'red':dict(rgba='.68 .025 .035 1',specular='.2',shininess='.3'),
 'yellow':dict(rgba='.83 .67 .08 1',specular='.2',shininess='.3'),
 'cyan':dict(rgba='.02 .42 .62 1',specular='.2',shininess='.3'),
 'plug_gray':dict(rgba='.30 .33 .33 1',specular='.15',shininess='.25'),
 'plastic':dict(rgba='.59 .64 .65 .66',specular='.4',shininess='.55'),
 'lid':dict(rgba='.70 .75 .77 .10',specular='.4',shininess='.6'),
 'tape':dict(rgba='.57 .53 .38 1',specular='.55',shininess='.6'),
 'core':dict(rgba='.72 .73 .68 1',specular='.04',shininess='.1'),
 'inner':dict(rgba='.83 .83 .80 1',specular='.06',shininess='.1'),
}
for tex in ['yellow_label','orange_label','spray_label','cup_print','fabric','lining','pouch']:
 MATERIALS[tex]=dict(texture='detail_'+tex,specular='.08' if tex in ['fabric','lining','pouch'] else '.15',shininess='.25')

def transformed(m,rot=None,pos=None):
    if rot is not None:m.apply_transform(rot)
    if pos is not None:m.apply_translation(pos)
    return m

def bottle(size,name,phi):
    r,h=size[:2];H=2*h;parts=[]
    if name=='spray_bottle_lying':
        p=[(-h,.65*r),(-.46*H,.88*r),(-.41*H,r),(.18*H,r),(.25*H,.9*r),(.30*H,.44*r),(.35*H,.42*r)]
        parts.append((lathe(p),'black'))
        parts.append((lathe([(.30*H,.44*r),(.37*H,.44*r)],ribs=32),'black'))
        parts.append((lathe([(.35*H,.19*r),(.46*H,.19*r)]),'black'))
        parts.append((box([.024,.012,.012],[.007,0,.46*H]),'black'))
        parts.append((disk(.003,.006,[.021,0,.46*H]),'gray'))
        zlo,zhi=-.34*H,.16*H;label='spray_label'
    else:
        p=[(-h,.76*r),(-.48*H,.91*r),(-.44*H,r),(.18*H,r),(.24*H,.93*r),(.29*H,.69*r),(.36*H,.67*r)]
        parts.append((lathe(p),'white'))
        parts.append((lathe([(.31*H,.74*r),(.335*H,.83*r),(.475*H,.83*r),(.50*H,.75*r)],ribs=48),'cap'))
        zlo,zhi=-.37*H,.18*H;label='yellow_label' if 'yellow' in name else 'orange_label'
    sleeve=lathe([(zlo,r+.00015),(zhi,r+.00015)],caps=False)
    sleeve.visual.uv[:,0]+=.5-phi/(2*np.pi)
    parts.append((sleeve,label));return parts

def pouch(size):
    x,y,z=size;parts=[(shaped(x,y,[(-z,.92),(-z*.7,1),(z*.65,1),(z,.95)],power=9),'pouch')]
    for zz in [z*.75,-z*.30]:
        p=outline(x*.98,y*.98,power=9);parts.append((tube(np.c_[p,np.full(len(p),zz)],.0020,closed=True),'blue'))
    p=outline(x*.995,y*.995,power=9);parts.append((tube(np.c_[p,np.full(len(p),z*.12)],.0010,closed=True),'zipper'))
    pull=box([.018,.008,.002],[x*.85,-y-.003,z*.15]);parts.append((pull,'black'))
    return parts

def strap(x,y,z,width=.014):
    # Flat woven handle arch extending from the end of the shell.
    pieces=[];ts=np.linspace(0,np.pi,20)
    path=np.c_[-x-.027*np.sin(ts),y+.028*np.cos(ts),z+.016*np.sin(ts)]
    for a,b in zip(path[:-1],path[1:]):
        m=box([np.linalg.norm(b-a),width,.0025]);direction=b-a
        T=trimesh.geometry.align_vectors([1,0,0],direction);m.apply_transform(T);m.apply_translation((a+b)/2);pieces.append(m)
    return trimesh.util.concatenate(pieces)

def headphones(size,kind):
    x,y,z=size;parts=[]
    if kind=='closed':
        zz=np.linspace(-.99*z,.99*z,23)
        p=list(zip(zz,(1-(zz/z)**2)**.28))
        parts.append((shaped(x,y,p,power=2.6),'fabric'))
        p=outline(x,y,power=2.6);parts.append((tube(np.c_[p,np.zeros(len(p))],.0022,closed=True),'black'))
        parts.append((tube(np.c_[p*1.002,np.full(len(p),.001)],.00065,closed=True),'zipper'))
        parts.append((strap(x,0,0),'fabric'))
        parts.append((box([.018,.009,.003],[x*.1,-y,.001]),'zipper'))
        return parts
    p=[(-z,.72),(-.8*z,.9),(.45*z,1),(.65*z,.99),(.65*z,.93),(-.58*z,.78)]
    parts.append((shaped(x,y,p,power=2.8),'lining'))
    p=outline(x,y,power=2.8);parts.append((tube(np.c_[p,np.full(len(p),.6*z)],.0018,closed=True),'black'))
    parts.append((strap(x,0,0),'fabric'))
    if kind=='base':
        for sign in [-1,1]:
            center=[sign*x*.38,sign*y*.09,z*.12]
            parts.append((ellipsoid([x*.32,y*.39,z*.75],center),'black'))
            parts.append((ellipsoid([x*.245,y*.30,z*.08],[center[0],center[1],z*.87]),'gray'))
            pad=ring(x*.29,y*.355,-z*.15,.003);pad.apply_translation([center[0],center[1],0]);parts.append((pad,'rubber'))
        t=np.linspace(.10*np.pi,1.1*np.pi,70)
        path=np.c_[x*.70*np.cos(t),y*.69*np.sin(t),np.full(len(t),z*.38)]
        parts.append((tube(path,.007,n=12),'black'))
        cable=[]
        for i in range(4):
            t=np.linspace(0,2*np.pi,50);cable.append(tube(np.c_[x*.54+(.009+i*.0012)*np.cos(t),-y*.52+.016*np.sin(t),np.full(len(t),-.003+i*.0007)],.0006,closed=True))
        parts.append((trimesh.util.concatenate(cable),'black'))
    else:
        # Molded lining contours matching the two earcup impressions in the real lid.
        for sign in [-1,1]:
            p=outline(x*.32,y*.36,n=96,power=2)
            path=np.c_[p[:,0]+sign*x*.35,p[:,1],np.full(len(p),-z*.48)]
            parts.append((tube(path,.0018,closed=True),'lining'))
    return parts

def cup(size,phi):
    r,h=size[:2];parts=[]
    wall=lathe([(-h,r*.65),(-h+.003,r*.67),(h-.002,r*.97)],caps=False)
    wall.visual.uv[:,0]+=.5-phi/(2*np.pi);parts.append((wall,'cup_print'))
    # The cavity wall faces inward; no solid cap spans the opening.
    inside=lathe([(h-.002,r*.94),(-h+.003,r*.62)],caps=False)
    parts.append((inside,'inner'));parts.append((disk(r*.64,.002,[0,0,-h+.002]),'inner'))
    parts.append((ring(r*.97,r*.97,h-.001,.0014),'cap'))
    parts.append((ring(r*.65,r*.65,-h+.001,.0008),'white'))
    return parts

def tape(size):
    r,h=size[:2];inner=r*.65
    film=lathe([(-h,r*.96),(-h+.001,r),(h-.001,r),(h,r*.96),(h,inner),(-h,inner),(-h,r*.96)],caps=False)
    core=lathe([(h,inner-.0002),(-h,inner-.0002)],caps=False)
    parts=[(film,'tape'),(core,'core')]
    for dz in np.linspace(-h+.002,h-.002,9):parts.append((ring(r,r,dz,.00012),'tape'))
    return parts

def organizer(size):
    x,y,z=size;parts=[]
    parts.append((shaped(x,y,[(-z,.92),(-z+.002,1),(-z+.004,1)],power=12),'plastic'))
    p=outline(x,y,power=12);parts.append((tube(np.c_[p,np.full(len(p),z)],.0016,closed=True),'plastic'))
    for yy in [-y,y]:parts.append((box([2*x,.002,2*z],[0,yy,0]),'plastic'))
    for xx in [-x,x]:parts.append((box([.002,2*y,2*z],[xx,0,0]),'plastic'))
    for xx in [-x/3,x/3]:parts.append((box([.0015,2*y,1.8*z],[xx,0,-.001]),'plastic'))
    parts.append((box([2*x,.0015,1.8*z],[0,0,-.001]),'plastic'))
    rng=np.random.default_rng(49)
    for row in [-1,1]:
        for col in [-1,0,1]:
            mat={(-1,-1):'red',(-1,0):'plug_gray',(-1,1):'yellow',(1,-1):'silver',(1,0):'cyan',(1,1):'silver'}[(row,col)]
            items=[]
            for i in range(4):
                length=.022 if mat=='silver' else .020;r=.0018 if mat=='silver' else .003
                zz=np.linspace(-length/2,length/2,30);rs=r*(.83+.17*np.cos(np.linspace(0,10*np.pi,30)))
                item=lathe(list(zip(zz,rs)),n=24)
                if mat=='silver':
                    head=disk(.0035,.0015,[0,0,length/2]);item=trimesh.util.concatenate([item,head])
                rotation=trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0])
                item.apply_transform(rotation);item.apply_transform(trimesh.transformations.rotation_matrix(rng.uniform(-.45,.45),[0,0,1]))
                item.apply_translation([col*x*.65+rng.uniform(-.010,.010),row*y*.5+(i-1.5)*.009,-z+.009+(i%2)*.003]);items.append(item)
            parts.append((trimesh.util.concatenate(items),mat))
    # The real organizer is open: a raised clear lid leaves the contents visible.
    angle=np.deg2rad(75);rotation=trimesh.transformations.rotation_matrix(angle,[1,0,0]);center=[0,y+y*np.cos(angle),z+y*np.sin(angle)]
    lid=box([2*x*.99,2*y*.99,.001]);lid.apply_transform(rotation);lid.apply_translation(center);parts.append((lid,'lid'))
    frame=np.array([[-x,-y,0],[x,-y,0],[x,y,0],[-x,y,0]])
    rim=tube(frame,.0012,n=8,closed=True);rim.apply_transform(rotation);rim.apply_translation(center);parts.append((rim,'plastic'))
    parts.append((box([.026,.010,.006],[0,-y-.004,z-.003]),'red'))
    for xx in [-x*.65,x*.65]:parts.append((box([.018,.006,.006],[xx,y,z]),'plastic'))
    return parts

def knob(size):
    r,h=size[:2]
    return [(lathe([(-h,r*.55),(-.7*h,r*.9),(.5*h,r),(.9*h,r*.85),(h,r*.3)]),'rubber'),(ring(r*.96,r*.96,0,.001),'black')]

report={}
for layout in ['nail_ep5','nail_ep1','nail_ep2','nail_ep3']:
    model=mujoco.MjModel.from_xml_path(str(PREV/f'{layout}.xml'));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    tree=ET.parse(PREV/f'{layout}.xml');root=tree.getroot();asset=root.find('asset')
    for tex in ['yellow_label','orange_label','spray_label','cup_print','fabric','lining','pouch']:
        ET.SubElement(asset,'texture',name='detail_'+tex,type='2d',file=str(A/f'{tex}.png'))
    for name,attrs in MATERIALS.items():ET.SubElement(asset,'material',name='clutter_'+name,**attrs)
    inventory=[]
    for body in root.findall('.//body'):
        name=body.get('name','')
        if body.find('freejoint') is None or name.startswith('clipper'):continue
        collider=next(g for g in body.findall('geom') if g.get('group')=='3')
        size=np.fromstring(collider.get('size'),sep=' ')
        bid=model.body(name).id;rad=data.xmat[bid].reshape(3,3).T@(data.cam_xpos[model.camera('ext').id]-data.xpos[bid]);phi=np.arctan2(rad[1],rad[0])
        if name.startswith('bottle') or name=='spray_bottle_lying':parts=bottle(size,name,phi)
        elif name=='blue_pouch':parts=pouch(size)
        elif name=='headphone_case_closed':parts=headphones(size,'closed')
        elif name.startswith('headphone_half'):parts=headphones(size,'base' if name.endswith('a') else 'lid')
        elif name=='paper_cup':parts=cup(size,phi)
        elif name=='tape_roll':parts=tape(size)
        elif name=='bits_box':parts=organizer(size)
        elif name=='mic_knob':parts=knob(size)
        else:raise ValueError('Unhandled object: '+name)
        # Retain original invisible visual geoms because some have nonzero inferred mass.
        for g in body.findall('geom'):
            if g is not collider and g.get('group')!='3':g.set('group','5')
        faces=0
        for i,(m,mat) in enumerate(parts):
            key=f'{layout}_{name}_{i:02d}';path=A/f'{key}.obj'
            uv=np.asarray(m.visual.uv) if hasattr(m.visual,'uv') and m.visual.uv is not None else np.full((len(m.vertices),2),.5)
            # Keep explicitly authored winding, including inward-facing cup and tape walls.
            normals=m.vertex_normals
            with path.open('w') as f:
                f.writelines('v %.8f %.8f %.8f\n'%tuple(v) for v in m.vertices)
                f.writelines('vt %.7f %.7f\n'%tuple(v) for v in uv)
                f.writelines('vn %.7f %.7f %.7f\n'%tuple(v) for v in normals)
                f.writelines('f '+' '.join(f'{j+1}/{j+1}/{j+1}' for j in face)+'\n' for face in m.faces)
            ET.SubElement(asset,'mesh',name=key,file=str(path),inertia='shell')
            ET.SubElement(body,'geom',name=key+'_visual',type='mesh',mesh=key,material='clutter_'+mat,group='2',contype='0',conaffinity='0',mass='0')
            faces+=len(m.faces)
        inventory.append(dict(name=name,visual_parts=len(parts),triangles=faces,legacy_collision_size=size.tolist()))
    scene=ROOT/f'{layout}.xml';ET.indent(tree);tree.write(scene,encoding='utf-8',xml_declaration=True)
    cfg=yaml.safe_load((PREV/f'config_{layout}.yaml').read_text());cfg['env']['robot']['scene']=str(scene)
    (ROOT/f'config_{layout}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));report[layout]=inventory
    print(layout,len(inventory),'objects upgraded')
(ROOT/'asset_inventory.json').write_text(json.dumps(report,indent=2))

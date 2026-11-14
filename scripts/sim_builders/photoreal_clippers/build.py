"""Reference-guided visual upgrade. Writes only beside this script; source assets are read-only.
Run: .venv/bin/python -B lerobot/scripts/sim_builders/photoreal_clippers/build.py
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import json, cv2, numpy as np, trimesh, yaml
ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[3]
SOURCE=WORKSPACE/'lerobot/src/lerobot/robots/rebot_b601_sim/scenes'
ASSETS=ROOT/'assets'; ASSETS.mkdir(exist_ok=True)
RNG=np.random.default_rng(47)

def write_mesh(name, mesh, uv=None, normals_override=None, fix_normals=True):
    if fix_normals: mesh.fix_normals()
    v=np.asarray(mesh.vertices); f=np.asarray(mesh.faces); normals=np.asarray(mesh.vertex_normals) if normals_override is None else normals_override
    if uv is None: uv=np.full((len(v),2),(.5,.025))
    path=ASSETS/f'{name}.obj'
    with path.open('w') as out:
        out.write('# Reference-guided visual mesh; metres, +x cutting head\n')
        out.writelines('v %.8f %.8f %.8f\n'%tuple(p) for p in v)
        out.writelines('vt %.7f %.7f\n'%tuple(p) for p in uv)
        out.writelines('vn %.7f %.7f %.7f\n'%tuple(p) for p in normals)
        out.writelines('f '+' '.join(f'{i+1}/{i+1}/{i+1}' for i in face)+'\n' for face in f)
    return path

def plate(name,xs,widths,zs,thick,bevel=.00018,textured=False,uv_bounds=None,yoffset=0):
    # Eight-vertex bevelled cross sections. Topography follows the real folded lever.
    verts=[]
    for x,w,z in zip(xs,widths,zs):
        b=min(bevel,thick*.42)
        for y,dz in [(-w/2+b,0),(w/2-b,0),(w/2,-b),(w/2,-thick+b),(w/2-b,-thick),(-w/2+b,-thick),(-w/2,-thick+b),(-w/2,-b)]:
            verts.append((x,y+yoffset,z+dz))
    f=[]
    for k in range(len(xs)-1):
        for j in range(8):
            a=k*8+j; b=k*8+(j+1)%8; c=(k+1)*8+j; d=(k+1)*8+(j+1)%8
            f.extend([(a,c,b),(b,c,d)])
    f += [(0,j+1,j) for j in range(1,7)]
    off=(len(xs)-1)*8; f += [(off,off+j,off+j+1) for j in range(1,7)]
    m=trimesh.Trimesh(verts,f,process=False)
    uv=np.full((len(verts),2),(.5,.025))
    if textured:
        for i,(x,y,z) in enumerate(verts):
            if i%8 in [0,1]:
                lo,hi=uv_bounds or (xs[0],xs[-1])
                uv[i]=((x-lo)/(hi-lo),.12+.88*((y-yoffset)/max(widths)+.5))
    return write_mesh(name,m,uv)

def make_texture(name,quad):
    # Source coordinates measured on the 1000x750 reference crop; all pixels come from nail.mp4.
    im=cv2.imread(str(ROOT/'references/clipper_top_detail.jpg'))
    quad=np.array(quad,np.float32)*[im.shape[1]/1000,im.shape[0]/750]
    dst=np.array([[1023,0],[1023,191],[0,191],[0,0]],np.float32)
    H=cv2.getPerspectiveTransform(quad.astype(np.float32),dst)
    raw=cv2.warpPerspective(im,H,(1024,192),flags=cv2.INTER_CUBIC)
    cv2.imwrite(str(ASSETS/f'{name}_reference_rectified.png'),raw)
    grey=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY).astype(float)
    # Remove broad illumination; preserve the observed dark lettering. Reflective grooves are geometry.
    smooth=cv2.GaussianBlur(grey,(0,0),17)
    detail=np.clip(grey/np.maximum(smooth,30),.26,1.10)
    detail[:,:300]=1; detail[:,935:]=1
    detail[:12]=1; detail[-12:]=1
    noise=RNG.normal(0,1.0,(192,1024))
    brushing=cv2.GaussianBlur(noise,(25,1),0)
    tex=np.clip(175*detail+brushing*3,0,255).astype(np.uint8)
    rgb=np.stack([tex*.98,tex,tex*1.01],-1).clip(0,255).astype(np.uint8)
    rgb=np.vstack([rgb,np.full((26,1024,3),175,np.uint8)])
    cv2.imwrite(str(ASSETS/f'{name}_lever.png'),rgb)
    return dict(source='sim/nail.mp4',time_s=18.6,quad_in_reference_crop=quad.tolist(),homography=H.tolist())

source_info={}
for name,L,W,H in [('clipper_large',.075,.012,.010),('clipper_small',.053,.012,.009)]:
    quad=([[748,141],[850,166],[800,615],[697,595]] if name.endswith('large') else [[211,80],[298,103],[223,425],[137,402]])
    source_info[name]=make_texture(name,quad)
    xs=np.linspace(-.5*L,.49*L,110); t=(xs/L+.5)/.99
    widths=W*(.83+.17*np.sin(np.pi*t/2)); widths[:6]*=np.linspace(.88,1,6)
    z=-H/2+.0013+.0011*t**3
    plate(name+'_lower',xs,widths,z,.0011)
    xs=np.linspace(-.47*L,.465*L,110); t=(xs/L+.47)/.935
    z=-H/2+.0018+H*.54*np.sin(t*np.pi/2)
    plate(name+'_spring',xs,W*(.79+.19*t),z,.0009)
    xs=np.linspace(-.455*L,.385*L,220); t=(xs/L+.455)/.84
    z=-H/2+.0032+(H-.0042)*t+.00035*np.sin(np.pi*t)
    # Four (small) / five (large) embossed transverse grip ribs, measured from the footage.
    for xc in np.linspace(-.405*L,-.225*L,5 if name.endswith('large') else 4):
        z+=.00048*np.exp(-((xs-xc)/(.0125*L))**2)
    widths=W*(.81+.025*np.sin(np.pi*t)); widths[:6]*=np.linspace(.93,1,6); widths[-6:]*=np.linspace(1,.90,6)
    keep=xs<=.285*L
    plate(name+'_lever',xs[keep],widths[keep],z[keep],.0010,textured=True,uv_bounds=(-.455*L,.385*L))
    for sign,side in [(-1,'left'),(1,'right')]:
        xx=np.linspace(xs[keep][-1],.385*L,30); tt=(xx/L+.455)/.84
        zz=-H/2+.0032+(H-.0042)*tt+.00035*np.sin(np.pi*tt)
        plate(name+'_fork_'+side,xx,np.full(len(xx),W*.22),zz,.001,yoffset=sign*W*.295)
    # The front edge curves back at the centre, as on the actual concave cutting jaws.
    for suffix,zz in [('jaw_lower',-H/2+.0023),('jaw_upper',H*.06)]:
        v=[]; n=41
        for k,y in enumerate(np.linspace(-W*.49,W*.49,n)):
            front=L*.50-.0012*(1-(y/(W*.49))**2)
            for x,dz in [(front-.0022,0),(front,0),(front,-.0011),(front-.0022,-.0011)]: v.append((x,y,zz+dz))
        f=[]
        for k in range(n-1):
            for j in range(4):
                a=k*4+j;b=k*4+(j+1)%4;c=(k+1)*4+j;d=(k+1)*4+(j+1)%4;f.extend([(a,b,c),(b,d,c)])
        f += [(0,2,1),(0,3,2),(4*(n-1),4*(n-1)+1,4*(n-1)+2),(4*(n-1),4*(n-1)+2,4*(n-1)+3)]
        write_mesh(name+'_'+suffix,trimesh.Trimesh(v,f,process=False))
    # Pivot shaft and transverse hinge pin; no new joints or collision surfaces.
    m=trimesh.creation.cylinder(radius=.0013,height=H*.74,sections=32); m.apply_translation([L*.345,0,-H*.05]); write_mesh(name+'_pivot',m)
    m=trimesh.creation.cylinder(radius=.00125,height=W*.80,sections=32); m.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0])); m.apply_translation([L*.345,0,H*.40]); write_mesh(name+'_pin',m)

from case_visual import build_case
case_info=build_case(SOURCE,ASSETS,write_mesh)
(ASSETS/'case_provenance.json').write_text(json.dumps(case_info,indent=2))

for layout in ['nail_ep5','nail_ep1','nail_ep2','nail_ep3']:
    tree=ET.parse(SOURCE/layout/'scene.xml'); root=tree.getroot()
    root.find('compiler').set('assetdir',str(SOURCE))
    for e in root.findall('include'): e.set('file',str((SOURCE/layout/e.get('file')).resolve()))
    asset=root.find('asset')
    for name in ['clipper_large','clipper_small']:
        ET.SubElement(asset,'texture',name=name+'_detail',type='2d',file=str(ASSETS/f'{name}_lever.png'))
        ET.SubElement(asset,'material',name=name+'_detail',texture=name+'_detail',specular='.85',shininess='.65')
        ET.SubElement(asset,'material',name=name+'_steel',rgba='.56 .58 .60 1',specular='.95',shininess='.75')
        ET.SubElement(asset,'material',name=name+'_edge',rgba='.30 .32 .34 1',specular='.9',shininess='.7')
        body=root.find(f".//body[@name='{name}']")
        for e in list(body):
            if e.tag=='geom' and e.get('mesh')==name+'_photo': body.remove(e)
        for part in ['lower','spring','lever','fork_left','fork_right','jaw_lower','jaw_upper','pivot','pin']:
            key=name+'_'+part
            ET.SubElement(asset,'mesh',name=key,file=str(ASSETS/f'{key}.obj'))
            mat=name+('_detail' if part=='lever' else '_edge' if part in ['jaw_lower','jaw_upper','pivot'] else '_steel')
            ET.SubElement(body,'geom',name=key+'_visual',type='mesh',mesh=key,material=mat,group='2',contype='0',conaffinity='0',mass='0')
    asset.find("texture[@name='clipper_case_photo_tex']").set('file',str(ASSETS/'case_surface.png'))
    case=root.find(".//body[@name='clipper_case']")
    for geom in list(case):
        if geom.tag=='geom' and geom.get('mesh') in ['clipper_case_photo','clipper_case_metal_photo']:case.remove(geom)
    for name,rgba,spec,shine in [('case_slot','.50 .49 .52 1','0','0'),('case_foam','.76 .74 .78 1','0','0'),('case_tin','.54 .56 .58 1','.8','.65'),('case_dark','.13 .15 .17 1','.35','.4')]:
        ET.SubElement(asset,'material',name=name,rgba=rgba,specular=spec,shininess=shine)
    for part,mat in [('case_slot_floors','case_slot'),('case_foam_top','clipper_case_photo_mat'),('case_foam_sides','case_foam'),('case_tin','case_tin'),('case_dark','case_dark'),('case_rolled_rim','case_tin')]:
        ET.SubElement(asset,'mesh',name=part,file=str(ASSETS/f'{part}.obj'),inertia='shell')
        ET.SubElement(case,'geom',name=part+'_visual',type='mesh',mesh=part,material=mat,group='2',contype='0',conaffinity='0',mass='0')
    for name in ['clipper_case_metal_photo_mat']:
        mat=asset.find(f"material[@name='{name}']"); mat.set('specular','.7');mat.set('shininess','.55')
    visual=root.find('visual'); visual.find('headlight').attrib.update(diffuse='.15 .15 .15',ambient='.37 .37 .37',specular='.10 .10 .10')
    visual.find('quality').set('shadowsize','4096')
    wb=root.find('worldbody')
    for e in list(wb):
        if e.tag=='light': wb.remove(e)
    # A low-contrast shadow light plus broad unshadowed fill avoids the old black arm-shadow patch.
    for attrs in [dict(name='detail_key',pos='.3 -.3 2',dir='-.1 .15 -1',diffuse='.40 .40 .40',specular='.8 .8 .8',castshadow='false'),dict(name='detail_shadow',pos='.3 -.3 2',dir='-.1 .15 -1',diffuse='.18 .18 .18',specular='.0 .0 .0',castshadow='true'),dict(name='detail_fill',pos='1.5 1 1.5',dir='-.6 -.5 -.6',diffuse='.16 .16 .16',specular='.45 .45 .45',castshadow='false')]:
        ET.SubElement(wb,'light',directional='true',**attrs)
    shadow=wb.find("light[@name='detail_shadow']"); wb.remove(shadow)
    for i,(dx,dy) in enumerate([(-.07,-.07),(-.07,.07),(.07,-.07),(.07,.07)]):
        ET.SubElement(wb,'light',name=f'soft_shadow_{i}',directional='true',pos='.3 -.3 2',dir=f'{-.1+dx} {.15+dy} -1',diffuse='.045 .045 .045',specular='0 0 0',castshadow='true')
    visual.find('global').attrib.update(offwidth='1280',offheight='960')
    scene=ROOT/f'{layout}.xml'; ET.indent(tree);tree.write(scene,encoding='utf-8',xml_declaration=True)
    # Run the inference entry point with this config; the original eval wrapper overrides the scene.
    config=yaml.safe_load((WORKSPACE/'config_rl_sim.yaml').read_text())
    config['env']['robot']['scene']=str(scene)
    config['env']['robot']['cameras']['top']['background']=str(SOURCE/layout/'background_top.png')
    (ROOT/f'config_{layout}.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
(ASSETS/'provenance.json').write_text(json.dumps(source_info,indent=2))
print('Wrote visual assets, four scene variants and four configs to',ROOT)

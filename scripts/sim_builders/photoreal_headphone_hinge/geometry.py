"""Small mesh construction helpers. All coordinates are in metres."""
import numpy as np,trimesh

def mesh(v,f,uv=None):
    m=trimesh.Trimesh(np.asarray(v),np.asarray(f),process=False)
    if uv is not None:m.visual=trimesh.visual.TextureVisuals(uv=np.asarray(uv))
    return m

def lathe(profile,n=96,caps=True,ribs=0):
    v=[];uv=[];f=[];lo=min(z for z,r in profile);hi=max(z for z,r in profile)
    for z,r in profile:
        for j in range(n+1):
            t=j/n*2*np.pi;rr=r*(1+.015*np.cos(ribs*t)) if ribs else r
            v.append([rr*np.cos(t),rr*np.sin(t),z]);uv.append([j/n,(z-lo)/max(hi-lo,1e-9)])
    for i in range(len(profile)-1):
        for j in range(n):
            a=i*(n+1)+j;b=a+1;c=a+n+1;d=c+1;f.extend([(a,b,c),(b,d,c)])
    if caps:
        for i,up in [(0,False),(len(profile)-1,True)]:
            c=len(v);v.append([0,0,profile[i][0]]);uv.append([.5,.5])
            for j in range(n):
                a=i*(n+1)+j;b=a+1;f.append((c,a,b) if up else (c,b,a))
    return mesh(v,f,uv)

def outline(hx,hy,n=128,power=3):
    t=np.linspace(0,2*np.pi,n,endpoint=False);c=np.cos(t);s=np.sin(t)
    return np.c_[hx*np.sign(c)*abs(c)**(2/power),hy*np.sign(s)*abs(s)**(2/power)]

def shaped(hx,hy,profile,power=3,caps=True):
    p=outline(hx,hy,power=power);n=len(p);v=[];uv=[];f=[]
    for z,scale in profile:
        for x,y in p*scale:v.append([x,y,z]);uv.append([x/(2*hx)+.5,y/(2*hy)+.5])
    for i in range(len(profile)-1):
        for j in range(n):
            a=i*n+j;b=i*n+(j+1)%n;c=(i+1)*n+j;d=(i+1)*n+(j+1)%n;f.extend([(a,b,c),(b,d,c)])
    if caps:
        for i,up in [(0,False),(len(profile)-1,True)]:
            c=len(v);v.append([0,0,profile[i][0]]);uv.append([.5,.5])
            for j in range(n):
                a=i*n+j;b=i*n+(j+1)%n;f.append((c,a,b) if up else (c,b,a))
    return mesh(v,f,uv)

def tube(path,r,n=8,closed=False):
    path=np.asarray(path,float);v=[];f=[]
    for i,p in enumerate(path):
        t=path[(i+1)%len(path)]-path[(i-1)%len(path)] if closed else path[min(i+1,len(path)-1)]-path[max(0,i-1)]
        t=t/np.linalg.norm(t);axis=np.array([0.,0.,1.])
        if abs(t@axis)>.9:axis=np.array([0.,1.,0.])
        u=np.cross(t,axis);u/=np.linalg.norm(u);w=np.cross(t,u)
        for a in np.linspace(0,2*np.pi,n,endpoint=False):v.append(p+r*(u*np.cos(a)+w*np.sin(a)))
    for i in range(len(path) if closed else len(path)-1):
        for j in range(n):
            a=i*n+j;b=i*n+(j+1)%n;c=((i+1)%len(path))*n+j;d=((i+1)%len(path))*n+(j+1)%n;f.extend([(a,b,c),(b,d,c)])
    return mesh(v,f)

def box(size,pos=(0,0,0)):
    m=trimesh.creation.box(extents=size);m.apply_translation(pos);return m

def ellipsoid(size,pos=(0,0,0)):
    m=trimesh.creation.icosphere(subdivisions=3);m.apply_scale(size);m.apply_translation(pos);return m

def disk(radius,height,pos=(0,0,0),sections=48):
    m=trimesh.creation.cylinder(radius=radius,height=height,sections=sections);m.apply_translation(pos);return m

def ring(rx,ry,z,r=.001):
    p=outline(rx,ry,n=128,power=2);return tube(np.c_[p,np.full(len(p),z)],r,closed=True)

"""Render clippers_only ext view with axis markers and print bounding boxes of the hinged headphone meshes."""
import json, sys, numpy as np, mujoco
from pathlib import Path
from PIL import Image, ImageDraw
WS = Path("/home/user/Documents/Research/RL/LeRobot"); S = WS/"lerobot/src/lerobot/robots/rebot_b601_sim/scenes"; OUT = Path(__file__).resolve().parent
model = mujoco.MjModel.from_xml_path(str(S/"clippers_only/scene.xml")); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
r = mujoco.Renderer(model, 480, 640); r.update_scene(data, "ext"); img = r.render()
cid = model.camera("ext").id; R = data.cam_xmat[cid].reshape(3,3); o = data.cam_xpos[cid]; f = 240/np.tan(np.deg2rad(model.cam_fovy[cid])/2)
def proj(p):
    c = R.T @ (np.asarray(p)-o); return (320 + f*c[0]/-c[2], 240 - f*c[1]/-c[2])
im = Image.fromarray(img); d = ImageDraw.Draw(im)
for p, lab in [((0,0,0),"origin"),((0.3,0,0),"x=.3"),((0.3,0.2,0),"x.3 y.2"),((0.3,-0.2,0),"x.3 y-.2"),((0.5,0,0),"x=.5"),((0.3670,0.0174,0.0),"clipper_case")]:
    u,v = proj(p); d.ellipse((u-4,v-4,u+4,v+4), fill="red"); d.text((u+6,v-6), lab, fill="yellow")
im.save(OUT/"clippers_only_ext_axes.png"); print("cam", o, "saved")
meta = json.load(open(WS/"lerobot/scripts/sim_builders/photoreal_headphone_hinge/hinge_metadata.json"))
for ep in ["nail_ep1","nail_ep2"]:
    m = meta[ep]; print("==", ep, "size", m["size"], "angle", m["angle"], "pivot", m["pivot_local"], "axis", m["axis_local"])
    n = m["parts"]; groups = {"base": [i for i in range(n) if f"hinged_{ep}_{i:02d}" not in m["lid_geoms"]+m["hinge_geoms"]], "lid": [int(g[-2:]) for g in m["lid_geoms"]], "hinge": [int(g[-2:]) for g in m["hinge_geoms"]]}
    ang = m["angle"]; Rz = np.array([[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0],[0,0,1]])
    for k, idx in groups.items():
        V = []
        for i in idx:
            V.append(np.array([[float(x) for x in l.split()[1:4]] for l in open(S/f"assets/photoreal/headphones/hinged_{ep}_{i:02d}.obj") if l.startswith("v ")]))
        V = np.vstack(V); L = V @ Rz  # un-rotate into the builder's canonical frame (hinge along +x at +y)
        print(f"  {k}: n={len(V)} world-local bbox min {V.min(0).round(4)} max {V.max(0).round(4)} | canonical bbox min {L.min(0).round(4)} max {L.max(0).round(4)}")

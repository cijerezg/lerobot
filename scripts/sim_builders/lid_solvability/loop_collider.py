"""Fit capsule segments along the loop cord (hinged_nail_ep2_21) and print <geom> lines in the headphone_case body frame.
Reads the scene (no write): the lines are pasted into scene.xml by hand."""
import mujoco, numpy as np, json
from pathlib import Path
WS = Path(__file__).resolve().parents[4]; OUT = Path(__file__).resolve().parent
m = mujoco.MjModel.from_xml_path(str(WS / "lerobot/src/lerobot/robots/rebot_b601_sim/scenes/close_lid/scene.xml")); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
case = m.body("headphone_case").id
g = m.geom("hinged_nail_ep2_21").id; mid = m.geom_dataid[g]
v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
gb = m.geom("headphone_case_block_a").id; Rc = d.geom_xmat[gb].reshape(3, 3); oc = d.geom_xpos[gb]
can = (w - oc) @ Rc  # canonical shell frame: x along the hinge, -y = outward past the shell edge, z up
# centreline: bin by the angle around the loop's attachment midpoint, take the mean vertex per bin
root = np.array([0.0, can[:, 1].max(), 0.0])
ang = np.arctan2(can[:, 1] - root[1], can[:, 0] - root[0])  # from +x (one leg) through -y (apex) to -x
order = np.argsort(ang); bins = np.linspace(ang.min(), ang.max(), 8)
centres = []
for lo, hi in zip(bins[:-1], bins[1:]):
    sel = can[(ang >= lo) & (ang <= hi)]
    if len(sel): centres.append(sel.mean(0))
centres = np.array(centres)
# cord radius: half the thickness of the band = rms distance of vertices from the centreline polyline
def dist_poly(p):
    best = 1e9
    for a, b in zip(centres[:-1], centres[1:]):
        t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1); best = min(best, np.linalg.norm(p - (a + t * (b - a))))
    return best
rad = np.array([dist_poly(p) for p in can])
print("centreline (canonical):\n", centres.round(4)); print("vertex distance from centreline: median %.4f, 90%% %.4f, max %.4f" % (np.median(rad), np.percentile(rad, 90), rad.max()))
R_case = d.xmat[case].reshape(3, 3); o_case = d.xpos[case]
loc = ((centres @ Rc.T) + oc - o_case) @ R_case  # case body frame
radius = 0.0025
lines = []
for i, (a, b) in enumerate(zip(loc[:-1], loc[1:])):
    lines.append(f'<geom name="headphone_case_loop_{i}" type="capsule" group="3" size="{radius:.4f}" fromto="{a[0]:.5f} {a[1]:.5f} {a[2]:.5f} {b[0]:.5f} {b[1]:.5f} {b[2]:.5f}" mass="0.0010" condim="4" friction="0.8 0.005 0.0001" />')
print("\n".join(lines))
(OUT / "loop_capsules.xml.txt").write_text("\n".join(lines) + "\n")
json.dump(dict(centreline_canonical=centres.round(5).tolist(), centreline_case_local=loc.round(5).tolist(), world=((centres @ Rc.T) + oc).round(5).tolist(), radius=radius), open(OUT / "loop_capsules.json", "w"), indent=1)

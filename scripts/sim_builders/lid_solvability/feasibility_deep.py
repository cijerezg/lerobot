"""Deeper multi-start check (60 random starts x tool candidates every 15 deg across the band) at a few hinge angles for the side-rim pinch."""
import json, sys
sys.argv = [sys.argv[0]]
import importlib.util
spec = importlib.util.spec_from_file_location("feas", __file__.replace("feasibility_deep.py", "feasibility.py"))
# re-use the module's model + ik without re-running its scan: exec only the definitions
src = open(__file__.replace("feasibility_deep.py", "feasibility.py")).read().split("FAMILIES = {")[0]
exec(src)
out = {}
for fam_name, F in {"side_xp_far_end": fam_side(xp_y, 0.235), "side_xp_near_hinge": fam_side(xp_y, 0.30), "far_rim_centre": fam_far(0.0174)}.items():
    res = {}
    for th in (1.2, 1.5, 1.85):
        P = H + rot(AX, th) @ (F["P0"] - H)
        tools = [rot(AX, th + np.deg2rad(a)) @ DOWN for a in range(-60, 61, 15)] if "side" in fam_name else [np.cos(np.deg2rad(a)) * lid_down(th) + np.sin(np.deg2rad(a)) * Y for a in range(-75, 76, 15)]
        best = None
        for tool_dir in tools:
            for _ in range(60):
                q, ep, es, et = ik(P, F["slide"](th), tool_dir, LO + rng.random(6) * (HI - LO), iters=150)
                ok = ep < 0.003 and es < 3 and et <= 8
                key = (0 if ok else 1, ep)
                if best is None or key < best[0]: best = (key, ok, round(ep, 4), round(es, 1), round(et, 1), dict(zip(NAMES, np.degrees(q).round(0).tolist())))
                if ok: break
            if best[1]: break
        res[str(th)] = dict(feasible=best[1], pos_err=best[2], slide_err_deg=best[3], tool_err_deg=best[4], q_deg=best[5]); print(fam_name, th, res[str(th)])
    out[fam_name] = res
json.dump(out, open(OUT / "feasibility_deep.json", "w"), indent=1)

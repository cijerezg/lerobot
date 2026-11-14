# real2sim 2026-10-17: phone video -> MuJoCo asset

First pass of the object reconstruction pipeline. Two objects built from two phone clips
(`sim/headphones.mp4`, `sim/nail.mp4`), both on a US Letter sheet (279.4 x 215.9 mm) for scale.
Everything runs in the ROOT venv (`uv run python ...` from the workspace root). Installed today
for this: vggt (git), open3d 0.20, coacd 1.0.14, trimesh 5.1.1.

## Pipeline (scripts/)

| step | script | in -> out |
|---|---|---|
| 1 | `extract_frames.py video out --every N` | ~150 sharp frames (Laplacian >= 0.5 x median), 1036 px wide |
| 2 | `run_vggt.py frames out.npz` | VGGT-1B: poses, intrinsics, depth, confidence, 518 px images |
| 3 | `table_frame.py work` | RANSAC table plane, table frame (z up), `cloud_table.ply`, ortho renders |
| 4 | `sheet_scale.py work` | Letter sheet = known-aspect rectangle search on the table layer: maximise paper pixels minus wood pixels inside, hole (object footprint) must be inside. `sheet.npz` (scale m/unit, box), `sheet_fit.png` |
| 5 | `tsdf_fuse.py work` | uniform TSDF (2 mm) centred on the sheet; `mesh_object.ply` in metres, z=0 at the table; `mesh_views.jpg` |
| 6a | `split_halves.py work` | clamshell: clusters standing on the sheet, seam = silhouette waist along the major axis, base = half with more volume above 3 cm; `base.ply`, `lid.ply`, `hinge.json` |
| 6b | `build_clamshell_asset.py work assets/<n> <n>` | body-frame OBJs, CoACD hulls (24 per half), hinge at mean rim height (a pi flip about height h maps z -> 2h - z), axis sign chosen so the lid rises; `<n>.xml` |
| 6c | `build_rigid_asset.py work assets/<n> <n> --scene_pos X Y [--box ...]` | one rigid body from the largest on-sheet cluster + primitive boxes placed in sheet-relative coordinates |
| 7 | `test_scene.py asset.xml out --pos X Y Z [--hinge J]` | asset in `scene_dm.xml` (floor = table, robot at origin), settle, optional hinge drive to pi, `test.mp4` + stills + `log.csv` |

Gotchas found: `ScalableTSDFVolume` returns an empty mesh in this open3d build (uniform volume works).
Visual mesh geoms need `mass="0"` or MuJoCo gives the shell a water-density convex volume (2.3 kg).
Hull masses are explicit per hull (mass share by trimesh volume). EGL rendering works with
`MUJOCO_GL=egl`; an EGLError at interpreter exit is cleanup noise.

## Results

| object | clip | frames | scale m/unit | object size (mm) | asset |
|---|---|---|---|---|---|
| headphone case, open | headphones.mp4, 102 s | 148 | 0.5077 | base 172 x 247 x 59, lid 192 x 217 x 48; hinge height 42 | `assets/headphone_case/headphone_case.xml` (free base + hinge lid, 0.30 kg nominal) |
| nail clipper case, open tin + foam | nail.mp4, 105 s | 150 | 0.4538 | 88 x 119 x 15 | `assets/clipper_case/clipper_case.xml` (rigid, 32 hulls, 0.06 kg nominal) |
| nail clippers (2) | same | | | large 75 x 12 x 10, small 53 x 12 x 9 (footprints measured on the wrist depth sidecar over 11 and 14 frames, object_photo_texture.py; the scan-cloud lengths 80 and 60 were 7 and 13 % long) | boxes in the same xml, 25 g and 15 g nominal |

Test runs: `work/headphones/test/test.mp4` (lid closes to 3.14 rad in 4 s, base drifts 6 mm),
`work/nail/test/test.mp4` (settle only). Stills: `still_start.jpg`, `still_mid.jpg`, `still_end.jpg`,
`*_hulls.jpg` (collision groups), `montage.jpg`. Each still is oblique | top | close.
Subagent report for the nail clip: `work/nail/REPORT.md`.

## Known limits

- Scale assumes the sheet is Letter; the aspect is fixed in the fit, so the ratio is no longer an
  independent check. `sheet_fit.png` is the check.
- Visual meshes are flat grey in MuJoCo (OBJ carries no vertex colours); the coloured PLYs are kept.
- The clippers are boxes. Shiny metal does not fuse; the cloud gives lengths only.
- The hinge is a single rigid pivot for a fabric seam. Closed-state scan of each case is still
  missing (would give the real axis and the closed geometry).
- Masses are nominal; the user was asked for kitchen-scale values.
- Not done yet: external/wrist camera calibration, object placement from the real episodes,
  photoreal rendering, closed-loop policy evaluation.

## Headphone case, second pass (2026-10-17 afternoon)

User feedback on the first asset: the pull flap on the lid was missing and the sim looked broken (one-sided TSDF shells).

- Flap: a stiff fabric tab with a slot standing up from the outer edge of the LID (the robot pinches it to close the case; see
  the wrist frames of episode 12 in `rebot_2026-10-14-annotated-v4`). It is in the raw cloud (16.6 k points above the rim)
  but the object crop lost it. `split_halves.py` now measures it from `cloud_table.ply` (`hinge.json: lid_tab`) and
  `build_clamshell_asset.py` adds it as four boxes (top bar, two posts, block that also plugs the rim notch under it):
  54 x 10 x 40 mm, slot 32 x 6 mm, standing 30..70 mm above the table at the lid's far edge (180 mm from the hinge, 1 mm off
  the centre line), visual + collision, 8 g.
- Hinge: axis = perpendicular bisector of the two wall-ring centroids (the SVD axis was 6.6 deg off, which offset the closed
  lid 20 mm sideways); pivot midway between the two outer edges (moved 6 mm towards the lid); triangles re-split at the pivot
  line so no lid triangle sits behind the hinge (a 17 mm strip there swept below the table while closing and shoved the base,
  19 mm drift); pivot height = mean of the rim heights (base 32, lid 44 -> 38 mm) with the hinge-facing bins skipped. The TSDF
  erodes the thin rim: the top wall band of each half is lifted to the raw-cloud rim height per 10 deg bin (<= 8 mm) and
  Taubin-smoothed.
- Look: holes in the shells (floor hole, cups, seam-cut slit) are fanned shut; the underside and the unseen lower outer wall
  (the phone only saw the walls from the rim down to mid-height) are closed with a flat bottom disc + a tapered can up to 80 %
  of the rim height on the hull of the mid-height wall ring. No offset-copy thickening (it poked through the rim as a crown).
- Numbers: closed state rim gap (flipped lid rim minus base rim, away from the hinge) median -0.5 mm; footprint overhang median
  -2 mm; lid and base hulls overlap up to 21 mm at the far wall / strap loop (convex-hull bulges; MuJoCo excludes
  parent-child contacts so this never fights, the joint range 0..pi is the closed stop; the tab clears the floor by 6 mm when
  closed). test_scene: hinge reaches 3.15 rad, base drift 6.8 mm during the close.
- Renders: `work/headphones/test/hp_states.jpg` (open / half / closed x 6 views, `scripts/hp_render_states.py`),
  `hp_states_hulls.jpg`, `hp_tab_closeup.jpg`, `hp_closed_closeup.jpg`, `montage.jpg`, `test.mp4`.
- Still open: the seam band between the halves is ragged (TSDF rim), the strap loop on the base is a leaning stub, the lid
  outside is a flat disc + can rather than the real dome, no textures.

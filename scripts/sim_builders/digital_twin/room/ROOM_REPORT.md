# Room around the sim desk (2026-10-29)

Files: `room.xml` (MuJoCo include, geoms only, included inside `<worldbody>`), `clippers_only_room/scene.xml` (copy of
`scenes/clippers_only` with the room, loads from here), `renders/`, `scripts/` (measure_room*.py, render_room.py, sim_check.py).
Table frame throughout: robot base at the origin, desk top z = 0, x toward the front wall / top camera, y to the image right of the top view.

## Dimensions

| item | value | status |
|---|---|---|
| desk top | 48 x 30 in = 1.219 x 0.762 m, 25 mm thick | user's desk size (2026-10-29); measured 0.74 (x) x 1.01-1.11 (y) from the top frames, the +y end is cut by the frame edge |
| desk front edge (toward the wall) | x = 0.7175 | measured (10-16 corner picks 0.710 / 0.725, matches the 10-29 top frame) |
| desk back edge (behind the base) | x = -0.0445 | chosen = front edge - 0.762 (corner picks -0.047 / -0.001; the old box used -0.024) |
| desk left edge (image left, -y) | y = -0.481 | measured (picks -0.477 / -0.485) |
| desk right edge (+y) | y = +0.610 | base centred on the long side (user 2026-10-29) |
| desk height above the floor (DESK_H) | 0.75 m, floor plane at z = -0.75 | ASSUMED; wrist depth on the floor gives z = -0.71 to -0.74 (noisy beyond 1 m) |
| front wall | x = 0.8175 (0.10 m past the edge), spans y -1.6 .. 1.0 | distance from the user; wrist depth plane fit x = 0.80 (f0 rms 16 mm, f4200 rms 7 mm, +-0.04 from the wrist calibration bias); the wall ends near y = 0.9-1.0 (floor points beyond it in f4200) |
| baseboard | 0.10 m tall, 12 mm proud of the wall, off-white | seen in the wrist frames, size assumed |
| side walls | y = -2.0 and y = +2.0 | ASSUMED; -y side: upright depth points at 1.2-1.9 m are a speaker, a box and a dark wardrobe |
| back wall | x = -3.0 | ASSUMED (chairs and floor behind the robot; keeps the skybox out of backward wrist views) |
| desk frame | two T feet 0.66 x 0.07 x 0.03 m, two 70 mm columns 0.15 m inboard of the side edges, one crossbar under the slab | assumed standing-desk frame, black |
| colours | wall rgba 1.0 0.96 0.91 (renders 127 vs real 130 grey level in wrist f0), floor 0.52 0.39 0.30 (fitted on the floor patch of wrist f2400) | fitted once on the 10-29 wrist frames; plain colours, no texture |

Which side the wall is on: the robot faces +x (the top camera is at x = 0.61 above the far half of the desk). At the park
pose the wrist camera looks at +x and sees the wall directly behind the desk edge, with the top-camera pole rising from
the far edge (`renders/real_wrist_f0.png`). Behind the robot (-x, top of the top frame) there is floor and chairs, no wall.

## Contacts

Every geom in `room.xml` has `contype="0" conaffinity="0"` (world geoms, no mass). The scene copy keeps the original
`table_contact` plane (z = 0, group 3) as the only desk collider, so the contact set is unchanged: there is no second
desk-top collider, and the plane stays the infinite one it was (an object pushed off the edge still rests at z = 0;
swapping the plane for the slab box would change the clipper contacts, see the box-vs-plane note in
`real2sim_2026-10-17/scripts/scene_from_episode.py`). The copy removes the old `floor` plane, the plain-wood `table`
box and the textured `table_top` plane; the slab box carries the same `table_top` material. The box face mapping equals
the old plane mapping (ext render diff inside the desk = the 3-6 % grain stretch to the new size only, `renders/ref_vs_room_ext.jpg`).

## Simulation check

`scripts/sim_check.py` on `clippers_only_room/scene.xml`, 1 s of mj_step (500 steps): ncon 9 -> 13 exactly as in the
reference scene, case / large / small clipper move 0.1 / 0.2 / 0.2 mm (same as the reference), no contact pair involves a
`room_*` geom, no new robot contacts.

## Renders

`scripts/render_room.py` (mujoco.Renderer 640x480, no composite, robot posed from the recorded state of
`outputs/rebot_teleop_2026-10-29-cut-v1` ep 0): `renders/side_by_side_f{0,300,2400,4200}.jpg` = real top | sim top | real wrist | sim wrist,
`renders/side_by_side_all.jpg` all four rows, `renders/sim_{top,wrist}_f*.jpg`, `renders/real_*.png` the real frames,
`renders/ref_vs_room_{ext,wrist}.jpg` old scene vs room at park. Poses: f0 park (wall ahead), f300 and f2400 looking
over the left edge at the floor and the wall corner, f4200 looking up at the wall.

## Notes

- The top camera has moved since the 10-16 calibration: in the 10-29 top frames the whole desk sits 45-55 px lower than
  the sim `ext` projection (robot base and both desk edges shift together), so `background_top.png` and the `ext` pose do
  not match the 10-29 recordings. Not touched here.
- Not modelled: the top-camera pole and clamp at the far edge, the speaker / box / wardrobe on the -y side, the power outlet.
- To use: point `scene:` of a `rebot_b601_sim` config at `lerobot/scripts/sim_builders/digital_twin/room/clippers_only_room/scene.xml`
  (`background_top.png` is copied next to it). Variants and other layouts need the same three-line edit of their `scene.xml`.

## Camera (2026-10-29)

`ext` in clippers_only_room/scene.xml is aligned with the desk on the user's request: pos (0.609, 0, 0.849), yaw and roll zero, tilt 19.6 deg from vertical toward the robot (calibrated 11.8; raised so the band below the far desk edge is a third of its former height; quat 0.696789 0.120356 0.120356 0.696789). The calibrated pose had 5 deg of yaw and a 14 mm lateral offset. The package scenes keep the calibrated camera; background_top.png composites are only valid with the calibrated one.

## Appearance pass 2026-10-30

room.xml now references the material `room_floor_mat` (plank texture) for the floor; it is defined in
`twin/appearance/materials.xml`, so any scene that includes room.xml must also include that file (asset level), or swap the
floor geom back to the plain rgba given in the comment next to it. Wall rgba 0.775 0.713 0.671 and the baseboard rgba are fits to
the real wrist frames under the light setup of `twin/appearance/lights.xml` + `lights_visual.xml` (they replace the clippers_only
lights and `<visual>`). Numbers and the before/after renders: `twin/appearance/APPEARANCE_REPORT.md`.

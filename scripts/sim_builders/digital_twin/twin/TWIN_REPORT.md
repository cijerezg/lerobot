# Digital twin, first assembly (2026-10-29)

Reference: `outputs/rebot_teleop_2026-10-29-cut-v1` ep 0, frame 0 (`room/renders/real_top_f0.png`). Table frame as in `room/ROOM_REPORT.md`:
robot base at the origin, desk top z = 0, x toward the far edge / top camera, y to the image right of the top view.

Files (all under `twin/`): `scene_twin.xml` (room + robot + aligned `ext` camera + 9 free bodies), `assets/{bits_kit,pill_bottle,pump_bottle}/`
(copies of the scanned / downloaded assets), `layout.json` (pixel picks, heights, unprojected table xy, axis angles, final poses, the
settled-pose override for `clipper_small`), `renders/` (see below). Scripts in `scripts/`: `twin_layout.py` (picks -> layout.json),
`twin_build.py` (layout.json -> scene_twin.xml, copies the assets), `twin_settle.py` (1 s settle, `--write-rest=NAME` stores a body's rest
pose as its start pose), `twin_render.py` (side-by-sides + beauty view), `twin_check_overlay.py` (projects the placed footprints back onto the
real frame). All run from the LeRobot root with `MUJOCO_GL=egl uv run python ...`; rebuild order = make_assets (v2), layout, build, settle `--write-rest=clipper_small`, build, settle, render, check_overlay.

## Camera offset

The room scene's `ext` is the ALIGNED camera (zero yaw/roll, tilt 19.6 deg), so it cannot reproduce the real frame's 3 deg desk slant or
its perspective; I unprojected the picks through the CALIBRATED camera of the package `clippers_only` scene instead (pos 0.609 -0.014 0.849,
5 deg yaw, tilt 11.8, same fovy 60.73): it has the real webcam's geometry up to the shift the webcam has had since 10-16. That shift,
measured on the robot base plate (green bar centre real (344,127) vs calibrated render (335,75)) and the far-right desk corner (real (598,73)
vs (588,29)), is real px = calibrated-render px + (+9, +50). Check on the desk-edge picks after the offset: back edge x = -0.049 / -0.060
(room -0.0445), front edge x = 0.69-0.70 (room 0.7175, 2 cm short at the far edge), right edge y = +0.615 (room +0.61), left edge y =
-0.618 / -0.624 at mid and front (room -0.61). `renders/placement_check_real_top_f0.jpg` draws every placed footprint on the real frame
through this camera + offset: kit, case, clipper, bottles and headphone case land on the real objects. The twin renders use the aligned
camera, so the twin top frame is NOT pixel-aligned with the real one (the kit looks more rotated, the basket further left); compare shapes
and relative positions, not pixels.

## Placed objects

| body | asset | source | pos (m) | yaw (deg) | real or stand-in |
|---|---|---|---|---|---|
| bits_kit | scanned open organizer, 486 x 266 x 24 mm, photo texture | `assets/bits_kit` (copied) | 0.322, -0.055, 0.003 | 245.6 (long side 65.6 deg from x; +180 puts the pocket end to the image right) | real object |
| headphone_case (+ `headphone_lid`, hinge at 0 = open) | hinged photoreal case of the 10-16 ep2 recording | package `close_lid` | 0.590, -0.290, 0.022 (base half; lid half at 0.459, -0.378) | 276.2 (lid toward the robot-left, as in the frame) | stand-in: smaller, older case; the real one is the open over-ear case with the headphones in the lower half |
| bottle | ep0 yellow-cap standing bottle, 90 mm | package `bottle_basket` | 0.248, -0.523, 0.047 | 0 | stand-in for the orange ROAR drink bottle (real ~ 2x taller, orange) |
| pill_bottle | Ibuprofen bottle 54 x 100 mm (blue cap) | `assets_dl/pill_bottle` (copied) | 0.336, -0.532, 0.002 | 0 | stand-in: real is a smaller white bottle with a white cap and a green label |
| pump_bottle | black pump bottle 61 x 180 mm | `assets_dl/pump_bottle` (copied) | 0.056, -0.512, 0.002 | 0 | stand-in for the dark blue drink bottle at the far-left corner (not in the brief's list, but on the desk) |
| basket | brown woven basket 0.29 x 0.24 x 0.21 | package `bottle_basket` | 0.496, 0.480, 0.002 | 0.9 (long side along x) | real object, EMPTY (real one holds bottles and a card) |
| clipper_case | scanned clipper tin, photo top | package `clippers_only` | 0.594, 0.129, 0.002 | 184.2 (long side along y, dark hinge half to the image left) | real object |
| clipper_large | photoreal large clipper | package `clippers_only` | 0.589, -0.118, 0.007 | 131.8 | real object (the clipper lying left of the case) |
| clipper_small | photoreal small clipper | package `clippers_only` | 0.594, 0.149, 0.0106 (settled pose) | 94.2 (in the slot nearer the image right) | real object (the clipper inside the case) |

Available but not placed (not in the frame): `assets_dl/{mug,tape_roll,power_strip,glasses_case}`. Not placed because nothing fits:
the four loose bits (`assets/bit` is a 231 x 93 mm scan, not a bit), the small clear bits box next to the basket, the grey bit holder
between the clippers and the kit, the contents of the basket, the top-camera mount at the near edge.

Yaw conventions: body yaw = angle of the asset's long side (bits_kit, basket, clipper_large: body x; clipper_case: body y, so yaw = axis - 90);
headphone case yaw = 242.18 + (base->lid angle - 180), the close_lid convention. Picks for standing bottles are the base ellipse centres; for
flat objects the top-face centre unprojected onto z = feature height (0.015-0.03 m) so parallax is accounted for.

## Asset paths

One `meshdir`/`assetdir` per MuJoCo model, and the robot include (`rebot_b601_dm_photo.xml`) needs it to be the package scenes dir, so
`scene_twin.xml` keeps `assetdir="../../../../src/lerobot/robots/rebot_b601_sim/scenes"` and references the kit and the two downloaded
bottles as `../../../../../scripts/sim_builders/digital_twin/twin/assets/<name>/<file>` from there (the downloaded XMLs use bare file
names, so the files were copied into `twin/assets/<name>/`, 20 MB, mostly the kit's 97k-face visual mesh). Package objects are pulled from
the package scenes by element name (`twin_build.py`), nothing under `lerobot/src` is touched. The room is `<include file="../room/room.xml"/>`
inside `<worldbody>`, with the room scene's `table_contact` plane, aligned `ext` camera, visual settings and lights copied over.

## Settle (1 s, 500 steps at 2 ms; `renders/settle_log.txt`)

| body | xy drift mm | dz mm | tilt deg |
|---|---|---|---|
| bits_kit | 0.0 | -4.4 | 0.09 |
| pill_bottle | 0.0 | -2.2 | 0.02 |
| pump_bottle | 0.0 | -1.2 | 0.06 |
| clipper_case | 0.0 | -2.1 | 0.02 |
| clipper_large | 0.0 | -2.2 | 0.00 |
| clipper_small | 0.2 | 0.0 | 0.77 |
| bottle | 0.0 | -2.1 | 0.00 |
| basket | 0.0 | -2.0 | 0.00 |
| headphone_case | 0.0 | -2.1 | 0.03 |

The -2 mm drops are the 2 mm start lift settling onto the plane; the kit drops 4.4 mm because its hulls' bottom sits 1 mm above its body
origin (its visual bottom ends 1.6 mm above the desk). Start penetrations > 1 mm: none between objects and the table or each other, except
`clipper_small` x `clipper_case` 1.2-1.7 mm (it rests in the foam slot; started flat at z = 0.014 it penetrated the slot hulls 2.8 mm and
tilted 2.8 deg while settling, so its 1 s rest pose is now the start pose, `rest_overrides` in layout.json). The robot's own link1/link2 x
link4 contacts (-11 to -12 mm) at the park pose are the robot model's, present in the reference scenes too. No object touches the robot.

## Renders (`renders/`)

`side_by_side_f{0,300,2400}.jpg` = real top | twin top | real wrist | twin wrist (480 x 640 each, robot posed from the recorded state),
`side_by_side_all.jpg`, `sim_top_f*.jpg`, `sim_wrist_f*.jpg`, `real_top_f*.jpg`, `beauty_oblique.jpg` (1280 x 720, free camera from the
front-left above the desk), `placement_check_real_top_f0.jpg`. f0 = park (wrist sees the case, clipper and headphone case as in the real
frame, the basket just outside the left edge); f300 the wrist looks at the two bottles on the left (real: white pill bottle in front, ROAR
behind-right; twin: same two positions, with the blue-cap and yellow-label stand-ins); f2400 the wrist looks at the dark bottle at the
far-left corner (twin: the black pump bottle at the matching spot; the real ROAR bottle had been moved into the basket by then, the twin
is the f0 layout).

## Visibly off versus the real frame

- Camera: the aligned `ext` is not the real webcam pose (no 3 deg slant, 8 deg more tilt), so the top views do not overlay; the real frame
  also shows floor, chairs and a wardrobe beyond the desk that the plain-colour room does not have.
- Scale: the ROAR bottle stand-in is about half the real height and white/yellow instead of orange; the pill bottle stand-in is a size
  larger with a blue cap; the headphone case is a different, flatter case (hard shell, no over-ear cups in the lower half).
- Colour / material: the kit renders much darker (black interior dominates, the blue rim is thin; real has bright blue walls and visible
  bit rows); the sim desk texture is redder than the real light wood; no shadows cast by the robot as in the real frame; the real wall is
  grey, the room wall off-white.
- Missing clutter: four loose bits, the clear bits box, the bit holder, the basket contents (two bottles and a card), the camera mount at
  the near edge, the cable and power strip under the desk.
- Lighting: the real frame is lit from the room's left (soft shadows toward +y), the sim from above with four faint shadow lights.

# v2 (2026-10-30): objects pass

What changed (appearance, room, lights untouched; v1 renders moved to `renders_v1/`):

- Bottles. `pill_bottle` is now `assets_dl/pill_bottle_real` (50 x 100 mm, Trader Joe's Vitamin D label), yaw 110 so the label faces the
  wrist at f300. The ROAR bottle (`roar_bottle`) is a generated asset in `twin/assets/roar_bottle/` (`twin_make_assets.py`): lathe mesh
  65 mm body with a shoulder, 20 mm neck, teal cap cylinder (rgb 0.05 0.58 0.61 sampled off the cap), 210 mm total; the label texture is the
  body silhouette of the real wrist frame 300 (quad (367,138)-(480,138)-(435,385)-(347,385)) remapped by column angle (arcsine warp of the
  visible half) onto half the circumference, the unseen half filled with the median label orange (bgr 59 81 177), 16 px seam blends;
  collision = 3 cylinders (body, neck, cap), 0.45 + 0.03 + 0.02 kg. The dark bottle at the far-left corner: the downloaded pump bottle is
  the WRONG object (wrist f2400 shows a plain navy bottle with a blue screw cap, no pump, ~70 x 200 mm), so `navy_bottle` is a generated
  lathe mesh in `twin/assets/navy_bottle/` (70 mm body, shoulder to 48 mm, blue cap 49 x 28 mm, 200 mm total, 0.5 kg, 3 cylinder hulls);
  the pump bottle stays out of this scene. `twin/assets/{pill_bottle,pump_bottle}` (v1 copies) were deleted.
- Clutter from the top frame picks (calibrated camera + the (+9, +50) px shift, `twin_layout.py`): six hex bits 25 x 6 mm (mesh
  `twin/assets/bits/bit_hex.obj`, 6 g each) — four are the visible loose bits at real px (475,126) (502,122) (489,141) (521,135), rendered
  grey, grey, red, yellow; `bit_5` (462,136) and `bit_6` (512,112) are added beside them to make six (not in the frame). The clear bits
  box (120 x 80 x 25, rgba 0.9 0.9 0.95 0.35, grey inner tray and five coloured bit blocks as visual geoms, 80 g) at pick (548,178), long
  side 78.9 deg from x. The bit holder at (322,350): the real one is silver, so it is a light-grey box 60 x 25 x 15 (rgba 0.75), not dark
  grey, 40 g, axis -39.8 deg. Basket contents, placed in the basket frame from what the top frame shows (floor top z = 12 mm, inner
  266 x 216): an upright navy bottle 64 x 200 with a white/green label band at (+0.06, +0.06), a lying silver can 60 x 130 at (-0.08, +0.02)
  along basket y, a navy card 90 x 60 x 6 at (0, -0.055) with a white card 80 x 56 x 2 on top. All free bodies with masses; the body ->
  asset map is `assets` in layout.json.
- Kit colour. `twin/assets/bits_kit/bits_kit_colors_gain.png` = the scan texture x (4.0, 2.9, 3.2) per BGR channel: the real blue rim
  (top frame f0, HSV mask, 5082 px) has mean BGR (142, 62, 31), the scan texture's blue pixels (35.5, 21.3, 9.8); after the gain they
  are (142, 61, 31); 2.6 % of texels saturate (the white bit shanks). A scalar x4 was tried first and left a cyan cast on the interior.
  `assets/bits_kit` is untouched; `twin_build.py` points the kit texture at the gained copy.
- Settle fix: the lying can rested 6 mm inside the basket floor with a cylinder collider (MuJoCo cylinder-on-box), so it has a capsule
  collider (r 30, half-length 45) under the cylinder visual.

Objects (table frame, start poses; `layout.json` poses):

| body | asset | pos (m) | yaw (deg) | mass kg | status |
|---|---|---|---|---|---|
| bits_kit | scan, texture gained | 0.322, -0.055, 0.003 | 245.6 | 0.25 | real object, colour now close |
| headphone_case (+lid) | package close_lid | 0.590, -0.290, 0.022 | 276.2 | pkg | stand-in (unchanged) |
| roar_bottle | generated, label from wrist f300 | 0.248, -0.523, 0.002 | 290 | 0.50 | real object, approximated |
| pill_bottle | assets_dl/pill_bottle_real | 0.336, -0.532, 0.002 | 110 | 0.12 | real object |
| navy_bottle | generated navy cylinder | 0.056, -0.512, 0.002 | 0 | 0.50 | real object, approximated |
| bit_1..bit_6 | bit_hex mesh | (0.064,0.316) (0.057,0.378) (0.103,0.343) (0.092,0.416) (0.088,0.284) (0.033,0.403), z 0.004 | 40 / 69 / 114 / 17 / 52 / 43 | 0.006 | 4 real + 2 added |
| bits_box | primitives | 0.204, 0.456, 0.0145 | 78.9 | 0.08 | real object, approximated |
| bit_holder | box | 0.558, -0.048, 0.0095 | 320.2 | 0.04 | real object (silver), approximated |
| basket | package bottle_basket | 0.496, 0.480, 0.002 | 0.9 | pkg | real object |
| basket_bottle / basket_can / basket_card / basket_card_white | primitives | (0.555,0.541,0.014) (0.416,0.499,0.044) (0.497,0.425,0.017) (0.497,0.425,0.022) | 0.9 / 0.9 / 30.9 / 10.9 | 0.5 / 0.35 / 0.03 / 0.01 | approximations of the basket contents |
| clipper_case / clipper_large / clipper_small | package clippers_only | as v1 | as v1 | pkg | real objects |

Settle (1 s, `renders/settle_log.txt`): every body drops its 2 mm start lift and stays (xy drift 0.0 mm, tilt < 0.1 deg) except the
bits (-1.3 mm, tilt 0.35-0.42 deg, the hex settling flat), basket_card_white -3.3 mm (onto the card below), basket_bottle -2.6, clipper_small
0.2 mm / 0.77 deg (rest override re-recorded, same as v1). Start penetrations > 1 mm: only the robot's own link contacts and
clipper_small x clipper_case 1.2-1.7 mm (foam slot, as v1). Object-object contact pairs over the second: basket x {can, bottle, card},
card x white card, case x small clipper; nothing touches the robot.

Still visibly off (renders/side_by_side_all.jpg, beauty_oblique.jpg, placement_check_real_top_f0.jpg):

- Camera / room as in v1 (aligned ext, no slant; plain room) — not this pass.
- navy_bottle at f2400: the twin wrist sees it from ~10 cm, the real one from ~30 cm; the real top f2400 shows that bottle ~30 px left of
  its f0 spot (moved during the episode), and the twin is the f0 layout. Its cap is a flat cylinder, the real cap is ribbed and taller.
- roar_bottle: the label strip covers the whole body height, the real label stops below a clear shoulder; the unseen half is flat orange;
  no transparency / liquid look. Pill bottle label is the Trader Joe's one; the wrist f300 shows a yellow-green label on that side.
- Kit: brighter and blue as the real one now, but the bit rows are white blobs (saturated scan texels) and the interior keeps the scan noise.
- Clutter: the bits box is a translucent slab with painted blocks; the real box shows a lid grid. bit_5/bit_6 do not exist in the frame.
  Basket contents are plain primitives (the real basket holds a labelled Yosemite bottle, a can and a printed card).
- Headphone case still the smaller hard-shell stand-in.

# v3 (2026-10-30): small realism pass

What changed (v2 renders moved to `renders_v2/`; rebuild order build -> settle -> render, unchanged):

- Kit bit rows. `scripts/fold_build.py` (regenerated `assets/bits_kit_fold/bits_kit_fold.xml` + meta; the five visual OBJs are
  byte-identical) adds visual-only geoms (group 2, no contact, mass 0) on every "bit column" content box: rows of 4.5 x 22 mm cylinders
  lying along the column (y) at 8.5 mm pitch, rows 26 mm apart (left columns 4 rows x 7 bits, centre 3 rows x 8), tops 2 mm above the
  box, material `bits_kit_fold_bit` (rgba 0.62 0.63 0.66, specular 0.6, shininess 0.5), one dark holder strip per row (`bits_kit_fold_holder`,
  rgba 0.08 0.08 0.10, 8 mm wide, 1 mm thick, 0.3 mm above the bit tops). 80 bits, 11 strips.
- Kit texture. `twin_build.py` now writes `twin/assets/bits_kit_fold/bits_kit_fold_colors_gain.png` itself (it was a hand-made file): the
  BGR gain (4.0, 2.9, 3.2) applies to the saturated (blue) texels only, neutral texels get x1.2, blended by the source HSV saturation
  (weight 0 at S 40, 1 at S 120); the bit-column faces (content boxes of the meta, mesh UVs, texture row = v * H) are painted the cloth
  colour BGR 30 30 32. Texel means: rim blue 142 61 31 (unchanged), left-panel interior fabric 89 78 78 -> ~30 33 31, bit-column texels
  above 150: 23 % -> 0; the texture's bright texels overall 11.4 % -> 5.6 % (the remaining ones are the product card and pocket flaps).
- Walls. `room/room.xml`: the four wall planes are 20 mm boxes with the inner face at the old plane position (front x = 0.8175, sides
  y = +-2.0, back x = -3.0), same rgba, visual-only. The floor stays a plane.
- Desk sheen. `twin_build.py` sets `table_top` specular 0.2, shininess 0.3 (was 0 / 0). Top-view desk mean (ext camera, f0 robot pose,
  segmentation-masked desk, luminance-trimmed 25..95 like the appearance fit): before 188.5 180.1 165.8, after 191.1 182.7 168.0
  (+2.6 +2.6 +2.2); specular 0.25 gave 191.8 183.3 168.6 (+3.3), over the 3-level budget, so 0.2.
- Beauty camera (`twin_render.py`, one line): azimuth 215 -> 300. The old camera sat at x = 1.79, y = 1.02, outside the front wall, and saw
  the desk only because the plane was one-sided; with box walls it would show the wall's back. The view is now from the right-back
  corner inside the room (camera about (-0.56, 1.54, 1.11)), robot on the right, front wall as backdrop; the wall's end at y = +1.0
  (the opening the depth shows) is visible at the left edge.

Settle (`renders/settle_log.txt`): same as v2. Start penetrations > 1 mm: only the robot's own link contacts and clipper_small x
clipper_case 1.2-1.7 mm (foam slot); no object-table or object-object penetration; all bodies drop their 2 mm lift and stay (xy 0.0-0.2 mm,
tilt <= 0.77 deg for clipper_small, <= 0.42 for the hex bits). The kit's visual geoms have no mass or contact, so its settle is unchanged
(dz -3.1 mm, tilt 0.00).

Still looks fake (renders/side_by_side_all.jpg, beauty_oblique.jpg):

- Kit: the bit rows read as neat metal rods on black, the real rows are blue-sleeved bits with silver tips; the interior still carries the
  scan's horizontal streak noise and the pocket flaps / product card are still saturated texels; the rim is a flat blue box, the real one
  is a rounded zipper.
- Desk: the sheen is a soft broad highlight; the real desk has fine warm grain and a sharper highlight; the robot's shadow band on the
  desk is a three-step penumbra.
- Room: bare walls, no furniture, chair or cables beyond the desk (the real top frame shows dark furniture and a chair past the far
  edge), and the wall opening at y = +1.0 ends in a hard vertical edge.
- Objects as in v2: flat-cap navy bottle, full-height ROAR label, translucent bits box, primitive basket contents, headphone stand-in.

# v4 (2026-10-30): tape roll, cup, product card

What changed (v3 renders moved to `renders_v3/`; rebuild order build -> settle -> render, unchanged):

- Two added objects, not in the real frame, both pulled by body name from the package `ep3/scene.xml` (the 2026-10-18 photoreal objects;
  the group-5 photo-projection geom of each body is dropped, the group-2 visual meshes and the cylinder collider stay, mass set on the
  collider). `tape_roll`: beige tape ring 75 x 30 mm lying flat with a hollow core (`nail_ep3_tape_roll_*`), 0.03 kg, at (0.16, -0.39),
  0.42 m from the base, between the kit's card end and the bottles. `paper_cup`: tapered cup 80 x 110 mm with `cup_print.png`
  (`nail_ep3_paper_cup_*`), 0.30 kg, at (0.20, 0.25), 0.32 m from the base, between the kit and the loose bits. Nearest colliders at the
  start (mj_geomDistance): tape 88 mm (navy bottle), 90 (ROAR), 97 (kit right slab); cup 70 mm (bit_5), 77 (kit left slab). The poses are
  `ADDED` in `twin_layout.py` and the same two entries in `layout.json` (`objects`, note "added, not in the real frame"; layout.json was
  edited in place because rerunning twin_layout.py would drop `rest_overrides`). The fallback downloads (assets_dl tape_roll / mug) were not
  needed. The cup's visual mesh bottom is 2.4 mm below its collider bottom, so its rim sinks 2.4 mm into the desk visually.
- Product card. In `twin_build.py` the right flap's content box "card pocket with product card" (115..220 x -101.5..98.5 mm aligned,
  i.e. nearly the whole flap: 105 x 200 of 114 x 240) is painted uniform white BGR 235 with a 2 mm border at 205 on every face of
  `bits_kit_right_visual.obj` inside the box footprint (18,128 faces, 67,824 + 4,688 texels). No z threshold: the scanned pocket surface
  dips to slab level between streaks, so a "top faces only" rule left dark lines.
- Texture mapping fix (affects the v3 bit-column paint too). The kit texture is a per-face atlas: every face owns a 2 x 2 texel block
  (median UV-triangle area 0.5 px), and MuJoCo samples OBJ textures at image row (1 - v) * H, not v * H (checked with a red/green row
  test: rows 453-623 render on the LEFT flap). The v3 code used cv2.fillPoly at row v * H, which hit only some blocks (dotted) and on the
  wrong panel, so the "bit columns painted cloth" of v3 never showed and the card patch stayed grey. `paint()` now fills each face's UV
  block at the flipped row; the bit columns are dark cloth under the bit rows for the first time.
- The flap's other bright texels (8,098 texels > 150 outside the card) are orange-brown scan texels at the rim's outer edge (desk colour
  bled into the scan; source HSV S ~160, V ~60), not saturated-white ones, so the "stray saturated texels -> cloth" rule matched none and
  they are left as they are.

Settle (`renders/settle_log.txt`, 1 s): tape_roll dxy 0.0 mm, dz -2.1 (the start lift), tilt 0.00; paper_cup dxy 0.0, dz -2.1, tilt 0.00.
No start penetration > 1 mm for either body; the only ones are the robot's own link contacts and clipper_small x clipper_case 1.2-1.7 mm
(foam slot, as before). Neither body touches anything over the second. All other bodies as v3.

Renders (`renders/`): `side_by_side_f{0,300,2400}.jpg`, `side_by_side_all.jpg`, `sim_top_*`, `sim_wrist_*`, `real_top_*`,
`beauty_oblique.jpg`. The tape roll sits left of the kit's card end in the top view, the cup right of the kit between it and the bits;
the card is a white rectangle at the kit's image-left end (the right flap is at y = -0.17, image left).

Still looks fake: the card is white over a bumpy scanned surface, so its far corner shades grey, and the blue zipper blob next to it is
scan colour; the rim's outer edge carries desk-coloured texels; everything listed under v3 otherwise.

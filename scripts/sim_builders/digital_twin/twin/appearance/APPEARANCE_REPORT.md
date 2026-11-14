# Appearance pass (2026-10-30)

Scope: desk colour, lights, wall and floor of the digital twin, tested on a copy (`scene_test.xml`), never on `scene_twin.xml`.
Reference frames: `outputs/rebot_teleop_2026-10-29-cut-v1` ep 0, top f0 (desk), wrist f0 (wall), wrist f2400 (floor).
Scripts (run from the LeRobot root): `scripts/appear_fit.py` (real patch statistics -> `fit_real.json`, patches drawn in
`fit_patches.jpg`), `scripts/appear_calibrate.py` (build `scene_test.xml` + `room_test.xml`, render, segment, fit, `--before`,
`--iters N`, `--no-update`, `--compose`, `--mult`). Colour statistics are trimmed (luminance 25th..95th percentile) on both sides;
sim regions come from a segmentation render of the geoms `room_desk_top` / `room_wall_front` / `room_floor`, so shadows and the
objects are excluded the same way as the hand-picked object-free real patches. Fit model: rendered = L * texture per channel,
L measured from the render, texture target = real / L; two iterations converge.

## Numbers

Mean RGB 0..255 of the trimmed regions. "before" = `scene_twin.xml` as it was (old lights, old texture, old room colours).

| region (real frame)   | real mean        | sim before       | err before | sim after        | err after |
|-----------------------|------------------|------------------|-----------:|------------------|----------:|
| desk, top f0          | 189.1 180.7 166.2 | 170.6 114.1 75.2 | 58.7 | 188.5 180.1 165.8 | 0.5 |
| wall, wrist f0        | 138.7 128.7 119.8 | 131.6 126.4 120.1 | 3.2 | 138.1 128.0 120.0 | 0.5 |
| floor, wrist f2400    | 141.8 120.8 105.8 | 159.1 119.0 92.8 | 10.7 | 141.4 120.2 105.4 | 0.5 |
| desk, wrist f0 (reference, not fitted) | 125.7 96.9 74.0 | 162.1 108.7 71.7 | 22.8 | 197.3 188.3 178.0 | 89.0 |

err = mean over channels of |real mean - sim mean|. The last row is the cross-camera problem, see "What remains off".

| quantity | value |
|----------|-------|
| desk texture source | `lerobot/src/lerobot/robots/rebot_b601_sim/scenes/table_top.png`, mean RGB 143.7 95.7 62.1 |
| desk affine (chosen, gain-only) | gain R 1.146 G 1.632 B 2.334, offset 0 0 0 -> `table_top_fit.png`, mean RGB 164.1 155.7 144.6, 0 % clipped |
| desk affine (alternative, equal gain) | gain 1.6 1.6 1.6, offset -67.0 +3.1 +46.7, 0.08 % clipped; same means, grey-green grain streaks; kept in `variant_affine/` |
| light factor L on the desk (render / texture) | 1.149 1.157 1.147 |
| real desk within-patch std | 12.6 12.1 12.6 ; sim after 9.1 7.4 9.0 (the gain clamp 1.6 of the affine variant was hit; the chosen variant does not match std) |
| wall rgba (all four walls) | 0.775 0.713 0.671 (L on the front wall 0.70) |
| baseboard rgba | 0.95 0.95 0.92, NOT fitted (strip under 3 px in the frames) |
| floor texture | Poly Haven "Laminate Floor 02" (CC0, Dario Barresi / Charlotte Baglioni), 1k diffuse jpg, `downloads/textures/laminate_floor_02/laminate_floor_02_diff_1k.jpg`, mean RGB 155.3 128.4 99.4; also fetched `laminate_floor_03` (unused, more yellow) |
| floor tone affine | gain 1.5 1.5 1.5 (clamp), offset -51.3 -38.0 -13.6 -> `floor_planks.png`; texrepeat 4 4 texuniform (0.25 m tile) |
| light factor L on the floor | 0.78 ; plain-colour fallback rgba 0.713 0.608 0.532 (in the room.xml comment) |
| shadow map | shadowsize 8192, shadowscale 0.6, shadowclip 1.0, `<statistic extent="2.0" center="0.34 0 -0.2"/>` (auto extent was 5.4 m from the room planes: 1.6 mm per texel at 4k; now ~0.3 mm) |

Lights (`lights.xml`, table frame: x toward the front wall, y image-right, z up; all directional):

| name | dir | diffuse | specular | shadow | role |
|------|-----|---------|----------|--------|------|
| key_a | 0.15 0.69 -0.71 | .17 | .12 | yes | key from -y, 45 deg elevation |
| key_b | 0.21 0.69 -0.69 | .17 | .12 | yes | key, +5 deg jitter (penumbra) |
| key_c | 0.09 0.72 -0.69 | .17 | .12 | yes | key, -5 deg jitter (penumbra) |
| fill_top | 0.55 0.1 -0.83 | .30 | .05 | no | ceiling light from above-behind the robot, lights the front wall |
| fill_side | 0 -0.8 -0.6 | .10 | 0 | no | bounce from +y, lifts the shadowed flanks |
| headlight (`lights_visual.xml`) | - | 0 | 0 | - | ambient .46 .46 .46, camera light off |

Shadow depth on the desk = (ambient + fills) / (all) ~ 0.7, three-step penumbra of about 10 deg total.

## Renders

`compare_f{0,300,2400}.jpg`: rows top / wrist, columns before | after | real. `sim_top_f*.jpg`, `sim_wrist_f*.jpg` = the after renders.
Note: the other agent re-rendered `twin/renders/sim_*.jpg` during this pass, so the "before" column shows their current object set
with the OLD appearance; the before numbers above were measured on `scene_twin.xml` at the time of the `--before` run.

## How to adopt in the twin build

In `twin_build.py` / `scene_twin.xml` (paths relative to `twin/`):

1. Replace the whole `<visual>...</visual>` block by `<include file="appearance/lights_visual.xml" />` (top level; it carries
   `<global offwidth/offheight>`, the headlight, the shadow quality and `<statistic>`).
2. Add `<include file="appearance/materials.xml" />` at top level, before `<asset>` (asset sections merge).
3. Point the desk texture at the fitted file: `<texture name="table_top_tex" type="2d"
   file="../../../../../scripts/sim_builders/digital_twin/twin/appearance/table_top_fit.png" />` (texture paths are relative to
   the compiler assetdir `lerobot/src/lerobot/robots/rebot_b601_sim/scenes`; the same convention is used inside materials.xml).
4. Inside `<worldbody>`: delete `detail_key`, `detail_fill`, `soft_shadow_0..3` and add `<include file="appearance/lights.xml" />`.
5. `room/room.xml` already references `room_floor_mat` and carries the wall / baseboard rgba, so step 2 is REQUIRED for
   `scene_twin.xml` to compile (until then it fails with an unknown material; the plain rgba fallback is in the room.xml comment).

`scene_test.xml` is exactly this applied to a copy of the current `scene_twin.xml` (it includes `room_test.xml`, a generated copy of
room.xml with the same edits, because room.xml was edited last).

## What remains off

- Grain: the fitted desk shows broad light/dark blotches; the real oak has fine, high-contrast, warm streaks. A per-channel affine
  on the existing photo texture cannot add that; a higher-resolution oak photo (or a Poly Haven oak veneer toned with the same fit)
  would.
- The two real cameras disagree: the wrist camera sees the desk at 126 97 74 (dark, orange), the top camera at 189 181 166 (light,
  nearly neutral). The texture is fitted to the top camera, so the sim wrist view of the desk is 89 units (mean abs) too light vs the
  real wrist frames, up from 23 before. A per-camera colour transform at render time is the only way to satisfy both.
- Soft shadows are a three-step penumbra (visible banding on the robot shadow); more jittered keys at lower power smooth it.
- Shadow direction / elevation (-y side, 45 deg) is from the brief, not measured on the frames.
- Beyond the far desk edge the top view shows floor planks; the real frame shows dark furniture and a chair there.
- Baseboard colour unfitted; desk specular left at 0 (real has a slight sheen); plank scale (0.25 m tile, ~6 planks) eyeballed.

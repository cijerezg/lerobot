# Sim scene builders

Scripts, small inputs and reports that produced the `rebot_b601_sim` scenes
(`src/lerobot/robots/rebot_b601_sim/scenes/`). Moved here from the workspace `migration/` folder on 2026-11-13.
The package scenes do not need these folders at runtime. Commands run from the workspace root (the folder that
holds `lerobot/` and `config_rl_sim.yaml`). Heavy raw sources live outside git in `<workspace>/assets/sim_sources/`.

Every builder writes into the package scenes or next to itself. Read its report before running one;
`make_sim_config.py` and the scene builders overwrite files.

| folder | builds | entry | inputs | heavy sources |
|---|---|---|---|---|
| `real2sim/` | nail-clipper layouts `ep0`-`ep3`, `clippers_only`, `rebot_b601_dm_photo.xml`, `config_rl_sim.yaml` | `scripts/build_layout.sh`, `scripts/run_layouts.sh`, `scripts/port_scenes.py`, `scripts/make_sim_config.py`; `HANDOFF.md` has the order | `assets/` (case scans), 10-16 recordings in `outputs/`; `work/` (frames, intermediates) was deleted before the move | none |
| `easy_envs/` | `lift_bottle`, `bottle_basket`, `close_lid` | `bottle_basket/build_basket.py`, `close_lid/build_close_lid.py`, `*/verify_*.py` | `basket_reference/`, package scenes | none |
| `lid_solvability/` | close_lid collider fix and the T071 demo poses (`demo_pose_summary.json`) | `solve_lid.py`, `demo_pose.py`, `loop_collider.py`; `README.md` | `frames/t071_ep*_top_f0.png`, `*.orig` | none |
| `variants/` | placement variants `<layout>_v1..v6` | `generate_variants.py` | `../lid_solvability/demo_pose_summary.json`, package scenes | none |
| `digital_twin/` | twin desk (`twin/scene_twin.xml`), room, twin assets, the 10-30 twin eval scenes, judge study | `scripts/twin_build.py`, `envs/build_twin_envs.py`; `HANDOFF_2026-10-30.md` | `twin/`, `room/` | `assets/sim_sources/digital_twin/` (`scans/` raw, `downloads/`, `assets/`, `assets_dl/`, `textures/`, the `work/` files the scripts read) |
| `twin_eval_v2/` | current eval set `twin_l<L>_<config>_p<1-4>` (64 scenes) | `build_twin_envs_v2.py` (imports `../digital_twin/envs/build_twin_envs.py`); `ENVS_REPORT_v2.md` | `../digital_twin/twin/` | `assets/sim_sources/twin_eval_v2/` (v1 scene backup, removed v2 scenes, pre-bits builder) |
| `photoreal_clippers/` | clipper visuals for `nail_ep*` | `prepare_references.py`, `build.py`, `validate.py`, `render.py` | `references/` (their source video `sim/nail.mp4` is no longer in the workspace), package scenes | none |
| `photoreal_objects/` | clutter visuals (bottles, cup, tape, cases) | `textures.py`, `build.py`, `validate.py`, `render.py` | `references/`, `../photoreal_clippers/nail_ep*.xml`; `textures.py` also reads real2sim `work/` frames that no longer exist (its outputs are in `assets/`) | none |
| `photoreal_headphone_hinge/` | connected headphone case | `build.py`, `validate.py`, `render.py` | `../photoreal_objects/nail_ep*.xml` | none |
| `photoreal_integration/` | promotion of the three photoreal passes into `scenes/assets/photoreal/` | `promote.py`, `validate.py` | the three folders above; `before/` = pre-promotion backup | none |

The photoreal passes chain: each reads the previous pass's `nail_ep*.xml` and `config_nail_ep*.yaml`, which carry
absolute workspace paths.

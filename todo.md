# TODO

- Update `src/lerobot/scripts/view_validation.py` to discover generic offline inference outputs under `offline_inference/unnormalized_eval` and `offline_inference/normalized_eval` in addition to the legacy pi05 `unnormalized` / `normalized` paths.
- Update `src/lerobot/scripts/view_validation.py` to discover generic action-drift Jacobian videos under `action_drift_jacobian/<dataset>/epXXXX_t*/Lxx/`, including crop panels like `causal_overlay_img_<camera>_cropNN_summary.mp4`.

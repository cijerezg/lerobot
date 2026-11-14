# pan2 (2026-11-11)

Question: can the PAN 2 video world model (`IFM/PAN2`, diffusers `ModularPipeline`) continue a real top-camera
frame of the rebot table into a plausible 10 s manipulation clip, judged blind against the real recording?

## Files

- `run_pan2.py`: generates one clip per record of a prompts JSON (`name`, `prompt`, optional `negative_prompt`,
  `image`, `seconds`, `height`, `width`); 24 fps, finished clips are skipped. Needs `diffusers` with PAN 2 and a
  CUDA GPU; it was run on the Spark from its own venv, not from this repo's environment.
- `build_prompts.py`: writes `prompts_robot.json` (lid, tape) and `prompts_cat.json` (catlid, cattape, wlid) next
  to itself from `examples.json` (the rewriter's caption examples and the fixed 10 s negative).
- `inputs/<name>_frame0.png`: the first frame of each real clip, the image the records condition on.
- `blind_key.json`: which side of each `<name>_blind.mp4` comparison video was real and which was PAN 2.

## Run

From this folder on the machine that has PAN 2 (record image paths are relative, `inputs/...`):

    python build_prompts.py
    python run_pan2.py prompts_robot.json out/ --steps 50 --seed 0

## Findings

No written verdict was kept. The generated, real and blind comparison videos were not kept in the repo.

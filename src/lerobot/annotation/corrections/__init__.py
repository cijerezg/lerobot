"""Corrected action chunks from hindsight clips (rubric: ``annotation/rubrics/correction_rubric.md``).

A pass is one results folder ``<work>`` (``outputs/_annotation/corrections_v1``). Everything specific to it is
data in ``<work>/pass.json``:

  {"sources": [{"name": ..., "root": ..., "kind": "inference" | "rebot", "top": ...}, ...],
   "contact": {subtask: [contact, precision], ...},   # inference runs carry no contact labels
   "top": {name: camera}, "cameras": {"<source>/ep<NNN>": camera}}   # optional top cameras, see fit_top

Commands, from the workspace root (M = lerobot.annotation.corrections):

  uv run python -m $M.sample <work>       frames to label              -> <work>/frames.parquet, stalls.json
  uv run python -m $M.fit_top <work>      optional: the top camera of each episode, for the tool-tip mark on the
                                          top view. Labels never use it
  uv run python -m $M.dump <work>         what the labeller reads      -> <work>/<source>/ep<NNN>/f<NNNNNN>/, jobs.txt
  (a labeller writes f<NNNNNN>/waypoints.json per the rubric)
  uv run python -m $M.build <work>        waypoints -> corrected chunk -> correction.npy per frame, corrections.parquet
  uv run python -m $M.export <work> <out> training root (new)          -> one 30-frame episode per ok correction: the frame's
                                          observation, the corrected chunk as its actions
  uv run python -m $M.inspector <work> .. Action Inspector page of chosen or flagged corrections

Frames are episode-local everywhere. ``arm`` holds the twin's FK / IK and the chunk check, ``source`` one dataset
root with its per-frame labels.
"""

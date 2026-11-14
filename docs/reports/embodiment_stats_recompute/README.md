# Embodiment stats recompute 2026-10-27

> Saved 2026-11-13 from the scratch folder `stats_recompute_2026-10-27`. Kept here: this page and the
> two comparison tables. `compute_filtered.py` is now `lerobot/scripts/compute_filtered_stats.py`; the
> run scripts, logs, `compare.py` and the all-roots `.pt` were not copied. On 2026-11-13
> `config_rl.yaml` still points at the 20261010 file; the 20261027 file exists under `outputs/stats/`.

Question: config_rl.yaml still points at `outputs/stats/action_stats_anchor_diverse-rebot-hand-20261010.pt`
(7 whole ReBot roots, 708k chunks). Recompute over everything sampled today and compare.

Files
- `run.sh` / `run-allroots.log` / `action_stats_anchor_diverse-rebot-hand-20261027-allroots.pt`: first attempt, 15 whole roots
  (947k chunks). Kept only as a reference; it includes the `excluded` episodes and the usefulness-cut frames.
- `compute_filtered.py` + `run_filtered.sh` / `run-filtered.log`: the real one -> `outputs/stats/action_stats_anchor_diverse-rebot-hand-20261027.pt`.
  Reads config_rl.yaml sources, drops weight-0 (`excluded`) episodes whole, drops chunk starts whose window overlaps
  a `rebot_exclusions_manifest` range (FrameExclusions.filter_starts, same rule as training), plus the two unannotated
  10-27 cut roots. 940,311 sampled frames, 876,261 chunks, 890,906 states. Diverse half identical to 10-10.
  The artifact carries a `rebot_selection` block (per-root counts).
- `compare.py`, `compare_old_vs_filtered.txt`, `compare_allroots_vs_filtered.txt`.

Result: ReBot action q01/q99 bands shrink 5-11 % (new data moves less per chunk), state bands within 2 % except pan +8 %
(new data reaches pan -71/+74 vs -68/+66). The cuts alone move bands < 1.5 deg (gripper < 4). Hand-block slots on the
diverse rows change too because ReBot sets the pooled hand scales (irrelevant while hand_block: false).

Caveat: with policy.pretrained_path set to a checkpoint's pretrained_model (which has saved processors), the trainer
uses the SAVED processors and ignores embodiment_stats_path (rl_molmoact2_trainer._saved_processor_path). A new stats
file only takes effect on a run with pretrained_path null.

"""Action Inspector in sim: the action_trace probe's flow samples played on the MuJoCo reBot.

Reads ``action_trace/action_trace_records.npz`` from the same validation step (so
``enable_action_trace`` must be on; this probe runs right after it) and writes
``action_trace_sim.rrd`` with ``lerobot.probes.action_trace_sim``. The viewer cannot show a
rerun log; open it with

    .venv/bin/rerun <step dir>/action_trace_sim/action_trace_sim.rrd

``probe_parameters.trace_sim_args`` passes extra flags to the sim CLI, e.g.
``--time-scale 2`` (half speed) or ``--execute`` (MuJoCo physics: achieved trails,
end-effector lag, table contact). The sim runs in its own process, so mujoco and rerun are
never imported into the training process.

action_trace_sim.rrd — every anchor in order on one ``time`` timeline, jump with ``anchor``.
"""

import os
import shlex
import subprocess
import sys

from lerobot.probes.manifest import write_index


def run(adapter, dataset, cfg, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    records = os.path.join(
        os.path.dirname(os.path.normpath(output_dir)), "action_trace", "action_trace_records.npz"
    )
    if not os.path.exists(records):
        raise FileNotFoundError(f"{records} missing: action_trace_sim plays the action_trace probe's records")
    rrd = os.path.join(output_dir, "action_trace_sim.rrd")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot.probes.action_trace_sim",
            "--records",
            records,
            "--out",
            rrd,
            *shlex.split(cfg.probe_parameters.trace_sim_args),
        ],
        check=True,
    )
    write_index(
        output_dir,
        sys.modules[__name__],
        claim="What does the arm do with each flow sample of the Action Inspector's chunks?",
        group="Actions",
        metrics=[],
        panels=[],
        see_also=["action_trace"],
        extra={"rrd": "action_trace_sim.rrd", "trace_sim_args": cfg.probe_parameters.trace_sim_args},
    )

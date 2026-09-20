"""Numeric check of probes.embodiment_swap with no model.

The adapter is faked so the answer is known: every trained foreign name (YAM included)
shifts the chunk by one shared posture vector (so the pairwise cosine among them is +1),
the ReBot name shifts it by its own larger vector, unseen names do nothing, and removing
the embodiment clause on a diverse frame shifts it by a small vector. On the second axis,
the other control-mode clause adds a large vector, no control-mode clause a small one.
Flow noise is a whisper, so the separations must come out far above 1 for the read
prompts and near 1 for the unseen names.
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.probes import embodiment_swap as es

CHUNK, DIM = 30, 8
TRAINED = {"Franka Panda", "UR5", "ARX5", "UR7e", "YAM"}
MOLMOACT_LAYOUT, UR5_LAYOUT = 7, 4
rng = np.random.default_rng(0)
BASE = torch.tensor(rng.normal(size=(CHUNK, DIM)), dtype=torch.float32)
FOREIGN = torch.tensor(rng.normal(size=(1, DIM)), dtype=torch.float32)
REBOT_VEC = 2.0 * torch.tensor(rng.normal(size=(1, DIM)), dtype=torch.float32)
NONE_VEC = 0.5 * torch.tensor(rng.normal(size=(1, DIM)), dtype=torch.float32)
MODE_VEC = 3.0 * torch.tensor(rng.normal(size=(1, DIM)), dtype=torch.float32)
NO_MODE_VEC = 0.25 * torch.tensor(rng.normal(size=(1, DIM)), dtype=torch.float32)


class FakeAdapter:
    def _set_probe_cuda_graph_enabled(self, enabled):
        pass

    def _restore_probe_cuda_graph_enabled(self):
        pass

    def flow_noise_like(self, n, seed):
        return torch.randn(n, CHUNK, DIM, generator=torch.Generator().manual_seed(seed))

    def predict_action_chunk_batch(
        self, obs, task, subtasks, *, metadatas, noise, embodiments, control_modes, extra_complementary
    ):
        # The frame's training-regime space: ReBot and every diverse layout but MolmoAct are joint.
        layout = None if extra_complementary is None else int(extra_complementary["action_layout_id"].reshape(-1)[0])
        own_mode = "end_effector" if layout == MOLMOACT_LAYOUT else "joint"
        chunks = []
        for i, (name, mode) in enumerate(zip(embodiments, control_modes, strict=True)):
            shift = torch.zeros(1, DIM)
            if name == "":
                shift = NONE_VEC if extra_complementary is not None else torch.zeros(1, DIM)
            elif name == "Rebot B601":
                shift = REBOT_VEC
            elif name in TRAINED:
                # One shared direction, a different magnitude per name: pairwise cosine +1,
                # and a frame's own label differs from every other trained name.
                shift = FOREIGN * (1 + sorted(TRAINED).index(name))
            if mode == "":
                shift = shift + NO_MODE_VEC
            elif mode != own_mode:
                shift = shift + MODE_VEC
            chunks.append(BASE + shift + 0.002 * noise[i])
        return None, torch.stack(chunks)


class FakeDataset:
    root, repo_id = "/does/not/exist", "fake"


class FakeBuffer:
    """Three holdout episodes: one MolmoAct (end-effector home), two joint-space."""

    rows = [
        {"episode_id": f"ep{e}", "source": ("molmoact", "droid", "robochallenge")[e],
         "embodiment": ("Franka", "Franka", "UR5")[e], "anchor_s": float(k), "subtask": "grasp the cup"}
        for e in range(3) for k in range(12)
    ]


def fake_open_holdout(cfg):
    return FakeBuffer(), set(TRAINED)


def fake_rebot_samples(dataset, cfg, n_frames):
    return [{"domain": es.REBOT, "source": es.REBOT, "own": es.REBOT_NAME, "episode": f"rebot/{e}",
             "frame": k, "index": e * 100 + k} for e in range(2) for k in range(n_frames)]


def fake_probe_frame_inputs(dataset, cfg, index, chunk_size):
    return {"obs": {}, "task": "put the sock in the basket", "subtask": "grasp the sock"}


def fake_diverse_inputs(buffer, cfg, sample):
    layout = MOLMOACT_LAYOUT if sample["source"] == "molmoact" else UR5_LAYOUT
    return {"obs": {}, "task": "hang the cup", "subtask": "grasp the cup",
            "extra": {"action_layout_id": torch.tensor([layout])}}


CFG = SimpleNamespace(
    policy=SimpleNamespace(action_mode="continuous", chunk_size=CHUNK, embodiment=None),
    probe_parameters=SimpleNamespace(n_seeds=3, n_frames_per_episode=4, max_episodes=None, random_seed=0,
                                     embodiment_swap_n_frames=None, embodiment_swap_n_seeds=None),
    diverse=SimpleNamespace(root="/does/not/exist", render_automatic_quality=False,
                            rebot_layout="rebot_b601_joint7_commanded"),
)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("embodiment_swap"))
    patches = [
        (es, "_open_holdout", fake_open_holdout),
        (es, "_rebot_samples", fake_rebot_samples),
        (es, "probe_frame_inputs", fake_probe_frame_inputs),
        (es, "_diverse_inputs", fake_diverse_inputs),
    ]
    originals = [(module, name, getattr(module, name)) for module, name, _ in patches]
    for module, name, fake in patches:
        setattr(module, name, fake)
    try:
        summary = es.run(FakeAdapter(), FakeDataset(), CFG, out)
    finally:
        for module, name, original in originals:
            setattr(module, name, original)
    with open(os.path.join(out, "embodiment_swap.json")) as f:
        rows = json.load(f)["per_frame"]
    return out, summary, rows


def test_artifacts_written(result):
    out, summary, rows = result
    assert os.path.isfile(os.path.join(out, "embodiment_swap.png"))
    assert os.path.isfile(os.path.join(out, "embodiment_swap.json"))
    assert os.path.isfile(os.path.join(out, "index.json"))
    assert summary["n_rebot_frames"] == 8 and summary["n_diverse_frames"] == 12


def test_name_axis(result):
    _, summary, rows = result
    assert summary["foreign_trained_on_rebot_sep_median"] > 20
    assert summary["rebot_label_on_rebot_sep_median"] > 20
    assert summary["foreign_unseen_on_rebot_sep_median"] < 1.5
    assert summary["none_on_diverse_sep_median"] > 5
    assert summary["rebot_label_on_diverse_sep_median"] > 5 and summary["foreign_trained_on_diverse_sep_median"] > 5
    # On diverse frames the fake reads an unseen name as "no clause", which is a move away
    # from the own-label home by construction; the near-1 unseen check is ReBot-side only.
    # On diverse frames the fake's own-label scale sits mid-ramp, so the other names' shifts
    # have mixed signs there; the +1 pairwise check is meaningful on ReBot frames only.
    assert summary["rebot_foreign_pair_cosine_median"] > 0.99
    assert abs(summary["null_pair_cosine_median"]) < 0.6
    # The fake is linear in its noise and reuses the seeds per frame, so its reseed null is
    # degenerate (+1); only the swap's own alignment is asserted.
    assert summary["rebot_shared_cosine_median"] > 0.99 and summary["diverse_shared_cosine_median"] > 0.99
    # A frame's own label is its home and gets no separation entry.
    assert all("franka_panda_sep" not in r for r in rows if r["domain"] == es.DIVERSE and r["own"] == "Franka Panda")
    assert all("none_sep" not in r for r in rows if r["domain"] == es.REBOT)
    assert set(summary["diverse_by_robot"]) == {"Franka Panda", "UR5"}


def test_yam_is_a_name(result):
    _, summary, rows = result
    assert es.LABELS["yam"] == "YAM" and "yam" in es.CONDITIONS
    assert "YAM" in summary["trained_labels"] and "YAM" not in summary["unseen_labels"]
    assert summary["per_label"][es.REBOT]["yam"]["sep_median"] > 20
    assert all("yam_sep" in r and "yam@end_effector_sep" in r for r in rows if r["domain"] == es.REBOT)


def test_control_mode_axis(result):
    _, summary, rows = result
    assert summary["rebot_mode"] == "joint"
    assert summary["mode_swapped_on_rebot_sep_median"] > 20
    assert summary["mode_none_on_rebot_sep_median"] > 5
    assert summary["mode_swapped_on_diverse_sep_median"] > 20
    assert summary["mode_none_on_diverse_sep_median"] > 5
    assert summary["rebot_label_joint_on_diverse_sep_median"] > 5
    assert set(summary["per_cell"][es.REBOT]) == {f"{n}@{m}" for n in es.CONDITIONS for m in es.MODES}
    # ReBot home = no name + joint: that cell has no entry, every other cell of the grid does.
    rebot = [r for r in rows if r["domain"] == es.REBOT]
    assert all(r["home"] == es.NONE and r["home_mode"] == "joint" for r in rebot)
    assert all("none@joint_sep" not in r and "none@none_sep" in r and "none@end_effector_sep" in r for r in rebot)
    assert all(r["franka_panda_sep"] == r["franka_panda@joint_sep"] for r in rebot)
    assert all(r["mode_swapped_sep"] == r["none@end_effector_sep"] for r in rebot)
    assert all(r["mode_none_sep"] == r["none@none_sep"] for r in rebot)


def test_diverse_homes_split_by_mode(result):
    _, summary, rows = result
    by_mode = summary["diverse_by_mode"]
    assert set(by_mode) == {"joint", "end_effector"}
    assert by_mode["joint"]["n"] == 8 and by_mode["end_effector"]["n"] == 4
    assert by_mode["end_effector"]["swapped_to"] == "joint" and by_mode["joint"]["swapped_to"] == "end_effector"
    assert by_mode["end_effector"]["mode_swapped_sep"]["median"] > 20
    molmoact = [r for r in rows if r["source"] == "molmoact"]
    assert all(r["home"] == "franka_panda" and r["home_mode"] == "end_effector" for r in molmoact)
    # Franka Panda@joint on a MolmoAct frame is the DROID/FMB prompt: a swap, not the home.
    assert all("franka_panda@end_effector_sep" not in r and "franka_panda@joint_sep" in r for r in molmoact)
    assert all(r["mode_swapped_sep"] == r["franka_panda@joint_sep"] for r in molmoact)
    assert all(r["rebot_b601_sep"] == r["rebot_b601@end_effector_sep"] for r in molmoact)
    joint = [r for r in rows if r["domain"] == es.DIVERSE and r["source"] != "molmoact"]
    assert all(r["rebot_b601_sep"] == r["rebot_b601@joint_sep"] for r in joint)

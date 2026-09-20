"""Numeric check of probes.embodiment_swap with no model.

The adapter is faked so the answer is known: every trained foreign name (YAM included)
shifts the chunk by one shared posture vector (so the pairwise cosine among them is +1),
the ReBot name shifts it by its own larger vector, unseen names do nothing, and removing
the embodiment clause on a diverse frame shifts it by a small vector. On the second axis,
the other control-mode clause adds a large vector, no control-mode clause a small one.
Flow noise is a whisper, so the separations must come out far above 1 for the read
prompts and zero for the ignored names under paired noise.
"""

import json
import os
from types import SimpleNamespace

from datasets import Dataset

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
        return torch.randn(1, CHUNK, DIM, generator=torch.Generator().manual_seed(seed)).expand(n, -1, -1).clone()

    def predict_action_chunk_batch(
        self, obs, task, subtasks, *, metadatas, noise, embodiments, control_modes, extra_complementary
    ):
        # The frame's training-regime space: ReBot and every diverse layout but MolmoAct are joint.
        layout = None if extra_complementary is None else int(extra_complementary["action_layout_id"].reshape(-1)[0])
        own_mode = "end_effector" if layout == MOLMOACT_LAYOUT else "joint"
        chunks = []
        for i, (name, mode) in enumerate(zip(embodiments, control_modes, strict=True)):
            shift = REBOT_VEC.clone() if layout is None else torch.zeros(1, DIM)
            if name == "":
                shift = NONE_VEC
            elif name == "Rebot B601":
                shift = REBOT_VEC
            elif name in TRAINED:
                # One shared direction, a different magnitude per name: pairwise cosine +1,
                # and a frame's own label differs from every other trained name.
                shift = FOREIGN * (1 + sorted(TRAINED).index(name))
                if layout is None:
                    shift = shift + REBOT_VEC
            if mode == "":
                shift = shift + NO_MODE_VEC
            elif mode != own_mode:
                shift = shift + MODE_VEC
            chunks.append(BASE + shift + 0.002 * noise[i])
        return None, torch.stack(chunks)


class FakeDataset:
    root, repo_id = "/does/not/exist", "fake"
    meta = SimpleNamespace(robot_type="rebot_b601_follower")
    hf_dataset = Dataset.from_dict({"frame_index": [0, 1]})

    def __len__(self):
        return len(self.hf_dataset)


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
    assert summary["none_on_rebot_sep_median"] > 20
    assert summary["rebot_home"] == "rebot_b601"
    assert "Rebot B601" in summary["trained_labels"]
    assert summary["foreign_unseen_on_rebot_sep_median"] < 1.5
    assert summary["none_on_diverse_sep_median"] > 5
    assert summary["rebot_label_on_diverse_sep_median"] > 5 and summary["foreign_trained_on_diverse_sep_median"] > 5
    # On diverse frames the fake reads an unseen name as "no clause", which is a move away
    # from the own-label home by construction; the zero-effect check is ReBot-side only.
    # On diverse frames the fake's own-label scale sits mid-ramp, so the other names' shifts
    # have mixed signs there; the +1 pairwise check is meaningful on ReBot frames only.
    assert summary["rebot_foreign_pair_cosine_median"] > 0.99
    assert abs(summary["null_pair_cosine_median"]) < 0.6
    # The fake is linear in its noise and reuses the seeds per frame, so its reseed null is
    # degenerate (+1); only the swap's own alignment is asserted.
    assert summary["rebot_shared_cosine_median"] > 0.99 and summary["diverse_shared_cosine_median"] > 0.99
    # A frame's own label is its home and gets no separation entry.
    assert all("franka_panda_sep" not in r for r in rows if r["domain"] == es.DIVERSE and r["own"] == "Franka Panda")
    assert all("rebot_b601_sep" not in r and "none_sep" in r for r in rows if r["domain"] == es.REBOT)
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
    # ReBot home is its training name + joint; removing the name is an intervention.
    rebot = [r for r in rows if r["domain"] == es.REBOT]
    assert all(r["home"] == "rebot_b601" and r["home_mode"] == "joint" for r in rebot)
    assert all("rebot_b601@joint_sep" not in r and "none@joint_sep" in r for r in rebot)
    assert all(r["franka_panda_sep"] == r["franka_panda@joint_sep"] for r in rebot)
    assert all(r["mode_swapped_sep"] == r["rebot_b601@end_effector_sep"] for r in rebot)
    assert all(r["mode_none_sep"] == r["rebot_b601@none_sep"] for r in rebot)


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


@pytest.mark.parametrize(
    "columns,override,expected",
    [
        ({}, None, "Rebot B601"),
        ({}, "UR5", "UR5"),
        ({"embodiment_index": [3, 3]}, "UR5", "ARX5"),
        ({"source.robot_type": ["Franka", "Franka"]}, "UR5", "Franka Panda"),
    ],
)
def test_both_probes_match_training_prompt_identity(tmp_path, monkeypatch, columns, override, expected):
    from lerobot.datasets.embodiment import embodiment_index
    from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2PackInputsProcessorStep
    from lerobot.probes import conditions_matrix as cm
    from lerobot.rl.offline_dataset_utils import _embodiment_indices_for_dataset

    dataset = FakeDataset()
    dataset.root = str(tmp_path)
    dataset.hf_dataset = Dataset.from_dict({"frame_index": [0, 1], **columns}).with_format("numpy")
    cfg = SimpleNamespace(
        policy=SimpleNamespace(embodiment="SO-101"),  # not the offline training identity
        dataset=SimpleNamespace(sources=[SimpleNamespace(root=str(tmp_path), embodiment=override)]),
        probe_parameters=SimpleNamespace(conditions_episodes_per_cell=2, conditions_frames_per_episode_cell=2),
    )
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "subtask_windows.json").write_text(json.dumps({
        "episodes": {"0": [{"from_index": 0, "to_index": 2, "subtask": "grasp the cup"}]},
    }))
    monkeypatch.setattr(cm, "build_episode_index", lambda _: {0: [0, 1]})
    monkeypatch.setattr(cm, "probe_image_stride", lambda _: 1)
    monkeypatch.setattr(cm, "probe_frame_inputs", lambda *a, **kw: {
        "obs": {}, "task": "put the cup away", "subtask": "grasp the cup",
    })
    sample = cm._rebot_samples(dataset, cfg, "rebot", False, np.random.RandomState(0))[0]
    inputs = cm._rebot_inputs(dataset, cfg, sample, CHUNK)
    training, _ = _embodiment_indices_for_dataset(dataset, len(dataset), override)
    pack = object.__new__(MolmoAct2PackInputsProcessorStep)
    pack.embodiment_names = list(es.EMBODIMENT_NAMES)
    assert int(training[0]) == sample["embodiment_index"] == embodiment_index(expected)
    assert pack._extract_embodiment_texts(inputs["extra"], 1) == [expected]
    assert es._rebot_home(dataset, cfg) == es._slug(expected)


def test_ignored_name_has_zero_paired_separation():
    row, _ = es._measure_frame(
        FakeAdapter(), {"obs": {}, "task": "task", "subtask": "step", "extra": None},
        "rebot_b601", "joint", {es._slug(n) for n in TRAINED} | {"rebot_b601"}, 3,
    )
    assert row["foreign_unseen_sep"] == 0.0
    assert row["none_sep"] > 0.0

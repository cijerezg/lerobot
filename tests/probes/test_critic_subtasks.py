"""Behaviour checks for counterfactual conditioning, returns and grouped gradients."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from lerobot.probes import critic as c
from lerobot.probes import critic_subtasks as cs
from lerobot.probes import critic_sensitivity as sensitivity


def test_recorded_returns_match_duration_and_charge_terminal_mistakes():
    boundary = np.zeros(100, dtype=bool)
    boundary[[59, 99]] = True
    onset = np.zeros(100, dtype=bool)
    got = cs.reference_returns(boundary, onset, 30, 0.97, 12, 5, -2)
    ends = np.where(np.arange(100) <= 59, 59, 99)
    ideal = c._ideal_duration_value(ends - np.arange(100), 30, 0.97, 12, -2)
    np.testing.assert_allclose(got, ideal)
    onset[59] = True
    penalized = cs.reference_returns(boundary, onset, 30, 0.97, 12, 5, -2)
    assert penalized[30] == pytest.approx(-5 / 12)
    assert penalized[0] == pytest.approx(-1 / 12 - 0.97 * 5 / 12)
    assert penalized[60] == pytest.approx(got[60])  # no leakage into next episode
    assert cs.reference_returns(boundary, onset, 30, 0.97, 12, 500, -2)[30] == -2


def _row(i=0, **changes):
    return {
        "domain": "rebot", "source": "rebot", "episode": 0, "episode_idx": 0, "frame_idx": i,
        "global_idx": i, "task": "sort shirts", "subtask": "grasp the blue shirt",
        "metadata": {"quality": 4, "mistake": False}, "segment_end": 300,
        "value": -0.6 + i / 3000, "reference": -0.7 + i / 3000,
        "seconds": i / 30, "progress_error": None, "swaps": {}, **changes,
    }


def test_progress_uses_only_comparable_pairs():
    a, b = _row(), _row(30)
    assert cs.progress_error(a, b, 30, -2) == pytest.approx(0)
    for change in (
        {"subtask": "release the blue shirt"}, {"metadata": {"quality": 3}},
        {"segment_end": 400}, {"episode": 1}, {"global_idx": 60}, {"value": None},
    ):
        assert cs.progress_error(a, {**b, **change}, 30, -2) is None
    assert cs.progress_error({**a, "reference": -2}, {**b, "reference": -2}, 30, -2) is None


def test_native_reference_matches_sampler_terminal_window():
    row = {"native_rate_hz": 15, "critic_end_timestep_exclusive": 31, "mistake_onset_timesteps": [30]}
    g = lambda frame: cs.diverse_reference(row, frame, 1, 0.97, 12, 5, -2, 0)
    assert g(16) == pytest.approx(-5 / 12)  # end == start + one native chunk
    assert g(1) == pytest.approx(-1 / 12 - 0.97 * 5 / 12)


def test_group_sampling_is_deterministic_and_keeps_exact_labels():
    rows = [_row(i * 30, subtask=text) for text in ("grasp blue shirt", "grasp beige shirt") for i in range(12)]
    picked = cs.sample_text_groups(rows, 16, 8, 42)
    assert picked == cs.sample_text_groups(rows, 16, 8, 42)
    assert len(picked) == 16
    assert len({r["subtask"] for r in picked[:8]}) == 1
    assert len({r["subtask"] for r in picked[8:]}) == 1
    assert cs.sample_text_groups(rows, 0, 8, 42) == []


def test_fit_report_keeps_unseen_and_missing_progress_separate(tmp_path):
    rows = [_row(progress_error=0.02), _row(30, subtask="grasp beige shirt", progress_error=None)]
    summary = cs.fit_report(rows, {"grasp the blue shirt": 10}, tmp_path)
    assert summary["critic_rare_value_error_texts"] == 1
    assert summary["critic_rare_progress_error"] is None
    assert summary["critic_common_progress_error"] == pytest.approx(0.02)
    data = json.loads((tmp_path / "critic_fit_vs_count.json").read_text())
    assert next(r for r in data["per_text"] if r["text"] == "grasp beige shirt")["train_episodes"] == 0
    assert (tmp_path / "critic_fit_vs_count.png").exists()


@pytest.mark.parametrize("fixed_control", [None, "close box"])
def test_trace_swaps_keep_frame_inputs_and_respect_folded_release(monkeypatch, tmp_path, fixed_control):
    n = 120
    boundary = np.zeros(n, dtype=bool)
    boundary[[29, 89, 119]] = True  # move+release share terminal 89
    texts = ["grasp blue"] * 30 + ["move blue"] * 30 + ["release blue"] * 30 + ["grasp blue"] * 30
    labels = {i: {"quality": 4, "mistake": False} for i in range(n)}
    targets = {"mode": "subtask", "labels": labels, "episode_end": np.arange(n) == n-1,
               "terminals": boundary, "frames_to_end": np.searchsorted(np.flatnonzero(boundary), np.arange(n)),
               "mistake_onset": np.zeros(n, dtype=bool)}
    targets["frames_to_end"] = np.flatnonzero(boundary)[targets["frames_to_end"]] - np.arange(n)
    monkeypatch.setattr(c, "_training_targets", lambda *_: targets)
    monkeypatch.setattr(c, "build_episode_index", lambda _: {0: list(range(n))})
    monkeypatch.setattr(c, "get_subtask_idx", lambda _, i: i)
    monkeypatch.setattr(c, "get_subtask_str", lambda _, i: texts[i])
    observations = {i: {"frame": i} for i in range(0, n, 30)}
    monkeypatch.setattr(c, "probe_frame_inputs", lambda d, cfg, i, ch, metadata=None: {
        "obs": observations[i], "subtask": texts[i], "task": "sort", "metadata": metadata,
        "frame_idx": i, "episode_idx": 0,
    })
    monkeypatch.setattr(c, "diverse_fit_records", lambda *_: [])
    calls = []
    class Adapter:
        chunk_size = 30
        def predict_value(self, obs, task, subtask, metadata):
            calls.append((obs, task, subtask, metadata))
            if obs["frame"] == 60 and subtask == "release blue":
                raise RuntimeError("synthetic failure")
            return -0.5
    count_path = tmp_path / "counts.json"
    count_path.write_text(json.dumps({"grasp blue": 10, "move blue": 10, "release blue": 10, "open drawer": 50, "close box": 5}))
    cfg = NS(
        policy=NS(discount=0.97, reward_normalization_constant=12, critic_mistake_penalty=5,
                  value_support_min=-2, value_support_max=0, image_stride=3),
        probe_parameters=NS(critic_subtask_swap=True, critic_text_counts_path=str(count_path)),
        env=NS(fps=30),
    )
    observed = []
    def observe(row, obs):
        assert obs is observations[row['global_idx']]
        row['snapshot_id'] = row['global_idx']
        observed.append(row['global_idx'])
    if fixed_control:
        monkeypatch.setattr(c, "fit_report", lambda *_: pytest.fail("Unrequested fit report"))
    trace = c.run_episode_critic_traces(
        Adapter(), object(), None, cfg, str(tmp_path / "episode_traces"),
        unrelated_text=fixed_control, on_record=observe, include_fit=not fixed_control,
    )
    assert observed == [0, 30, 60, 90]
    saved = json.loads((tmp_path / "episode_traces/ep0000/critic_values.json").read_text())
    assert [r['snapshot_id'] for r in saved['records']] == observed
    assert all(r['swaps']['unrelated']['text'] == (fixed_control or 'open drawer') for r in saved['records'])
    with pytest.raises(ValueError, match="fixed control"):
        c.run_episode_critic_traces(Adapter(), object(), None, cfg, str(tmp_path / 'invalid'), unrelated_text='grasp blue')
    rows = trace["records"]
    assert rows[0]["swaps"]["next"]["text"] == "move blue"
    assert rows[1]["swaps"]["next"]["text"] == "grasp blue"
    assert rows[2]["subtask"] == "release blue" and rows[2]["value"] is None
    assert rows[-1]["swaps"]["next"] is None
    assert rows[1]["progress_error"] is None
    for obs, task, text, metadata in calls:
        assert obs is observations[obs["frame"]]
        assert task == "sort" and metadata == labels[obs["frame"]]
    assert (tmp_path / "episode_traces/ep0000/critic_subtask_swap.png").exists()


def _input_gradient_fixture(scale):
    # Full norm grows while the image/state norm shrinks: catch wrong ranking.
    image, state, full = .3 * (10 - scale), .4 * (10 - scale), 20 + scale
    return {"norm": full, "value": -.5, "boundary": "test embeddings",
            "state_format": "discrete", "tokens": [
                {"group": "state", "text": "<state_42>", "norm": state}],
            "groups": {"img_external_0": {"norm": image}, "state": {"norm": state},
                       "metadata": {"norm": (full**2 - image**2 - state**2)**.5}}}


def test_grouped_gradients_forward_true_conditioning_and_omit_failures(monkeypatch, tmp_path):
    rows = [_row(i * 30) for i in range(10)]
    monkeypatch.setattr(sensitivity, "probe_frame_inputs", lambda d, cfg, i, ch, metadata: {
        "obs": {"index": i}, "task": "sort shirts", "subtask": "grasp the blue shirt", "metadata": metadata,
    })
    saved = []
    def save(obs, cfg, row, root):
        assert obs["index"] == row["global_idx"]
        saved.append(row["global_idx"])
        return [{"camera": "test", "path": f"frames/{row['global_idx']}.jpg"}]
    monkeypatch.setattr(sensitivity, "_save_frame_images", save)
    class Adapter:
        chunk_size = 30
        def critic_input_gradients(self, obs, task, subtask, metadata):
            assert subtask == "grasp the blue shirt" and metadata["quality"] == 4
            if obs["index"] == 0:
                raise RuntimeError("synthetic failure")
            norm = obs["index"] / 30
            return _input_gradient_fixture(norm)
    cfg = NS(probe_parameters=NS(critic_grad_frames=10, critic_grad_frames_per_subtask=10,
                                random_seed=42, max_labels=8), env=NS(fps=30))
    raw, summary = sensitivity.run_critic_gradients(Adapter(), object(), cfg, rows, str(tmp_path))
    assert summary["grad_n_failed"] == 1 and summary["grad_n_frames"] == 9
    assert raw["grad_mags"].min() > 0
    data = json.loads((tmp_path / "critic_gradients.json").read_text())
    assert summary["gradient_scope"] == "observation"
    assert summary["image_state_grad_norm_median"] == pytest.approx(2.5)
    assert summary["grad_norm_median"] == pytest.approx(25)
    assert raw["grad_mags"].tolist() == pytest.approx([r["image_state_grad_norm"] for r in data["records"]])
    ranked = sorted(data["records"], key=lambda r: r["within_text_rank"])
    assert [r["global_idx"] for r in ranked] == list(range(270, 0, -30))
    assert saved == [r["global_idx"] for r in data["records"]]
    assert all(r["images"][0]["path"] == f"frames/{r['global_idx']}.jpg" for r in data["records"])
    assert all(r["value"] == -0.5 for r in data["records"])  # from gradient forward, not stale trace
    html = (tmp_path / "gradient_explorer.html").read_text()
    assert "test embeddings" in html and "__PROBE_DATA__" not in html
    sensitivity.write_sensitivity_manifest(tmp_path, summary)
    manifest = json.loads((tmp_path / "index.json").read_text())
    assert manifest["title"] == "Critic Input Sensitivity"
    assert any(p["file"] == "gradient_explorer.html" and p["primary"] for p in manifest["panels"])


def test_shirt_family_matches_progress_mix_instead_of_raw_frame_counts(tmp_path):
    rows = []
    for text, near_n, far_n in (("grasp the blue shirt", 12, 2), ("grasp the beige shirt", 2, 12)):
        for seconds, n, error in ((1, near_n, 0.1), (7, far_n, 0.9)):
            for i in range(n):
                rows.append(_row(i, subtask=text, seconds_to_end=seconds, value=-1 + error,
                                 reference=-1, progress_error=error / 10))
    summary = cs.shirt_family_report(rows, {"grasp the blue shirt": 2, "grasp the beige shirt": 12}, tmp_path)
    data = json.loads((tmp_path / "critic_shirt_family.json").read_text())
    assert summary["critic_shirt_value_error_matched_cells"] == 2
    assert summary["critic_shirt_colour_bias_range"] == pytest.approx(0)
    assert all(r["matched_value_error"] == pytest.approx(0.5) for r in data["per_text"])
    assert all(r["matched_progress_error"] == pytest.approx(0.05) for r in data["per_text"])
    assert (tmp_path / "critic_shirt_family.png").exists()


def test_shirt_family_balances_episodes_and_reports_missing_overlap(tmp_path):
    rows = [
        *[_row(i, seconds_to_end=1, episode=0, value=-0.5, reference=-1) for i in range(10)],
        _row(20, seconds_to_end=1, episode=1, value=-1, reference=-1),
        *[_row(i, subtask="grasp the beige shirt", seconds_to_end=8) for i in range(2)],
        _row(0, subtask="grasp the blue cup", seconds_to_end=1),
    ]
    summary = cs.shirt_family_report(rows, {}, tmp_path)
    data = json.loads((tmp_path / "critic_shirt_family.json").read_text())
    blue = next(r for r in data["cells"] if r["text"] == "grasp the blue shirt" and r["residual"]["mean"] is not None)
    assert blue["residual"]["mean"] == pytest.approx(0.25)
    assert summary["critic_shirt_texts"] == 2
    assert summary["critic_shirt_rare_value_error"] is None
    assert data["common_cells"]["value_error"] == []


def test_group_sampling_spans_each_episode_after_budget_allocation():
    rows = [_row(i * 30, episode=ep) for ep in range(2) for i in range(100)]
    picked = cs.sample_text_groups(rows, 8, 8, 42)
    for ep in range(2):
        frames = [r["global_idx"] for r in picked if r["episode"] == ep]
        assert len(frames) == 4
        assert min(frames) == 0 and max(frames) == 99 * 30
    assert cs._sample_episode_buckets([[0], list(range(100, 200))], 5) == [0, 100, 133, 166, 199]
    assert cs._sample_episode_buckets([[0, 1, 2], [3, 4, 5]], 1) == [1]
    assert cs._sample_episode_buckets([], 10) == []


def test_input_gradient_partition_covers_remaining_tokens_and_rejects_overlap():
    result = cs.gradient_group_summary([3, 4, 12, 999], {"camera": [0, 0], "state": [1], "task": []}, [0, 1, 2], 4)
    assert result["norm"] == pytest.approx(13)
    assert result["groups"]["other_prompt"]["norm"] == 12
    assert result["groups"]["state"]["rms"] == 2
    assert result["groups"]["task"]["norm"] is None
    assert sum(g["squared_norm_share"] or 0 for g in result["groups"].values()) == pytest.approx(1)
    with pytest.raises(ValueError, match="both"):
        cs.gradient_group_summary([1, 2], {"a": [0], "b": [0]}, [0, 1], 4)
    with pytest.raises(ValueError, match="sum"):
        sensitivity._validate_measurement({"norm": 10, "groups": {"image": {"norm": 1}}, "boundary": "partial"})


def test_gradient_report_escapes_prompt_text_and_preserves_point_identity(tmp_path):
    row = _row(subtask="</script><script>bad()</script>")
    row.update(grad_norm=21, images=[], gradient=_input_gradient_fixture(1))
    sensitivity.render_gradient_report([row], {"grad_n_frames": 1}, tmp_path)
    html = (tmp_path / "gradient_explorer.html").read_text()
    packed = html.split('<script type="application/json" id="probe-data">')[1].split('</script>')[0]
    decoded = json.loads(packed)
    assert "<script>bad" not in packed
    assert decoded["records"][0]["subtask"] == row["subtask"]


def test_sensitivity_manifest_does_not_show_stale_disabled_outputs(tmp_path):
    (tmp_path / "gradient_explorer.html").write_text("old result")
    (tmp_path / "episode_swaps").mkdir()
    (tmp_path / "episode_swaps/ep0000.png").write_bytes(b"old figure")
    sensitivity.write_sensitivity_manifest(tmp_path, {"grad_n_frames": 0, "grad_status": "disabled"})
    manifest = json.loads((tmp_path / "index.json").read_text())
    assert manifest["panels"] == []


def test_explicit_episode_gradient_frames_include_sparse_subtasks_and_all_points(monkeypatch, tmp_path):
    rows = [_row(i, subtask=f'step {i % 3}') for i in range(5)]
    monkeypatch.setattr(sensitivity, 'probe_frame_inputs', lambda d, cfg, i, chunk, metadata: {
        'obs': {'index': i}, 'task': rows[i]['task'], 'subtask': rows[i]['subtask'], 'metadata': metadata,
    })
    monkeypatch.setattr(sensitivity, '_save_frame_images', lambda obs, cfg, row, root: [
        {'camera': 'test', 'path': f"frames/{obs['index']}.jpg"}])
    class Adapter:
        chunk_size = 30
        def critic_input_gradients(self, obs, task, subtask, metadata):
            i = obs['index']
            assert subtask == rows[i]['subtask'] and metadata == rows[i]['metadata']
            norm = float(i+1)
            return _input_gradient_fixture(norm)
    # The explicit episode list must bypass the old eight-point minimum, label
    # cap and disabled random-sample budget. Every displayed point needs a frame.
    cfg = NS(probe_parameters=NS(critic_grad_frames=0, critic_grad_frames_per_subtask=1,
                                random_seed=42, max_labels=1))
    _, summary = sensitivity.run_critic_gradients(Adapter(), object(), cfg, rows, tmp_path,
                                                  selected_records=rows)
    result = json.loads((tmp_path/'critic_gradients.json').read_text())
    assert summary['grad_status'] == 'complete' and summary['grad_n_frames'] == 5
    assert summary['grad_n_subtasks'] == 3
    assert [r['global_idx'] for r in result['records']] == list(range(5))
    assert [r['images'][0]['path'] for r in result['records']] == [f'frames/{i}.jpg' for i in range(5)]
    assert all('grad_norm' not in r for r in rows)

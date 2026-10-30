"""Behavioral checks for the set representations probe: sampling helpers, block means and the model-free replot."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.probes import set_representations as sr


def test_even_picks_cover_both_ends_without_repeats():
    assert sr.even_picks(100, 5).tolist() == [0, 25, 50, 74, 99]
    assert sr.even_picks(3, 20).tolist() == [0, 1, 2]
    assert sr.parse_sources("droid_success:Franka, robochallenge:ARX5:2") == [("droid_success", "Franka", 1), ("robochallenge", "ARX5", 2)]
    assert sr.parse_sources("yam:YAM:ep=yam__duster__ep000106|yam__duster__ep000174,ur7e:UR7e:ep=ur7e__stack_block__ep000012") == [
        ("yam", "YAM", ["yam__duster__ep000106", "yam__duster__ep000174"]), ("ur7e", "UR7e", ["ur7e__stack_block__ep000012"])]
    with pytest.raises(ValueError):
        sr.parse_sources("droid")


def _fake_buffer(episodes):
    """A diverse buffer stand-in: rows (source, embodiment, episode_id, anchor_frame, anchor_s) per episode."""
    rows = [{"source": src, "embodiment": emb, "episode_id": ep, "anchor_frame": 3 * k, "anchor_s": 0.2 * k}
            for src, emb, ep, n in episodes for k in range(n)]
    records = {ep: {"task": f"task {ep}"} for _, _, ep, _ in episodes}
    return SimpleNamespace(rows=rows, selection=SimpleNamespace(episode_records=records))


def test_diverse_set_explicit_episodes_and_split_tag():
    buffer = _fake_buffer([("yam", "YAM", "yam__a", 12), ("yam", "YAM", "yam__b", 4), ("yam", "YAM", "yam__c", 12)])
    rng = np.random.default_rng(0)
    # explicit ids: taken as listed, even a short one (all 4 anchors), in id order
    samples = sr.diverse_set(buffer, sr.parse_sources("yam:YAM:ep=yam__c|yam__b"), 5, rng, "holdout")
    assert [s["episode"] for s in samples] == ["yam/yam__b"] * 4 + ["yam/yam__c"] * 5
    assert {s["split"] for s in samples} == {"holdout"} and samples[-1]["task"] == "task yam__c"
    assert [s["frame"] for s in samples[4:]] == [0, 9, 18, 24, 33]  # even picks over the 12 anchors, ends included
    # the random form still picks among the episodes with enough anchors, tagged train by default
    samples = sr.diverse_set(buffer, sr.parse_sources("yam:YAM:2"), 5, rng)
    assert sorted({s["episode"] for s in samples}) == ["yam/yam__a", "yam/yam__c"] and {s["split"] for s in samples} == {"train"}
    with pytest.raises(ValueError, match="yam__zz"):
        sr.diverse_set(buffer, sr.parse_sources("yam:YAM:ep=yam__zz"), 5, rng)


def test_spec_parsers_and_balanced_picks():
    assert sr.parse_rebot_sets("outputs/a:0,3; outputs/b:4") == [("outputs/a", [0, 3]), ("outputs/b", [4])]
    assert sr.parse_named("hero=outputs/x/inference_dataset:0:0-234@1;c1=outputs/y:4:13.5-19@0.5") == [
        {"label": "hero", "root": "outputs/x/inference_dataset", "episode": 0, "t0": 0.0, "t1": 234.0, "stride_s": 1.0},
        {"label": "c1", "root": "outputs/y", "episode": 4, "t0": 13.5, "t1": 19.0, "stride_s": 0.5}]
    with pytest.raises(ValueError):
        sr.parse_named("hero=outputs/x:0:0-234")
    assert sr.phase_verb("Grasp the cup") == "grasp" and sr.phase_verb("return home") == "other" and sr.phase_verb("") == "other"
    # 12 grasp, 4 move, 2 release frames, 8 picks: 3 / 3 / 2, the move and release frames all used before grasp fills up
    verbs = ["grasp"] * 12 + ["move"] * 4 + ["release"] * 2
    picks = sr.balanced_picks(verbs, 8)
    assert len(picks) == 8 and len(set(picks)) == 8
    assert [verbs[i] for i in picks].count("release") == 2 and [verbs[i] for i in picks].count("move") == 3
    assert sr.balanced_picks(["grasp"] * 5, 8).tolist() == [0, 1, 2, 3, 4]


def test_pick_episodes_covers_distinct_tasks_before_repeating():
    task_of = {0: "a", 1: "a", 2: "a", 3: "b", 4: "c"}
    for seed in range(5):
        chosen = sr.pick_episodes(list(task_of), task_of, 3, np.random.default_rng(seed))
        assert sorted(task_of[e] for e in chosen) == ["a", "b", "c"]
    chosen = sr.pick_episodes(list(task_of), task_of, 5, np.random.default_rng(0))
    assert chosen == [0, 1, 2, 3, 4]


def test_block_means_exclude_same_episode_pairs_and_gaps_follow():
    # Two sets, two episodes each; within-set cross-episode cosine 1, between-set 0.
    x = np.array([[1., 0], [1., 0], [1., 0], [1., 0], [0., 1], [0., 1], [0., 1], [0., 1]])
    episodes = ["a", "a", "b", "b", "c", "c", "d", "d"]
    sets = ["s", "s", "s", "s", "t", "t", "t", "t"]
    cosine, norms = sr.cosine_matrix(x, centered=False)
    assert norms.tolist() == [1.0] * 8
    distance = sr.distance_matrix(cosine, norms)
    assert distance[0, 4] == pytest.approx(np.sqrt(2)) and distance[0, 1] == pytest.approx(0)
    d_names, d_means, _ = sr.block_means(distance, sets, episodes)
    assert sr.set_gaps(d_names, d_means, "distance")["s_t"]["gap"] == pytest.approx(np.sqrt(2))
    names, means, counts = sr.block_means(cosine, sets, episodes)
    assert names == ["s", "t"]
    assert counts[0, 0] == 8 and counts[0, 1] == 16  # 2x2x2 ordered cross-episode pairs within; 4x4 between
    assert means[0, 0] == pytest.approx(1) and means[0, 1] == pytest.approx(0)
    gaps = sr.set_gaps(names, means)
    assert gaps["s_t"]["gap"] == pytest.approx(1)
    ep_names, ep_means, ep_counts = sr.block_means(cosine, episodes, episodes, exclude="self")
    assert ep_names == ["a", "b", "c", "d"] and ep_counts[0, 0] == 2 and ep_means[0, 0] == pytest.approx(1)
    # Centering removes the shared direction: identical-up-to-mean vectors become anticorrelated across sets.
    centered, _ = sr.cosine_matrix(x, centered=True)
    assert centered[0, 4] == pytest.approx(-1)


def _fake_cache(tmp_path, n_sets=3, eps_per_set=2, frames=4, layers=(1, 3), groups=("img_external_0", "action")):
    cache = tmp_path / "cache"
    (cache / "thumbs").mkdir(parents=True)
    rows, vectors = [], []
    rng = np.random.default_rng(0)
    for s, set_name in enumerate(sr.SETS[:n_sets]):
        for e in range(eps_per_set):
            for f in range(frames):
                rows.append({"row": len(rows), "set": set_name, "source": f"src{s}", "robot": f"robot{s}",
                             "episode": f"src{s}/{e}", "time_s": float(f), "frame": f, "task": "t", "subtask": "u",
                             "kind": "lerobot", "index": len(rows)})
                direction = np.zeros(6)
                direction[s] = 1.0
                vectors.append(direction + 0.05 * rng.standard_normal(6))
    x = np.stack(vectors).astype(np.float32)
    for group in groups:
        np.save(cache / f"{group}.npy", np.stack([x, 2 * x], axis=1))  # [rows, layers, dim]
        np.save(cache / f"{group}.present.npy", np.ones(len(rows), dtype=bool))
    meta = {"protocol": sr.PROTOCOL, "view": "attention_input", "layers": list(layers), "groups": list(groups),
            "requested_groups": list(groups), "rows": rows, "noise_seed": 0, "checkpoint": "ckpt", "flow_time": 0}
    (cache / "meta.json").write_text(json.dumps(meta))
    return cache


def test_model_free_replot_writes_page_sidecars_csvs_and_manifest(tmp_path):
    cache = _fake_cache(tmp_path)
    p = SimpleNamespace(mode="plot", setrep_view="attention_input", setrep_layers="1,3",
                        setrep_groups="img_external_0,action", random_seed=0)
    cfg = SimpleNamespace(probe_parameters=p, policy=SimpleNamespace(pretrained_path=""))
    summary = sr.run(None, None, cfg, tmp_path)
    assert summary["frames"] == 24 and summary["episodes"] == 6 and summary["sets"] == 3
    assert summary["headline_layer"] == "L3"
    gaps = summary["gap"]["centered"]["img_external_0"]["L3"]
    assert set(gaps) == {"rebot_external", "rebot_diverse", "external_diverse"}
    assert all(g > 0.5 for g in gaps.values())  # the sets point in orthogonal directions
    assert all(g > 0.5 for g in summary["gap_distance"]["centered"]["img_external_0"]["L3"].values())
    assert (tmp_path / "sets.html").exists() and (tmp_path / "blocks.csv").exists()
    sidecars = sorted(f.name for f in (tmp_path / "panels").glob("*.js"))
    assert sidecars == sorted(f"{g}.L{l}.{geo}.js" for g in ("img_external_0", "action") for l in (1, 3) for geo in sr.GEOMETRIES)
    panel = (tmp_path / "panels" / "action.L3.centered.js").read_text()
    assert panel.startswith('window.__setPanel("action.L3.centered",')
    data = json.loads(panel[panel.index(",") + 1:-2])
    assert set(data["blocks"]) == {"cosine", "distance"} and len(data["norms"]) == 24
    assert data["blocks"]["distance"]["gaps"]["rebot_diverse"]["gap"] > 0
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["id"] == tmp_path.name and index["panels"][0]["file"] == "sets.html"
    keys = {m["key"] for m in index["metrics"]}
    assert "gap.centered.action.L3.rebot_diverse" in keys and "gap.centered.img_external_0.L3.rebot_external" in keys
    trend = [m for m in index["metrics"] if m["trend"]]
    assert {m["key"] for m in trend} == {"gap.centered.action.L3.rebot_diverse", "gap.centered.img_external_0.L3.rebot_diverse"}
    assert all(m["value"] is not None for m in index["metrics"])


def test_cache_mismatch_is_rejected(tmp_path):
    _fake_cache(tmp_path)
    p = SimpleNamespace(mode="plot", setrep_view="attention_input", setrep_layers="1,2",
                        setrep_groups="img_external_0", random_seed=0)
    cfg = SimpleNamespace(probe_parameters=p, policy=SimpleNamespace(pretrained_path=""))
    with pytest.raises(ValueError, match="differ from cache"):
        sr.run(None, None, cfg, tmp_path)

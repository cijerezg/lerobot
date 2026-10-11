"""Original-space measurements for the representation trajectory probe (CPU only)."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

MATCH_KINDS = (
    "same_robot_same_task", "same_robot_different_task",
    "different_robot_same_task", "different_robot_different_task",
)


def unit_vectors(x):
    norms = np.linalg.norm(x, axis=1)
    valid = np.isfinite(x).all(axis=1) & (norms > 1e-8)
    z = np.zeros_like(x)
    z[valid] = x[valid] / norms[valid, None]
    return z, norms, valid


def robot_center(x, rows):
    """Fit per-robot means on training rows, with an explicit descriptive fallback."""
    y = x.copy()
    robots = np.array([r["robot"] for r in rows])
    training = np.array([not r["holdout"] for r in rows])
    means, provenance = {}, {}
    for robot in sorted(set(robots)):
        mask = robots == robot
        fit = mask & training
        fallback = not fit.any()
        if fallback:
            fit = mask
        means[robot] = x[fit].mean(axis=0)
        y[mask] -= means[robot]
        provenance[robot] = {"rows": int(fit.sum()), "fit": "all (no training rows)" if fallback else "training"}
    return y, means, provenance


def project_pca(x, rows):
    """One raw-scale, centered PCA across robots; never fit a basis per episode."""
    fit = np.array([not r["holdout"] for r in rows])
    fallback = int(fit.sum()) < 2
    if fallback:
        fit[:] = True
    if len(x) < 2:
        raise ValueError("PCA requires at least two captured frames")
    n = min(3, int(fit.sum()), x.shape[1])
    # Seeded randomized PCA bounds work for wide model states; it is still linear PCA.
    pca = PCA(n_components=n, svd_solver="randomized", random_state=0)
    if np.sum(np.var(x[fit], axis=0)) < 1e-16:
        mean = x[fit].mean(axis=0)
        components = np.eye(x.shape[1], dtype=x.dtype)[:n]
        coords = (x - mean) @ components.T
        variance = np.zeros(n)
    else:
        pca.fit(x[fit])
        mean, components = pca.mean_, pca.components_
        coords = pca.transform(x)
        variance = np.nan_to_num(pca.explained_variance_ratio_)
    return np.pad(coords, ((0, 0), (0, 3 - n))), {
        "mean": mean, "components": components, "variance": variance,
        "fit_rows": np.flatnonzero(fit), "fallback": fallback,
    }


def temporal_metrics(x, rows, window_s, raw_norms=None):
    """Self-excluded local cosines in seconds, plus actual previous-frame gaps."""
    z, norms, valid = unit_vectors(x)
    raw_norms = norms if raw_norms is None else raw_norms
    episodes = np.array([r["episode"] for r in rows])
    times = np.array([r["time_s"] for r in rows])
    result = [dict(local_cosine=None, local_count=0, previous_cosine=None,
                   previous_dt_s=None, step_l2=None, raw_norm=float(raw_norms[i]),
                   geometry_norm=float(norms[i])) for i in range(len(rows))]
    for episode in sorted(set(episodes)):
        ids = np.flatnonzero(episodes == episode)
        ids = ids[np.argsort(times[ids], kind="stable")]
        for pos, i in enumerate(ids):
            near = ids[(np.abs(times[ids] - times[i]) <= window_s) & (ids != i) & valid[ids]]
            if valid[i] and len(near):
                result[i]["local_cosine"] = float(np.clip(z[near] @ z[i], -1, 1).mean())
                result[i]["local_count"] = len(near)
            if pos:
                prev = ids[pos - 1]
                result[i]["previous_dt_s"] = float(times[i] - times[prev])
                result[i]["step_l2"] = float(np.linalg.norm(x[i] - x[prev]))
                if valid[i] and valid[prev]:
                    result[i]["previous_cosine"] = float(np.clip(z[i] @ z[prev], -1, 1))
    return result


def cross_episode_matches(x, rows, window_s, raw_norms=None, block_size=128):
    """Exact nearest cosine by category, in bounded blocks; no N x N output/cache.

    Invalid/zero vectors are ineligible. Neighborhood support compares the anchor
    with frames in the *matched* episode, near the matched time and with its task.
    Candidate counts and absent matches are retained rather than inventing zeros.
    """
    z, norms, valid = unit_vectors(x)
    raw_norms = norms if raw_norms is None else raw_norms
    ep = np.array([r["episode"] for r in rows])
    robot = np.array([r["robot"] for r in rows])
    task = np.array([r["task"] for r in rows])
    times = np.array([r["time_s"] for r in rows])
    result = [{kind: None for kind in MATCH_KINDS} for _ in rows]
    for start in range(0, len(rows), block_size):
        sims = z[start:start + block_size] @ z.T
        for offset, similarities in enumerate(sims):
            i = start + offset
            if not valid[i]:
                continue
            foreign = (ep != ep[i]) & valid
            same_robot, same_task = robot == robot[i], task == task[i]
            masks = (same_robot & same_task, same_robot & ~same_task,
                     ~same_robot & same_task, ~same_robot & ~same_task)
            for kind, mask in zip(MATCH_KINDS, masks, strict=True):
                candidates = np.flatnonzero(foreign & mask)
                if not len(candidates):
                    continue
                j = int(candidates[np.argmax(similarities[candidates])])
                near = np.flatnonzero((ep == ep[j]) & (task == task[j]) & valid & (np.abs(times - times[j]) <= window_s))
                result[i][kind] = {
                    "target": j, "cosine": float(np.clip(similarities[j], -1, 1)),
                    "raw_norm_delta": float(raw_norms[j] - raw_norms[i]),
                    "raw_norm_ratio": float(raw_norms[j] / raw_norms[i]) if raw_norms[i] > 1e-8 else None,
                    "distance": float(np.linalg.norm(x[i] - x[j])),
                    "target_window_cosine": float(np.clip(z[near] @ z[i], -1, 1).mean()),
                    "target_window_count": len(near), "candidate_count": len(candidates),
                    "candidate_episodes": len(set(ep[candidates])),
                }
    return result


def _write_csv(path, records):
    if not records:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def analyze(meta, arrays, present, layers, groups, window_s, output_dir, constant_text=True):
    root = Path(output_dir)
    (root / "pca").mkdir(parents=True, exist_ok=True)
    all_rows = meta["rows"]
    texts = ("real", "neutral") if constant_text else ("real",)
    panels, temporal_csv, matches_csv = [], [], []
    # Small matrices are slower with dozens of BLAS threads, especially in tests.
    with threadpool_limits(limits=2):
        for group in groups:
            if group not in arrays:
                logging.warning("Representation group %s was absent at capture", group)
                continue
            for layer in layers:
                li = meta["layers"].index(layer)
                for text in texts:
                    ids = np.flatnonzero(present[group] & np.array([r["text"] == text for r in all_rows]))
                    if len(ids) < 2:
                        continue
                    rows = [all_rows[i] for i in ids]
                    x = np.asarray(arrays[group][ids, li], dtype=np.float64)
                    if not np.isfinite(x).all():
                        raise ValueError(f"Nonfinite cached vectors in {group}/L{layer}/{text}")
                    raw_norms = np.linalg.norm(x, axis=1)
                    centered, means, centering = robot_center(x, rows)
                    for geometry, values in (("raw", x), ("robot_centered", centered)):
                        logging.info("Analyze trajectories: %s L%s %s %s", group, layer, text, geometry)
                        coords, pca = project_pca(values, rows)
                        temporal = temporal_metrics(values, rows, window_s, raw_norms)
                        matches = cross_episode_matches(values, rows, window_s, raw_norms)
                        key = f"{group}.L{layer}.{text}.{geometry}"
                        np.savez(root / "pca" / f"{key}.npz", mean=pca["mean"], components=pca["components"],
                                 variance=pca["variance"], fit_rows=ids[pca["fit_rows"]],
                                 row_ids=ids, coordinates=coords,
                                 robot_names=np.array(list(means)), robot_means=np.stack(list(means.values())))
                        header = {"group": group, "layer": layer, "text": text, "geometry": geometry}
                        for i, row in enumerate(rows):
                            identity = {"row": row["row"], "robot": row["robot"], "episode": row["episode"],
                                        "time_s": row["time_s"], "task": row["task"], "subtask": row["subtask"],
                                        "holdout": row["holdout"]}
                            temporal_csv.append({**header, **identity, **temporal[i]})
                            for kind, match in matches[i].items():
                                if match is not None:
                                    target = rows[match["target"]]
                                    matches_csv.append({**header, **identity, "kind": kind,
                                                        "target_row": target["row"], "target_episode": target["episode"],
                                                        "target_time_s": target["time_s"], "target_task": target["task"],
                                                        "target_subtask": target["subtask"],
                                                        **{k: v for k, v in match.items() if k != "target"}})
                        panels.append({**header, "key": key, "ids": ids.tolist(),
                                       "xyz": coords.tolist(), "variance": pca["variance"].tolist(),
                                       "pca_fit": "all (fewer than two training rows)" if pca["fallback"] else "training",
                                       "centering": centering, "temporal": temporal, "matches": matches})
    if not panels:
        raise ValueError("No selected group/text has enough captured frames to analyze")
    _write_csv(root / "temporal.csv", temporal_csv)
    _write_csv(root / "matches.csv", matches_csv)
    real_rows = [r for r in all_rows if r["text"] == "real"]
    summary = {"frames": len(real_rows), "episodes": len({r["episode"] for r in real_rows}),
               "robots": len({r["robot"] for r in real_rows}), "view": meta["view"], "panels": len(panels),
               "local_window_s": window_s, "missing_groups": sorted(set(groups) - set(arrays))}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return {"rows": all_rows, "panels": panels, "window_s": window_s,
            "view": meta["view"], "checkpoint": meta.get("checkpoint", "")}, summary

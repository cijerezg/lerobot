"""Original-space measurements for the representation trajectory probe (CPU only)."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

MATCH_KINDS = (
    "same_robot_same_task", "same_robot_different_task",
    "different_robot_same_task", "different_robot_different_task",
)
GEOMETRIES = ("raw", "robot_centered", "episode_centered")
# Matches per category: the best frame of each of the top-k other episodes.
MATCH_TOP_K = 5
BACKGROUND_KEYS = ("background_same_robot", "background_other_robot",
                   "background_distance_same_robot", "background_distance_other_robot",
                   "background_lognorm_same_robot", "background_lognorm_other_robot")
TEMPORAL_KEYS = ("local_cosine", "local_count", "previous_cosine", "previous_dt_s", "step_l2",
                 "raw_norm", "geometry_norm", *BACKGROUND_KEYS)
MATCH_KEYS = ("target", "cosine", "raw_norm_delta", "raw_norm_ratio", "distance",
              "target_window_cosine", "target_window_count", "candidate_count", "candidate_episodes")


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


def episode_center(x, rows):
    """Subtract each episode's own sampled mean: what remains is motion within the episode.

    This removes the scene/episode identity that dominates raw cosines, so the
    within-episode structure is about where the frame sits in the
    episode rather than which episode it is. Descriptive by construction: every
    episode uses all of its own sampled frames, train or held out.
    """
    y = x.copy()
    episodes = np.array([r["episode"] for r in rows])
    for episode in set(episodes):
        mask = episodes == episode
        y[mask] -= x[mask].mean(axis=0)
    return y


def episode_order(rows):
    """Row indices of each episode in time order."""
    episodes = np.array([r["episode"] for r in rows])
    times = np.array([r["time_s"] for r in rows])
    order = {}
    for episode in sorted(set(episodes)):
        ids = np.flatnonzero(episodes == episode)
        order[episode] = ids[np.argsort(times[ids], kind="stable")]
    return order, times


def temporal_metrics(x, rows, window_s, raw_norms=None):
    """Self-excluded local cosines in seconds, plus actual previous-frame gaps."""
    z, norms, valid = unit_vectors(x)
    raw_norms = norms if raw_norms is None else raw_norms
    order, times = episode_order(rows)
    result = [dict(local_cosine=None, local_count=0, previous_cosine=None,
                   previous_dt_s=None, step_l2=None, raw_norm=float(raw_norms[i]),
                   geometry_norm=float(norms[i])) for i in range(len(rows))]
    for ids in order.values():
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


def full_cosine(x):
    """Every frame against every frame in the original dimensions; invalid rows are NaN."""
    z, _, valid = unit_vectors(x)
    c = np.clip(z @ z.T, -1, 1)
    c[~valid, :] = np.nan
    c[:, ~valid] = np.nan
    return c


def episode_cosine_matrices(x, rows, full=None):
    """Within-episode cosine matrices in time order; invalid rows/columns are NaN.

    Every local or lagged statistic is a function of this matrix and the frame times,
    so the report derives them interactively instead of fixing one window here.
    """
    full = full_cosine(x) if full is None else full
    order, times = episode_order(rows)
    return {episode: {"ids": ids, "times": times[ids], "cosine": full[np.ix_(ids, ids)]}
            for episode, ids in order.items()}


def pack_cosine(matrix):
    """Base64 little-endian uint16: 0 is missing, 1..65535 spans cosine -1..1 (3e-5 steps).

    The report decodes this for the anchored cross-episode grid; it is the only
    cross-episode object stored whole, so it is quantized rather than written as text.
    """
    import base64

    m = np.asarray(matrix, dtype=np.float64)
    q = np.where(np.isfinite(m), np.round((np.clip(m, -1, 1) + 1) / 2 * 65534) + 1, 0)
    return base64.b64encode(q.astype("<u2").tobytes()).decode()


def unpack_cosine(text, n):
    import base64

    q = np.frombuffer(base64.b64decode(text), dtype="<u2").reshape(n, n).astype(np.float64)
    return np.where(q > 0, (q - 1) / 65534 * 2 - 1, np.nan)


def cross_episode_matches(x, rows, window_s, raw_norms=None, block_size=128, top_k=MATCH_TOP_K):
    """Exact nearest cosine by category, in bounded blocks; no N x N output/cache.

    Each category returns up to ``top_k`` matches, best first: the best frame of each
    of the ``top_k`` closest other episodes (one frame per episode, so the list is a
    browse over alternatives rather than the same moment five times). Invalid/zero
    vectors are ineligible. Neighborhood support compares the anchor with frames in
    the *matched* episode, near the matched time and with its task. Candidate counts
    and absent matches are retained rather than inventing zeros. Also returns the
    per-frame *background*: mean cosine, mean L2 distance and mean log2 norm ratio
    (other / this, geometry norms) to every valid frame of other episodes, split by
    same/other robot: the level a "local" value must clear before it says anything
    about the episode rather than the robot.
    """
    z, norms, valid = unit_vectors(x)
    raw_norms = norms if raw_norms is None else raw_norms
    ep = np.array([r["episode"] for r in rows])
    robot = np.array([r["robot"] for r in rows])
    task = np.array([r["task"] for r in rows])
    times = np.array([r["time_s"] for r in rows])
    result = [{kind: [] for kind in MATCH_KINDS} for _ in rows]
    background = [dict.fromkeys(BACKGROUND_KEYS) for _ in rows]
    for start in range(0, len(rows), block_size):
        sims = np.clip(z[start:start + block_size] @ z.T, -1, 1)
        for offset, similarities in enumerate(sims):
            i = start + offset
            if not valid[i]:
                continue
            foreign = (ep != ep[i]) & valid
            same_robot, same_task = robot == robot[i], task == task[i]
            for side, pool in (("same_robot", foreign & same_robot), ("other_robot", foreign & ~same_robot)):
                if pool.any():
                    n_i, n_j, c = norms[i], norms[pool], similarities[pool]
                    background[i][f"background_{side}"] = float(c.mean())
                    background[i][f"background_distance_{side}"] = float(np.sqrt(np.maximum(n_i**2 + n_j**2 - 2 * n_i * n_j * c, 0)).mean())
                    background[i][f"background_lognorm_{side}"] = float(np.log2(n_j / n_i).mean())
            masks = (same_robot & same_task, same_robot & ~same_task,
                     ~same_robot & same_task, ~same_robot & ~same_task)
            for kind, mask in zip(MATCH_KINDS, masks, strict=True):
                candidates = np.flatnonzero(foreign & mask)
                if not len(candidates):
                    continue
                ranked = candidates[np.argsort(-similarities[candidates], kind="stable")]
                taken = set()
                for j in ranked:
                    if ep[j] in taken:
                        continue
                    taken.add(ep[j])
                    j = int(j)
                    near = np.flatnonzero((ep == ep[j]) & (task == task[j]) & valid & (np.abs(times - times[j]) <= window_s))
                    result[i][kind].append({
                        "target": j, "cosine": float(similarities[j]),
                        "raw_norm_delta": float(raw_norms[j] - raw_norms[i]),
                        "raw_norm_ratio": float(raw_norms[j] / raw_norms[i]) if raw_norms[i] > 1e-8 else None,
                        "distance": float(np.linalg.norm(x[i] - x[j])),
                        "target_window_cosine": float(np.clip(z[near] @ z[i], -1, 1).mean()),
                        "target_window_count": len(near), "candidate_count": len(candidates),
                        "candidate_episodes": len(set(ep[candidates])),
                    })
                    if len(taken) >= top_k:
                        break
    return result, background


def _write_csv(path, records):
    if not records:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def compact(values, decimals=4):
    """Rounded nested lists with None for non-finite entries, for a small JSON payload."""
    a = np.asarray(values, dtype=np.float64)
    out = np.round(a, decimals).astype(object)
    out[~np.isfinite(a)] = None
    return out.tolist()


def _columns(records, keys):
    return {key: [record.get(key) for record in records] for key in keys}


def _ranked_columns(lists, keys):
    """Per key, one list per frame holding that key for each ranked match (best first)."""
    return {key: [[match[key] for match in matches] for matches in lists] for key in keys}


def analyze(meta, arrays, present, layers, groups, window_s, output_dir, constant_text=True, cross_matrices=True):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
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
                    centered, _, centering = robot_center(x, rows)
                    geometries = {"raw": x, "robot_centered": centered, "episode_centered": episode_center(x, rows)}
                    for geometry in GEOMETRIES:
                        values = geometries[geometry]
                        logging.info("Analyze trajectories: %s L%s %s %s", group, layer, text, geometry)
                        temporal = temporal_metrics(values, rows, window_s, raw_norms)
                        matches, background = cross_episode_matches(values, rows, window_s, raw_norms)
                        full = full_cosine(values)
                        matrices = episode_cosine_matrices(values, rows, full)
                        key = f"{group}.L{layer}.{text}.{geometry}"
                        header = {"group": group, "layer": layer, "text": text, "geometry": geometry}
                        for i, row in enumerate(rows):
                            temporal[i].update(background[i])
                            identity = {"row": row["row"], "robot": row["robot"], "episode": row["episode"],
                                        "time_s": row["time_s"], "task": row["task"], "subtask": row["subtask"],
                                        "holdout": row["holdout"]}
                            temporal_csv.append({**header, **identity, **temporal[i]})
                            for kind, ranked in matches[i].items():
                                for rank, match in enumerate(ranked, start=1):
                                    target = rows[match["target"]]
                                    matches_csv.append({**header, **identity, "kind": kind, "rank": rank,
                                                        "target_row": target["row"], "target_episode": target["episode"],
                                                        "target_time_s": target["time_s"], "target_task": target["task"],
                                                        "target_subtask": target["subtask"],
                                                        **{k: v for k, v in match.items() if k != "target"}})
                        episodes = {episode: {"ids": m["ids"].tolist(), "times": compact(m["times"]),
                                              "cosine": compact(m["cosine"])}
                                    for episode, m in matrices.items()}
                        panels.append({**header, "key": key, "ids": ids.tolist(),
                                       "centering": centering, "temporal": _columns(temporal, TEMPORAL_KEYS),
                                       "matches": {kind: _ranked_columns([m[kind] for m in matches], MATCH_KEYS)
                                                   for kind in MATCH_KINDS},
                                       "episodes": episodes,
                                       "cross": pack_cosine(full) if cross_matrices else None})
    if not panels:
        raise ValueError("No selected group/text has enough captured frames to analyze")
    _write_csv(root / "temporal.csv", temporal_csv)
    _write_csv(root / "matches.csv", matches_csv)
    real_rows = [r for r in all_rows if r["text"] == "real"]
    summary = {"frames": len(real_rows), "episodes": len({r["episode"] for r in real_rows}),
               "robots": len({r["robot"] for r in real_rows}), "view": meta["view"], "panels": len(panels),
               "local_window_s": window_s, "match_top_k": MATCH_TOP_K, "geometries": list(GEOMETRIES),
               "cross_matrices": bool(cross_matrices),
               "missing_groups": sorted(set(groups) - set(arrays))}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return {"rows": all_rows, "panels": panels, "window_s": window_s, "match_top_k": MATCH_TOP_K,
            "stride_s": meta.get("stride_s"), "view": meta["view"], "checkpoint": meta.get("checkpoint", "")}, summary

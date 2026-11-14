#!/usr/bin/env python
"""Re-key ReBot memmap caches for new dataset roots made by `cp -al` of old ones.

The replay cache key (ReplayBuffer._dataset_fingerprint) hashes the root path, frame and
episode counts, state keys and image storage settings; the cache itself holds frames,
states, actions, rewards, dones and depth rows and nothing from meta/ (quality, mistake,
speed, precision and contact are materialized from the parquets at load). A new root
that shares every data/videos/depth file with its old twin therefore needs the same
bytes under a new fingerprint. For each NEW=OLD pair this script

  * finds the old cache dir by its metadata.json `dataset_root` (spelled as OLD),
  * checks the repo's fingerprint of OLD reproduces that dir's name (same settings),
  * checks NEW has OLD's frame/episode counts, state keys, and shares every file of OLD
    by inode (files only in NEW, e.g. new meta tables, are listed, not fatal),
  * creates <cache_dir>/<fingerprint(NEW)>/ with hardlinks to OLD's .bin files and a new
    metadata.json (fingerprint, dataset_root, linked_from). The old dir is never touched.

--exclude NAME.bin leaves a .bin out (someone supplies a fixed copy). The metadata is
then written as metadata.pending.json so the incomplete dir is not a cache hit; rename it
to metadata.json once every excluded .bin is in place. Video files of an excluded image
key may differ from OLD. --allow-changed PATH accepts another root-relative file that
differs from OLD (only when you know the cache does not depend on it).

Roots are hashed as spelled: pass them exactly as config_rl.yaml spells them (relative to
the LeRobot dir), and run from there.

    uv run python lerobot/scripts/link_caches.py \\
        [--config config_rl.yaml] [--cache-dir DIR] [--dry-run] \\
        [--exclude observation.images.external_0.bin ...] [--allow-changed meta/x.json ...] \\
        NEW=OLD [NEW=OLD ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path

from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.offline_dataset_utils import buffer_state_keys, get_offline_dataset_sources, load_offline_dataset
from lerobot.scripts.diverse_smoke import _load_config


def _settings(cfg, dataset) -> dict:
    return dict(
        state_keys=buffer_state_keys(cfg, dataset),
        image_storage_dtype=getattr(cfg.policy, "image_storage_dtype", "bfloat16"),
        image_storage_size=getattr(cfg.policy, "image_storage_size", (224, 224)),
        image_stride=getattr(cfg.policy, "image_stride", 1),
    )


def _load(cfg, root: str):
    sources = get_offline_dataset_sources(cfg)
    source = next((s for s in sources if s.root == root), None)
    if source is None:
        source = dataclasses.replace(
            sources[0], name=Path(root).name, root=root, normalization_source=False, episodes=None
        )
    return load_offline_dataset(cfg, source)


def _files(root: Path) -> dict[str, tuple[int, int]]:
    out = {}
    for dirpath, _dirs, names in os.walk(root):
        for name in names:
            path = Path(dirpath) / name
            try:
                st = os.stat(path)  # follows symlinks: a re-pointed video link differs
            except FileNotFoundError:
                st = os.lstat(path)
            out[str(path.relative_to(root))] = (st.st_dev, st.st_ino)
    return out


def _old_cache(cache_dir: Path, old: str) -> Path:
    hits = []
    for meta_path in sorted(cache_dir.glob("*/metadata.json")):
        if json.loads(meta_path.read_text()).get("dataset_root") == old:
            hits.append(meta_path.parent)
    if len(hits) != 1:
        raise SystemExit(f"{old}: expected one cache with dataset_root == {old!r} under {cache_dir}, found {hits}")
    return hits[0]


def _plan(cfg, cache_dir: Path, new: str, old: str, exclude: set[str], allow_changed: set[str]) -> dict:
    old_cache = _old_cache(cache_dir, old)
    old_meta = json.loads((old_cache / "metadata.json").read_text())

    old_ds = _load(cfg, old)
    settings = _settings(cfg, old_ds)
    old_fp = ReplayBuffer._dataset_fingerprint(old_ds, **settings)
    if old_fp != old_cache.name or old_meta.get("fingerprint") != old_fp:
        raise SystemExit(
            f"{old}: fingerprint under this config is {old_fp}, cache dir is {old_cache.name} "
            f"(metadata {old_meta.get('fingerprint')}); settings differ from the build — refusing"
        )

    new_ds = _load(cfg, new)
    new_settings = _settings(cfg, new_ds)
    if sorted(new_settings["state_keys"]) != sorted(settings["state_keys"]):
        raise SystemExit(f"{new}: state keys {new_settings['state_keys']} != {settings['state_keys']}")
    counts = lambda ds: (int(ds.meta.total_frames), int(ds.meta.total_episodes))  # noqa: E731
    if counts(new_ds) != counts(old_ds):
        raise SystemExit(f"{new}: frames/episodes {counts(new_ds)} != {old} {counts(old_ds)}")

    bins = sorted(p.name for p in old_cache.iterdir() if p.suffix == ".bin")
    unknown = exclude - set(bins)
    if unknown:
        raise SystemExit(f"--exclude names not in {old_cache}: {sorted(unknown)}")
    # Video files of an excluded image key are expected to differ (camera fix).
    excluded_videos = tuple(f"videos/{name[:-4]}/" for name in exclude)

    new_files, old_files = _files(Path(new)), _files(Path(old))
    only_old = sorted(set(old_files) - set(new_files))
    only_new = sorted(set(new_files) - set(old_files))
    changed = sorted(k for k in set(old_files) & set(new_files) if old_files[k] != new_files[k])
    blocking = [
        k for k in only_old + changed if k not in allow_changed and not k.startswith(excluded_videos)
    ]
    if blocking:
        raise SystemExit(
            f"{new}: {len(blocking)} file(s) of {old} missing or not hardlinked in the new root, "
            f"so the cache bytes may not be its bytes: {blocking[:20]}"
        )

    new_fp = ReplayBuffer._dataset_fingerprint(new_ds, **new_settings)
    target = cache_dir / new_fp
    if target.exists():
        raise SystemExit(f"{target} already exists; refusing to touch it")

    new_meta = dict(old_meta)
    new_meta["fingerprint"] = new_fp
    new_meta["dataset_root"] = str(new_ds.root)
    new_meta["linked_from"] = {"cache": str(old_cache), "dataset_root": old}
    if "prepared_contract_sha256" in old_meta:
        new_meta["prepared_contract_sha256"] = hashlib.sha256(
            (Path(new) / "meta" / "cache_ready.json").read_bytes()
        ).hexdigest()
    if exclude:
        new_meta["excluded_bins"] = sorted(exclude)
    return dict(
        old_cache=old_cache,
        target=target,
        link=[b for b in bins if b not in exclude],
        exclude=sorted(exclude),
        meta=new_meta,
        only_new=only_new,
        changed=changed,
        frames=counts(new_ds),
        dataset=new_ds,
        settings=new_settings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pairs", nargs="+", metavar="NEW=OLD")
    parser.add_argument("--config", default="config_rl.yaml")
    parser.add_argument("--cache-dir", default=None, help="default: buffer_cache_dir of --config")
    parser.add_argument("--exclude", action="append", default=[], metavar="NAME.bin")
    parser.add_argument("--allow-changed", action="append", default=[], metavar="ROOT_REL_PATH")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    cache_dir = Path(args.cache_dir or cfg.buffer_cache_dir)
    pairs = []
    for pair in args.pairs:
        new, sep, old = pair.partition("=")
        if not sep or not new or not old:
            raise SystemExit(f"bad pair {pair!r}; expected NEW=OLD")
        pairs.append((new.rstrip("/"), old.rstrip("/")))

    # Plan every pair before writing anything.
    plans = [_plan(cfg, cache_dir, new, old, set(args.exclude), set(args.allow_changed)) for new, old in pairs]
    for (new, old), plan in zip(pairs, plans):
        print(f"{new} = {old}  ({plan['frames'][0]} frames, {plan['frames'][1]} episodes)")
        print(f"  old cache {plan['old_cache']}")
        print(f"  new cache {plan['target']}")
        print(f"  link {len(plan['link'])} .bin: {plan['link']}")
        if plan["exclude"]:
            print(f"  EXCLUDED {plan['exclude']} -> metadata.pending.json until supplied")
        print(f"  files only in new root: {plan['only_new']}")
        if plan["changed"]:
            print(f"  files changed vs old root (allowed): {plan['changed'][:20]}")
        if args.dry_run:
            continue
        target = plan["target"]
        target.mkdir(parents=False)
        for name in plan["link"]:
            os.link(plan["old_cache"] / name, target / name)
        meta_name = "metadata.pending.json" if plan["exclude"] else "metadata.json"
        (target / meta_name).write_text(json.dumps(plan["meta"], indent=2))
        if not plan["exclude"]:
            resolved = ReplayBuffer.find_cache(plan["dataset"], cache_dir, **plan["settings"])
            assert resolved == target, (resolved, target)
        print(f"  WROTE {target} ({meta_name})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU sample test for the mixed ReBot + diverse collection (integration plan gate J).

Builds exactly what rl_offline.py builds -- the ReBot buffers, the diverse buffer, the
hierarchical iterator and the real preprocessor -- draws batches, and reports what came
out. No model, no GPU, no gradient: this is the cheap check that runs before the smoke
training run and answers the questions that do not need a forward pass.

With the preprocessor on, the first batch is also checked clause by clause: every
sample's rendered prompt must carry exactly the quality / mistake / speed / precision /
contact its metadata columns say, and a diverse sample's speed and step text must be the
ones its selection row holds (the reviewed atom spanning the anchor). Precision and contact
are optional channels: a -1 column must render no clause. A mismatch is an error, not a
log line.

    uv run --no-project --python .venv/bin/python \\
        python -m lerobot.scripts.diverse_smoke --config_path=config_rl.yaml --batches 8
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter

import torch

from lerobot.datasets.contact_vocab import CONTACT_VOCAB
from lerobot.rl.data_sources.diverse_integration import (
    align_rebot_buffers,
    build_diverse_buffer,
    build_mixture_groups,
    probe_rebot_caches,
    sample_spec_from_config,
)
from lerobot.rl.data_sources.diverse_mixture import (
    MixtureTelemetry,
    allocate_group_quotas,
    make_hierarchical_offline_iterator,
)
from lerobot.utils.constants import ACTION, OBS_STATE

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

SOURCE_NAMES = {0: "droid", 1: "droid_success", 2: "fmb", 3: "robochallenge", 4: "ur7e", 5: "rebot", 6: "molmoact", 7: "yam"}
REBOT_SOURCE_ID = 5

_QUALITY_CLAUSE = re.compile(r"The quality is (\d) of 5\.")
_SPEED_CLAUSE = re.compile(r"The speed is (\d) of 5\.")
_PRECISION_CLAUSE = re.compile(r"The precision is (\d) of 5\.")
_CONTACT_CLAUSE = re.compile(r"The contact is (.+?)\.(?= The | Given |$)")
_CONTACT_CODE_BY_PHRASE = {element.phrase: element.code for element in CONTACT_VOCAB}
_STEP_CLAUSE = re.compile(r"The current step is (.+?)\.(?= The | Given )")


def parse_precision_clauses(text: str) -> list[int]:
    """Every precision level rendered in ``text`` (empty when the clause is omitted)."""
    return [int(level) for level in _PRECISION_CLAUSE.findall(text)]


def parse_contact_clauses(text: str) -> list[int]:
    """Every contact clause in ``text`` as its CONTACT_VOCAB code (empty when omitted).
    A phrase outside the vocabulary is an error, not a silent miss."""
    codes = []
    for phrase in _CONTACT_CLAUSE.findall(text):
        if phrase not in _CONTACT_CODE_BY_PHRASE:
            raise ValueError(f"contact clause phrase {phrase!r} is not in the vocabulary")
        codes.append(_CONTACT_CODE_BY_PHRASE[phrase])
    return codes


def _expected_clause(value) -> list[int]:
    """The clause list a column value must render: none for the -1 (unlabelled) sentinel."""
    return [int(value)] if float(value) >= 0 else []


def check_metadata_prompts(packed: dict, info: dict, step, diverse_rows: list[dict]) -> Counter:
    """Every rendered prompt against its own metadata columns and, for diverse rows, the
    selection row the sample came from. Returns the count of checked clauses."""
    if float(step.metadata_dropout) > 0:
        raise ValueError("metadata_dropout > 0: the prompt check needs deterministic clauses.")
    tokenizer = step.processor.tokenizer
    quality = info["metadata_quality"].reshape(-1).tolist()
    quality_valid = info["metadata_quality_is_valid"].reshape(-1).tolist()
    mistake = info["metadata_mistake"].reshape(-1).tolist()
    speed = info["metadata_speed"].reshape(-1).tolist()
    # A batch without the column is a pre-channel buffer: every row unlabelled.
    precision, contact = (
        info[key].reshape(-1).tolist() if key in info else [-1] * len(speed)
        for key in ("metadata_precision", "metadata_contact")
    )
    source = info["source_id"].reshape(-1).tolist()
    row_index = info["diverse_row_index"].reshape(-1).tolist()
    counts: Counter = Counter()
    for i, ids in enumerate(packed["input_ids"]):
        text = tokenizer.decode(ids)
        found_quality = _QUALITY_CLAUSE.findall(text)
        found_speed = _SPEED_CLAUSE.findall(text)
        found_precision = parse_precision_clauses(text)
        found_contact = parse_contact_clauses(text)
        if int(source[i]) != REBOT_SOURCE_ID:
            row = diverse_rows[int(row_index[i])]
            if int(speed[i]) != int(row["speed"]):
                raise ValueError(f"sample {i}: column speed {speed[i]} != selection row speed {row['speed']}")
            for name, column in (("precision", precision), ("contact", contact)):
                if int(column[i]) != int(row.get(name, -1)):
                    raise ValueError(
                        f"sample {i}: column {name} {column[i]} != selection row {name} {row.get(name, -1)}"
                    )
            # The step clause is the reviewed atom, not the parent interval it tiles.
            step = _STEP_CLAUSE.findall(text)
            if step != [str(row["subtask"]).rstrip(".")]:
                raise ValueError(f"sample {i}: prompt step {step} != atom {row['subtask']!r}")
            counts["diverse_speed_and_step_vs_selection"] += 1
            counts["step_is_the_parent_interval"] += int(row["subtask"] == row["parent_subtask"])
        if found_speed != [str(int(speed[i]))]:
            raise ValueError(f"sample {i}: prompt speed {found_speed} vs column {speed[i]}: {text[-400:]}")
        if found_precision != _expected_clause(precision[i]):
            raise ValueError(f"sample {i}: prompt precision {found_precision} vs column {precision[i]}")
        if found_contact != _expected_clause(contact[i]):
            raise ValueError(f"sample {i}: prompt contact {found_contact} vs column {contact[i]}")
        if bool(quality_valid[i]):
            if found_quality != [str(int(quality[i]))]:
                raise ValueError(f"sample {i}: prompt quality {found_quality} vs column {quality[i]}")
        elif found_quality:
            raise ValueError(f"sample {i}: quality withheld but rendered: {found_quality}")
        sentence = "The robot made a mistake." if mistake[i] > 0.5 else "The robot made no mistakes."
        if sentence not in text:
            raise ValueError(f"sample {i}: mistake column {mistake[i]} but '{sentence}' absent")
        counts["prompts"] += 1
        counts[f"speed_{int(speed[i])}"] += 1
        counts["speed_clauses"] += len(found_speed)
        counts["precision_clauses"] += len(found_precision)
        counts["contact_clauses"] += len(found_contact)
    return counts


def _load_config(path: str):
    # Import for their registry side effects, the way the training entry point does.
    import draccus

    import lerobot.teleoperators.rebot_102_leader  # noqa: F401
    from lerobot.cameras import opencv, realsense  # noqa: F401
    from lerobot.configs.train import TrainRLServerPipelineConfig
    from lerobot.rl.molmoact2.rl_molmoact2 import MolmoAct2RLConfig  # noqa: F401
    from lerobot.robots.rebot_b601_follower import RebotB601Follower  # noqa: F401

    return draccus.parse(TrainRLServerPipelineConfig, args=[f"--config_path={path}"])


def _rebot_buffers(cfg, history_offsets):
    from lerobot.rl.buffer import ReplayBuffer
    from lerobot.rl.offline_dataset_utils import (
        buffer_state_keys,
        get_offline_dataset_sources,
        load_metadata_rows,
        load_offline_dataset,
        materialize_dataset_labels,
    )

    sources = get_offline_dataset_sources(cfg)
    datasets, buffers = [], []
    for index, source in enumerate(sources):
        dataset = load_offline_dataset(cfg, source)
        cached = ReplayBuffer.find_cache(
            dataset,
            cfg.buffer_cache_dir,
            state_keys=buffer_state_keys(cfg, dataset),
            image_storage_dtype=cfg.policy.image_storage_dtype,
            image_storage_size=cfg.policy.image_storage_size,
            image_stride=cfg.policy.image_stride,
        )
        if cached is None:
            raise FileNotFoundError(f"no ReBot cache for {source.name!r} under {cfg.buffer_cache_dir!r}")
        buffer = ReplayBuffer.from_cache(
            cached, device="cpu", use_drq=False, history_offsets=history_offsets
        )
        materialize_dataset_labels(
            buffer,
            dataset,
            datasets[0] if datasets else dataset,
            source_index=index,
            is_main_process=True,
            require_depth_gripper_event_labels=bool(cfg.policy.depth_gripper_event_loss.enabled),
            embodiment=source.embodiment,
        )
        if cfg.policy.memory.metadata_enabled:
            buffer.materialize_metadata(*load_metadata_rows(dataset.root))
        # build_mixture_groups reads the sampling group off it, as in training.
        buffer.offline_source = source
        datasets.append(dataset)
        buffers.append(buffer)
    return datasets[0], buffers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", default="config_rl.yaml")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--skip-preprocessor", action="store_true")
    parser.add_argument(
        "--cache-policy",
        default=None,
        choices=("require", "fallback"),
        help="Override the config, e.g. to smoke the uncached path before the cache exists.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config_path)
    if args.cache_policy is not None:
        cfg.cache_policy = args.cache_policy
    if not cfg.diverse.enabled:
        raise SystemExit(f"{args.config_path} has diverse.enabled=false; nothing to smoke.")
    spec = sample_spec_from_config(cfg)
    fps = float(getattr(cfg, "fps", 30) or 30)
    history_offsets = cfg.policy.memory.history_offsets(fps)
    logger.info("Sample contract: %s", spec.fingerprint())

    main_dataset, rebot = _rebot_buffers(cfg, history_offsets)
    if cfg.diverse.probe_rebot_caches:
        probe_rebot_caches(
            [buffer.cache_dir for buffer in rebot],
            history_offsets_frames=cfg.policy.memory.history_offsets_frames(fps),
            depth_role=spec.depth_role,
        )
    diverse = build_diverse_buffer(
        cfg, cfg.diverse, main_dataset=main_dataset, device="cpu", seed=int(cfg.seed or 0)
    )
    groups = build_mixture_groups(
        cfg.diverse,
        align_rebot_buffers(rebot, cfg.diverse, spec),
        diverse,
        rebot_weights=[source.weight for source in __import__(
            "lerobot.rl.offline_dataset_utils", fromlist=["get_offline_dataset_sources"]
        ).get_offline_dataset_sources(cfg)],
    )
    logger.info("Batch allocation: %s", allocate_group_quotas(cfg.batch_size, groups))

    preprocessor = None
    if not args.skip_preprocessor:
        from lerobot.rl.rl_trainer import Trainer

        preprocessor, _ = Trainer.for_config(cfg).make_processors(cfg, dataset=main_dataset)

    telemetry = MixtureTelemetry()
    speed_by_half: dict[str, Counter] = {"rebot": Counter(), "diverse": Counter()}
    # Per source name; -1 = unlabelled (clause omitted).
    channel_by_source: dict[str, dict[str, Counter]] = {"precision": {}, "contact": {}}
    iterator = make_hierarchical_offline_iterator(
        groups,
        batch_size=cfg.batch_size,
        async_prefetch=False,
        action_chunk_size=cfg.policy.n_action_steps,
    )
    for step in range(args.batches):
        batch = next(iterator)
        telemetry.observe(batch, depth_key=f"depth.{spec.depth_role}.depth")
        info = batch["complementary_info"]
        widths = (~info["action_dim_is_pad"]).sum(dim=1)
        finite = all(
            bool(torch.isfinite(value).all())
            for value in (batch[ACTION], batch["state"][OBS_STATE])
        )
        logger.info(
            "batch %d: action %s widths %s finite=%s cameras/sample %s depth %d/%d",
            step,
            tuple(batch[ACTION].shape),
            sorted(set(widths.tolist())),
            finite,
            sorted(set(info["camera_is_present"].sum(dim=1).tolist())),
            int(info[f"depth.{spec.depth_role}.depth_is_present"].sum()),
            int(batch[ACTION].shape[0]),
        )
        speeds = info["metadata_speed"].reshape(-1).tolist()
        sources = info["source_id"].reshape(-1).tolist()
        if any(not 1 <= int(v) <= 5 for v in speeds):
            raise ValueError(f"metadata_speed outside 1-5 in batch {step}: {sorted(set(speeds))}")
        for value, source_id in zip(speeds, sources, strict=True):
            speed_by_half["rebot" if int(source_id) == REBOT_SOURCE_ID else "diverse"][int(value)] += 1
        for name, key, valid in (
            ("precision", "metadata_precision", range(1, 6)),
            ("contact", "metadata_contact", range(len(CONTACT_VOCAB))),
        ):
            values = info[key].reshape(-1).tolist() if key in info else [-1] * len(sources)
            if any(int(v) != -1 and int(v) not in valid for v in values):
                raise ValueError(f"{key} outside {{-1}} + {valid} in batch {step}: {sorted(set(values))}")
            for value, source_id in zip(values, sources, strict=True):
                source_name = SOURCE_NAMES.get(int(source_id), str(int(source_id)))
                channel_by_source[name].setdefault(source_name, Counter())[int(value)] += 1
        if preprocessor is not None and step == 0:
            from lerobot.rl.rl_trainer import Trainer  # noqa: F401 (kept local)

            trainer = Trainer.for_config(cfg)
            observations = trainer._inject_depth_observations(
                dict(batch["state"]), info, cfg
            )
            packed = trainer.build_training_batch(
                raw_batch=batch,
                observations=observations,
                actions=batch[ACTION],
                preprocessor=preprocessor,
                dataset=main_dataset,
                cfg=cfg,
            )
            logger.info(
                "  preprocessed: input_ids %s attention %s labels %s",
                tuple(packed["input_ids"].shape),
                tuple(packed["attention_mask"].shape),
                tuple(packed["labels"].shape) if "labels" in packed else None,
            )
            attended = packed["attention_mask"].sum(dim=1)
            logger.info("  attended tokens per sample: min %d max %d", int(attended.min()), int(attended.max()))
            checked = check_metadata_prompts(packed, info, trainer._pack_step(preprocessor), diverse.rows)
            logger.info("  metadata clauses match their columns on every prompt: %s", dict(checked))

    logger.info("\n%s", telemetry.describe(SOURCE_NAMES))
    logger.info("Speed label shares by half: %s", {half: dict(sorted(c.items())) for half, c in speed_by_half.items()})
    for name, by_source in channel_by_source.items():
        logger.info(
            "%s label shares by source (-1 = no clause): %s",
            name.capitalize(),
            {source: dict(sorted(c.items())) for source, c in sorted(by_source.items())},
        )


if __name__ == "__main__":
    main()

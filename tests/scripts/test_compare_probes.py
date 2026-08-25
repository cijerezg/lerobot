import json
from pathlib import Path

import pytest

from lerobot.scripts.compare_probes import (
    _alignment_keys,
    _interactive_samples,
    build_compare_index,
    resolve_compare_sources,
)


def _write_run(
    root: Path,
    name: str,
    steps: tuple[int, ...] = (0,),
    examples: tuple[str, ...] = (),
) -> Path:
    run_dir = root / name
    for step in steps:
        probe_dir = run_dir / "validation" / f"step_{step:08d}" / "attention"
        probe_dir.mkdir(parents=True)
        (probe_dir / "ep0000_L00").mkdir()
        (probe_dir / "ep0000_L00" / "overlay_img_top_summary.png").write_bytes(b"not-a-real-png")
        if examples:
            (probe_dir / "examples").mkdir()
        for example in examples:
            (probe_dir / "examples" / example).write_bytes(b"not-a-real-png")
        manifest = {
            "schema": 1,
            "id": "attention",
            "title": "Attention Maps",
            "group": "Attention",
            "claim": "Where does the model look?",
            "doc": "Test manifest.",
            "status": "info",
            "metrics": [
                {
                    "key": "mass",
                    "label": "Attention mass",
                    "value": 0.5 + step,
                    "good": "high",
                    "fmt": 2,
                    "note": "",
                    "baseline": None,
                    "warn": None,
                    "bad": None,
                    "refs": [],
                    "primary": True,
                    "status": "info",
                }
            ],
            "panels": [
                {
                    "file": "ep0000_L00/overlay_img_top_summary.png",
                    "caption": "Episode 0, layer 0",
                    "how": "Read the overlay.",
                    "kind": "image",
                    "primary": True,
                    "refs": [],
                }
            ]
            + [
                {
                    "file": f"examples/{example}",
                    "caption": "Percentile example",
                    "how": "",
                    "kind": "image",
                    "primary": False,
                    "refs": [],
                }
                for example in examples
            ],
            "see_also": [],
            "extra": {},
        }
        (probe_dir / "index.json").write_text(json.dumps(manifest))
    return run_dir


def test_resolve_compare_sources_retains_explicit_step_and_labels(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "experiment", steps=(400, 800))

    sources = resolve_compare_sources(
        [run_dir / "validation" / "step_00000400", run_dir],
        labels=["early", "whole run"],
    )

    assert [source.label for source in sources] == ["early", "whole run"]
    assert sources[0].initial_step == 400
    assert sources[1].initial_step is None
    assert sources[0].val_dir == run_dir / "validation"


@pytest.mark.parametrize("count", [0, 5])
def test_resolve_compare_sources_enforces_one_to_four_columns(tmp_path: Path, count: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 4"):
        resolve_compare_sources([tmp_path] * count)


def test_build_compare_index_namespaces_assets_and_chooses_initial_steps(tmp_path: Path) -> None:
    first = _write_run(tmp_path, "first", steps=(0, 10))
    second = _write_run(tmp_path, "second", steps=(5,))
    sources = resolve_compare_sources(
        [first / "validation" / "step_00000000", second],
        labels=["baseline", "candidate"],
    )

    index = build_compare_index(sources)

    assert len(index["runs"]) == 2
    assert index["runs"][0]["run"]["label"] == "baseline"
    assert index["runs"][0]["run"]["initial_step"] == 0
    assert index["runs"][1]["run"]["initial_step"] == 5
    first_panel = index["runs"][0]["probes"][0]["panels"]["0"][0]
    second_panel = index["runs"][1]["probes"][0]["panels"]["5"][0]
    assert first_panel["url"].startswith("/asset/0/step_00000000/attention/")
    assert second_panel["url"].startswith("/asset/1/step_00000005/attention/")


def test_default_labels_distinguish_step_inputs(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "experiment", steps=(400, 800))

    sources = resolve_compare_sources(
        [
            run_dir / "validation" / "step_00000400",
            run_dir / "validation" / "step_00000800",
        ]
    )

    assert sources[0].label == "experiment · step_00000400"
    assert sources[1].label == "experiment · step_00000800"

def test_percentile_examples_align_on_their_band_across_runs() -> None:
    keys = _alignment_keys(
        [
            {"file": "temporal_attention.png"},
            {"file": "examples/frame_p10_ep0002_fr004248.png"},
            {"file": "examples/frame_p90_ep0003_fr000171.png"},
        ]
    )

    assert keys == {
        "temporal_attention.png": "temporal_attention.png",
        "examples/frame_p10_ep0002_fr004248.png": "examples/frame_p10.png",
        "examples/frame_p90_ep0003_fr000171.png": "examples/frame_p90.png",
    }


def test_declared_alignment_key_wins_over_the_derived_one() -> None:
    keys = _alignment_keys(
        [{"file": "examples/p10_top.png", "align": "examples/frame_p10.png"}]
    )

    assert keys == {"examples/p10_top.png": "examples/frame_p10.png"}


def test_several_examples_per_band_keep_their_exact_filenames() -> None:
    # Two frames share the "hurt" band, so the band names no single figure and cannot
    # order the columns; both fall back to the filename and stay unaligned.
    files = [
        "examples/hurt_ep0001_fr000114.png",
        "examples/hurt_ep0002_fr002613.png",
        "examples/helped_ep0000_fr002754.png",
    ]

    keys = _alignment_keys([{"file": file} for file in files])

    assert keys["examples/hurt_ep0001_fr000114.png"] == "examples/hurt_ep0001_fr000114.png"
    assert keys["examples/hurt_ep0002_fr002613.png"] == "examples/hurt_ep0002_fr002613.png"
    assert keys["examples/helped_ep0000_fr002754.png"] == "examples/helped.png"


def test_build_compare_index_labels_every_panel_with_its_alignment_key(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "experiment", examples=("frame_p10_ep0002_fr004248.png",))

    index = build_compare_index(resolve_compare_sources([run_dir]))

    panels = {panel["file"]: panel["align"] for panel in index["runs"][0]["probes"][0]["panels"]["0"]}
    assert panels == {
        "ep0000_L00/overlay_img_top_summary.png": "ep0000_L00/overlay_img_top_summary.png",
        "examples/frame_p10_ep0002_fr004248.png": "examples/frame_p10.png",
    }


def test_interactive_samples_expose_stable_frame_keys(tmp_path: Path) -> None:
    action_dir = tmp_path / "action_trace"
    action_dir.mkdir()
    (action_dir / "metrics.csv").write_text(
        "episode,frame,global_idx,path_mse\n0,0,0,0.1\n1,240,500,0.2\n"
    )

    alignment, samples = _interactive_samples(
        action_dir, "action_trace", "action_trace.html"
    )

    assert alignment == "frame"
    assert [sample["key"] for sample in samples] == ["0:0", "1:240"]
    assert samples[1]["label"] == "episode 1 · frame 240"

    objective_dir = tmp_path / "objective"
    objective_dir.mkdir()
    (objective_dir / "objective.json").write_text(
        json.dumps(
            {
                "exemplars": [
                    {
                        "percentile": 5,
                        "label": "best fit",
                        "frames": [
                            {
                                "episode_idx": 3,
                                "frame_idx": 744,
                                "global_idx": 14765,
                            }
                        ],
                    }
                ]
            }
        )
    )

    alignment, samples = _interactive_samples(
        objective_dir, "objective", "action_exemplars.html"
    )

    assert alignment == "rank"
    assert samples == [
        {
            "key": "3:744",
            "label": "p5 best fit · episode 3 · frame 744",
            "episode": 3,
            "frame": 744,
            "global_idx": 14765,
        }
    ]

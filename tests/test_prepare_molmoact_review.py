"""Finalizing a partial review must not turn unreviewed nominees into rejections."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'examples/dataset/diverse_robot_dataset/prepare_molmoact.py'
spec = importlib.util.spec_from_file_location('prepare_molmoact_review_test', SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


@pytest.mark.parametrize('episodes,match', [
    ({'1': {'accept': True, 'quality': 5}}, 'missing='),
    ({'1': {'accept': False}, '2': {'accept': False}, '99': {'accept': False}}, 'extra='),
    ({'1': {'accept': False}, '2': {'accept': 'false'}}, 'explicit boolean'),
])
def test_invalid_coverage_leaves_existing_selection_untouched(tmp_path, episodes, match):
    candidates = tmp_path / 'candidates.json'
    candidates.write_text(json.dumps({'candidates': [{'episode_index': 1}, {'episode_index': 2}]}))
    verdicts = tmp_path / 'verdicts.json'
    verdicts.write_text(json.dumps({'episodes': episodes}))
    selection = tmp_path / 'selection.json'
    selection.write_text('{"previously_accepted": [7]}\n')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    # Metadata intentionally absent: coverage must be checked before metadata or writes.
    paths = SimpleNamespace(candidates=candidates, review_root=tmp_path, metadata_root=tmp_path/'missing_metadata')
    with pytest.raises(ValueError, match=match):
        prepare.finalize(paths, verdicts, 'review test')
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before

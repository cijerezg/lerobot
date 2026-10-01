"""Suite integration: one shared capture, plot-only reuse, and discoverable reports."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.configs.train import ProbeConfig
from lerobot.probes import conditions_matrix as cm
from lerobot.probes import subspace_spans as spans


def write_cache(directory):
    directory = Path(directory) / 'cache'
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for robot in ('rebot', 'droid'):
        for frame in range(10):
            rows.append(dict(row=len(rows), robot=robot, episode=f'{robot}/{frame}',
                             frame=frame, object_class='cup', phase='grasp',
                             text='real', holdout=frame >= 8))
    values = np.random.default_rng(42).normal(size=(len(rows), 3, 12)).astype(np.float16)
    np.save(directory / 'img_external_0.npy', values)
    np.save(directory / 'img_external_0.present.npy', np.ones(len(rows), dtype=bool))
    (directory / 'meta.json').write_text(json.dumps(dict(
        protocol=cm.PROTOCOL, rows=rows, groups={'img_external_0': [3, 12]})))


@pytest.mark.parametrize('mode,shared,captures,reports', [
    ('all', True, 0, True),
    ('all', False, 1, True),
    ('plot', False, 0, True),
    ('collect', False, 1, False),
    ('collect', True, 0, False),
])
def test_suite_cache_lifecycle(tmp_path, monkeypatch, mode, shared, captures, reports):
    cfg = SimpleNamespace(probe_parameters=ProbeConfig(
        mode=mode, enable_conditions_matrix=shared, enable_subspace_spans=True,
        subspace_layers='0,1,2', subspace_n_null=2, subspace_n_pivots=2))
    calls = []

    def collect(adapter, dataset, config, output):
        calls.append(output)
        write_cache(output)

    monkeypatch.setattr(cm, 'collect_cache', collect)
    if shared or mode == 'plot':
        write_cache(tmp_path / 'conditions_matrix')
    output = tmp_path / 'subspace_spans'
    result = spans.run(None, None, cfg, str(output))
    assert len(calls) == captures
    assert (output / 'index.json').exists() == reports
    if reports:
        index = json.loads((output / 'index.json').read_text())
        assert index['id'] == 'subspace_spans'
        assert any(p['file'] == 'explorer.html' and p['primary'] for p in index['panels'])
        assert result['layers'] == [0, 1, 2]
        assert result['headline_tau'] == .01
        assert (output / 'pairs.csv').is_file()
    else:
        assert result is None


def test_missing_shared_cache_does_not_silently_recollect(tmp_path, monkeypatch):
    cfg = SimpleNamespace(probe_parameters=ProbeConfig(enable_conditions_matrix=True))
    monkeypatch.setattr(cm, 'collect_cache', lambda *args: pytest.fail('unexpected capture'))
    with pytest.raises(FileNotFoundError):
        spans.run(None, None, cfg, str(tmp_path / 'subspace_spans'))

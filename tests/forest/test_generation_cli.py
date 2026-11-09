import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import numpy as np

from forest_generator import __main__ as cli
from forest_generator import maps
from forest_generator.generator import DEFAULT_CONFIG, forest_scene_id, layout_config
from omni_drones.navigation.forest.scene import load_scene, recorded_scene_splits, scene_split


@pytest.mark.parametrize('split', [None, 'train', 'val', 'test'])
@pytest.mark.parametrize('count', [31, 51])
def test_generate_keeps_every_requested_seed(monkeypatch, tmp_path, split, count):
    calls = []

    def build(root, seed, config, task, flight, difficulty, variant, reverse=False, split=None):
        calls.append((seed, task, reverse, difficulty, config['forest_size'][0], split))
        return Path(root)/task/str(seed)

    monkeypatch.setattr(cli, 'build_map', build)
    args = ['forest_generator', 'generate', '--seeds', *map(str, range(count)),
            '--difficulty', 'hard', '--output', str(tmp_path)]
    if split is not None:
        args += ['--split', split]
    monkeypatch.setattr(sys, 'argv', args)
    cli.main()

    assert {call[0] for call in calls} == set(range(count))
    assert Counter(call[1] for call in calls) == {
        'lateral': count, 'vertical': count, 'choice': count, 'mixed': count}
    assert not any(call[2] for call in calls)
    assert all(level == 'hard' and length == 36 and assigned == (split or 'train')
               for _, _, _, level, length, assigned in calls)
    report, = (tmp_path/'hard').glob('generation_*.json')
    records = json.loads(report.read_text())
    assert len(records) == 4*count
    assert all(row['split'] == (split or 'train') for row in records)
    assert report.name.endswith(f'_{split or "train"}.json')


@pytest.mark.parametrize('task', ['mixed', 'vertical'])
@pytest.mark.parametrize('seeds', [[0], [0, 50], list(range(30))])
def test_task_generates_one_map_per_seed(monkeypatch, tmp_path, task, seeds):
    calls = []
    def build(root, seed, config, task, flight, difficulty, variant, reverse=False, split=None):
        calls.append((seed, task, reverse, split))
        return Path(root)/task/str(seed)
    monkeypatch.setattr(cli, 'build_map', build)
    monkeypatch.setattr(sys, 'argv', ['forest_generator', 'generate', '--seeds', *map(str, seeds),
        '--tasks', task, '--difficulty', 'hard', '--split', 'train', '--output', str(tmp_path)])
    cli.main()
    assert calls == [(seed, task, False, 'train') for seed in seeds]
    report, = (tmp_path/'hard').glob('generation_*.json')
    assert len(json.loads(report.read_text())) == len(seeds)


@pytest.mark.parametrize('split', ['train', 'val', 'test'])
def test_export_index_and_scene_load_preserve_assignment(tmp_path, split):
    scene_id = next(str(i) for i in range(100) if scene_split([str(i)])[str(i)] != split)
    info = dict(scene_id=scene_id, task='mixed', split=split, validation=dict(passed=True))
    folder = tmp_path/'hard'/'mixed'/'000000'
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    maps.export(folder, info, vertices, np.array([[0, 1, 2]]))
    maps.write_index(tmp_path)
    assert json.loads((folder/'map.json').read_text())['split'] == split
    assert json.loads((tmp_path/'index.json').read_text())[0]['split'] == split
    loaded, _, _, _ = load_scene(0, tmp_path, split=split)
    assert loaded['split'] == split and loaded['scene_id'] == scene_id


def test_existing_forest_cannot_move_between_splits(tmp_path, monkeypatch):
    config = layout_config(json.loads(DEFAULT_CONFIG.read_text()), 'short')
    scene_id = forest_scene_id(0, config)
    folder = tmp_path/'hard'/'mixed'/'000000'
    folder.mkdir(parents=True)
    manifest = folder/'map.json'
    manifest.write_text(json.dumps(dict(scene_id=scene_id, split='train')))
    original = manifest.read_bytes()
    monkeypatch.setattr(maps, 'sample_forest', lambda *args: pytest.fail('split conflict must precede geometry generation'))
    with pytest.raises(ValueError, match='conflicting splits'):
        maps.build_map(tmp_path, 0, config, 'vertical', {}, 'hard', split='test')
    assert manifest.read_bytes() == original
    assert not (tmp_path/'hard'/'vertical').exists()


def test_recorded_splits_group_variants_and_keep_legacy_fallback():
    scenes = [dict(scene_id='forest', split='train'), dict(scene_id='forest'), dict(scene_id='legacy')]
    assert recorded_scene_splits(scenes) == dict(forest='train', **scene_split(['legacy']))
    assert recorded_scene_splits(reversed(scenes)) == recorded_scene_splits(scenes)
    with pytest.raises(ValueError, match='invalid split'):
        recorded_scene_splits([dict(scene_id='forest', split='typo')])

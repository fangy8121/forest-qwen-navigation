"""Exported pilot maps retain provenance and separate preview from runnable tasks."""
import hashlib
import json
import numpy as np
import pytest
from forest_generator.executor_pilot import build_executor_maps, diagnostic_scene
from forest_generator.generator import DEFAULT_CONFIG
from omni_drones.navigation.forest.scene import load_scene


def test_export_matches_physics_geometry_and_keeps_native_preview_unvalidated(tmp_path):
    root = build_executor_maps(tmp_path/'maps', seeds=(202610240,))
    info, vertices, faces, _ = load_scene(0, root, 'diagnostic/000')
    _, expected_v, expected_f, _ = diagnostic_scene()
    np.testing.assert_array_equal(vertices, expected_v)
    np.testing.assert_array_equal(faces, expected_f)
    assert info['validation']['passed']
    assert info['validation']['direct_bypass_blocked']
    assert 'not flight validation' in info['validation']['scope']
    native = json.loads((root/'native/202610240/map.json').read_text())
    assert native['branch_count'] > 0
    assert not native['validation']['passed']
    original = json.loads(DEFAULT_CONFIG.read_text())
    changed = {k for k in original if original[k] != native['parameters']['config'][k]}
    assert changed == {'forest_size'}
    with pytest.raises(ValueError, match='not a validated flight task'):
        load_scene(0, root, 'native/202610240')
    manifest = json.loads((root/'manifest.json').read_text())
    assert all(hashlib.sha256((root/p).read_bytes()).hexdigest() == digest
               for p, digest in manifest['files_sha256'].items())
    with pytest.raises(FileExistsError):
        build_executor_maps(root, seeds=())

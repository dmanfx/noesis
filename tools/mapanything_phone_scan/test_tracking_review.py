"""Focused admission checks plus the retained real mesh/frame source chain."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from .tracking_review import (_keys, _pose, _input, build_tracking_review,
                             build_high_detail_tracking_review, PcfTrackingReviewError)


def test_review_transform_preserves_metric_height_and_handedness():
    matrix = np.eye(4)
    matrix[:3, 3] = [2, .1905, -1]
    assert np.array_equal(_pose(matrix, "edge"), matrix)
    for bad in (np.diag([2, 2, 2, 1]), np.diag([-1, 1, 1, 1]),
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]):
        with pytest.raises(PcfTrackingReviewError):
            _pose(bad, "edge")


def test_recipe_unknown_fields_and_mutated_inputs_fail(tmp_path):
    with pytest.raises(PcfTrackingReviewError, match="unknown"):
        _keys({"known": 1, "extra": 1}, {"known"}, "recipe")
    file = tmp_path / 'source.json'
    file.write_text('{}')
    with pytest.raises(PcfTrackingReviewError, match="digest"):
        _input({"path":str(file),"sha256":"0"*64,"size_bytes":2}, tmp_path)


def test_retained_real_rgb_mesh_matches_current_live_frame_payloads():
    root = Path(__file__).resolve().parents[2]
    recipe = root / 'data/scene_priors/review/pcf-tracking-review-20261003/binding.json'
    if not recipe.is_file():
        pytest.skip("retained machine-private RGB mesh evidence is not installed")
    data = json.loads(recipe.read_text())
    selected = Path(data['source_descriptor']['path'])
    result = build_tracking_review(recipe,catalog_path=root/'data/scene_priors/catalog.json',
                                  current_descriptor=json.loads(selected.read_text()),
                                  current_descriptor_path=selected,repo_root=root)
    descriptor = root / 'data/scene_priors/review/pcf-tracking-review-20261003/descriptor.json'
    saved = json.loads(descriptor.read_text())
    assert result == saved
    for camera, row in saved['room_frames'].items():
        assert row['frame_binding']['world_frame']['revision'] == row['frame_binding']['scene_prior_id']
        assert len(row['assembly_from_world_col_major']) == 16
    assert saved['room_frames']['living-room']['registration_status'] == 'rejected'
    assert saved['accepted_for_canonical_use'] is False


def test_retained_high_detail_preserves_current_edges_and_rejects_source_drift(tmp_path):
    selected_high = os.environ.get('NOESIS_TEST_HIGH_DETAIL_MANIFEST')
    if not selected_high:
        pytest.skip('retained high-detail review artifact is not selected')
    root = Path(__file__).resolve().parents[2]
    recipe = root / 'data/scene_priors/review/pcf-tracking-review-20261003/binding.json'
    data = json.loads(recipe.read_text())
    selected_low = Path(data['source_descriptor']['path'])
    low = build_tracking_review(recipe, catalog_path=root/'data/scene_priors/catalog.json',
        current_descriptor=json.loads(selected_low.read_text()),
        current_descriptor_path=selected_low, repo_root=root)
    high, artifact = build_high_detail_tracking_review(Path(selected_high),
        recipe_path=recipe, low_descriptor=low, repo_root=root)
    assert high['contract_version'] == 2 and high['mesh_detail'] == 'high'
    assert high['room_frames'] == low['room_frames']
    assert high['accepted_for_canonical_use'] is False
    assert artifact.is_file() and high['artifact']['triangle_count'] > low['artifact']['triangle_count']
    mutated = json.loads(Path(selected_high).read_text())
    mutated['source']['low_glb_sha256'] = '0' * 64
    drift = tmp_path / 'tracking_high.json'
    drift.write_text(json.dumps(mutated))
    with pytest.raises(PcfTrackingReviewError, match='source chain'):
        build_high_detail_tracking_review(drift, recipe_path=recipe, low_descriptor=low, repo_root=root)
    mutated['source']['low_glb_sha256'] = low['artifact']['sha256']
    mutated['room_offset'] = [1, 0, 0]
    drift.write_text(json.dumps(mutated))
    with pytest.raises(PcfTrackingReviewError, match='unknown or missing'):
        build_high_detail_tracking_review(drift, recipe_path=recipe, low_descriptor=low, repo_root=root)

import hashlib
from pathlib import Path

from scripts.derive_authored_floor_level import derive


def test_parallel_floor_uses_measured_rise_and_does_not_move_shared_walls(tmp_path: Path):
    source = tmp_path / 'room.obj'
    source.write_text('v 0 0 0\nv 2 0 0\nv 0 0 2\ng wall\nf 1 2 3\nv 1 -12 1\ng upper\nf 1 2 3\nf 1 4 2\n')
    room_map = {'authored_scene_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'rooms': {'Upper': ['upper']}}
    surface = {'surface_id': 'measured', 'revision': 'r1', 'normal_world': [0, 1, 0], 'offset_m': -0.1905, 'extent_xz': [[0, 0], [2, 0], [0, 2]]}
    text, evidence = derive(source, room_map, 'Upper', surface, 100,
                            lower_world_offset_m=0.0, authored_lower_y=0.0)
    assert 'g wall\nf 1 2 3' in text
    assert 'g upper\nf 5 6 7\nf 5 4 6' in text
    assert 'v 0 19.05 0' in text
    assert 'v 1 -12 1' in text
    assert evidence['authored_upper_y'] == 19.05
    assert evidence['duplicated_top_vertices'] == 3


def test_floor_rise_is_relative_to_explicit_nonzero_lower_plane(tmp_path: Path):
    source = tmp_path / 'room.obj'
    source.write_text('v 0 30 0\nv 2 30 0\nv 0 30 2\ng upper\nf 1 2 3\n')
    room_map = {'authored_scene_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'rooms': {'Upper': ['upper']}}
    surface = {'surface_id':'measured', 'revision':'r1', 'normal_world':[0,1,0],
               'offset_m':-1.3905, 'extent_xz':[[0,0],[2,0],[0,2]]}
    _, evidence = derive(source, room_map, 'Upper', surface, 100,
                          lower_world_offset_m=-1.2, authored_lower_y=30.0)
    assert abs(evidence['authored_upper_y'] - 49.05) < 1e-10

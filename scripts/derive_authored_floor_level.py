#!/usr/bin/env python3
"""Derive a parallel authored floor from bound measured surface evidence.

Only explicitly mapped floor-group vertices change. Shared wall/furniture
vertices are duplicated rather than moved. No person tracks are inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path


def derive(source: Path, room_map: dict, room: str, surface: dict, scale: float, *,
           lower_world_offset_m: float, authored_lower_y: float) -> tuple[str, dict]:
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if room_map['authored_scene_sha256'] != digest:
        raise ValueError('room map does not bind the exact authored artifact')
    groups = room_map['rooms'][room]
    if not groups or not scale > 0:
        raise ValueError('floor group or scene scale missing')
    normal = surface['normal_world']
    if abs(normal[0]) > 1e-8 or abs(normal[2]) > 1e-8 or abs(normal[1] - 1) > 1e-8:
        raise ValueError('this derivation requires a measured parallel horizontal surface')
    rise = lower_world_offset_m - surface['offset_m']
    height = authored_lower_y + rise * scale
    if not all(math.isfinite(v) for v in (height, rise, scale)) or not rise > 0:
        raise ValueError('upper plane must be above lower authored floor')
    rows = raw.decode('utf8').splitlines()
    vertices = [list(map(float, row.split()[1:4])) for row in rows if row.startswith('v ')]
    if any(int(ref.split('/')[0]) <= 0 for row in rows if row.startswith('f ') for ref in row.split()[1:]):
        raise ValueError('relative vertex references are not admitted by this adapter')
    selected = set()
    group = ''
    for row in rows:
        if row.startswith('g '):
            group = row[2:].strip()
        if group in groups and row.startswith('f '):
            for ref in row.split()[1:]:
                index = int(ref.split('/')[0])
                if index <= 0:
                    raise ValueError('negative face indices are not admitted')
                selected.add(index)
    if not selected:
        raise ValueError('bound authored floor group has no faces')
    top = max(vertices[i - 1][1] for i in selected)
    mapping = {}
    added = []
    for index in sorted(selected):
        x, y, z = vertices[index - 1]
        # Keep the underside; extending vertical sides produces the measured
        # riser instead of a detached raised slab or a fabricated slope.
        if abs(y - top) <= 1e-8:
            mapping[index] = len(vertices) + len(added) + 1
            added.append(f'v {x:.12g} {height:.12g} {z:.12g}')
    # Normalize the interleaved OBJ's absolute vertex declarations before its
    # first faces. Original indices remain exact; derived indices append only.
    result = [row for row in rows if row.startswith('v ')] + added
    group = ''
    for line_index, row in enumerate(rows):
        if row.startswith('v '):
            continue
        if row.startswith('g '):
            group = row[2:].strip()
        if group in groups and row.startswith('f '):
            refs = []
            for ref in row.split()[1:]:
                components = ref.split('/')
                components[0] = str(mapping.get(int(components[0]), int(components[0])))
                refs.append('/'.join(components))
            row = 'f ' + ' '.join(refs)
        result.append(row)
    content = '\n'.join(result) + '\n'
    return content, {
        'contract': 'noesis.authored_scene.measured_floor_derivation', 'contract_version': 1,
        'parent_sha256': digest, 'derived_sha256': hashlib.sha256(content.encode()).hexdigest(),
        'room': room, 'groups': groups, 'surface_id': surface['surface_id'], 'surface_revision': surface['revision'],
        'normal_world': normal, 'offset_m': surface['offset_m'], 'extent_xz': surface['extent_xz'],
        'scene_per_m': scale, 'lower_world_offset_m': lower_world_offset_m,
        'authored_lower_y': authored_lower_y, 'authored_previous_upper_y': top,
        'authored_upper_y': height,
        'duplicated_top_vertices': len(added),
        'limitation': 'Authored group provides presentation extent; measured PCF patch provides height. No new global home floor survey is implied.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authored-scene', type=Path, required=True)
    parser.add_argument('--room-map', type=Path, required=True)
    parser.add_argument('--room', required=True)
    parser.add_argument('--lower-room', required=True, help='Actual authored room containing the admitted lower reference floor')
    parser.add_argument('--surfaces', type=Path, required=True)
    parser.add_argument('--surface-id', required=True)
    parser.add_argument('--calibration-bundle', type=Path, required=True)
    parser.add_argument('--structural-review', type=Path, help='Use the replayed fitted scale for a review-only floor preview, not a guessed scale')
    parser.add_argument('--parent-scene-payload', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bundle = json.loads(args.calibration_bundle.read_text())
    surfaces = json.loads(args.surfaces.read_text())
    if surfaces['authority'] != 'measured_calibration':
        raise ValueError('only measured calibration surfaces can derive authored geometry')
    surface = next(s for s in surfaces['surfaces'] if s['surface_id'] == args.surface_id)
    binding = bundle['cameras']['frame_bindings'][surfaces['camera_id']]
    if (binding['world_frame']['revision'] != surfaces['world_frame_revision']
            or binding['transform_sha256'] != surfaces['world_transform_sha256']
            or binding['camera_calibration_sha256'] != surfaces['calibration_revision']):
        raise ValueError('surface and admitted camera binding differ')
    if args.structural_review:
        from noesis.calibration.structural_registration import validate_solve_result
        review = json.loads(args.structural_review.read_text())
        if (review.get('contract') != 'noesis.structural_registration.review'
                or review.get('contract_version') != 1 or review.get('mode') != 'solve'):
            raise ValueError('floor preview requires a structural solve review')
        candidate = validate_solve_result(review['rooms'][surfaces['camera_id']]['candidate'])
        if candidate['solver_status'] != 'accepted':
            raise ValueError('floor preview requires a passed central structural solve')
        scale = candidate['solution']['scene_per_m']
    else:
        presentation = bundle['world_frame_presentations'][surfaces['world_frame_revision']]
        if presentation['acceptance'] != 'accepted':
            raise ValueError('floor derivation requires an admitted target-frame presentation')
        matrix = presentation['world_to_scene_col_major']
        scale = math.sqrt(sum(float(v)**2 for v in matrix[:3]))
    from noesis.validation.authored_scene import AuthoredSceneGeometry
    room_map = json.loads(args.room_map.read_text())
    geometry = AuthoredSceneGeometry.from_obj(args.authored_scene)
    lower_groups = room_map['rooms'][args.lower_room]
    floor_y = [geometry.vertices[i][1] for item in geometry.walkable_triangles
               if item.authority_group in lower_groups for i in item.triangle.vertex_indices]
    if not floor_y or max(floor_y) - min(floor_y) > 1e-3:
        raise ValueError('lower authored reference must be one actual horizontal floor')
    lower_plane = binding['world_floor_plane']
    if lower_plane['normal'] != [0.0,1.0,0.0]:
        raise ValueError('floor derivation requires a bound gravity-aligned lower reference')
    content, report = derive(args.authored_scene, room_map, args.room, surface, scale,
                             lower_world_offset_m=lower_plane['offset_m'], authored_lower_y=floor_y[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write(content)
    if args.parent_scene_payload:
        parent = json.loads(args.parent_scene_payload.read_text())['release']
        if parent['authored_scene']['sha256'] != report['parent_sha256']:
            raise ValueError('dependency manifest does not bind the parent model')
        for dependency in parent['authored_scene_dependencies']:
            relative = Path(dependency['relative_path'])
            source = args.authored_scene.parent / relative.name
            if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != dependency['sha256']:
                raise ValueError('parent material dependency identity differs')
            destination = args.output.parent / relative.name
            if destination.exists():
                raise ValueError('derived dependency already exists')
            shutil.copy2(source, destination)
    args.output.with_suffix('.floor-evidence.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

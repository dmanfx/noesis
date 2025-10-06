import json
import math
import os
from typing import Dict, Any, Optional, Tuple


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        if not path:
            return None
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def load_intrinsics(path: str) -> Dict[str, Any]:
    """Load intrinsics.json from root. Returns a dict keyed by model name.

    Expected format (per model key):
    {
      "model": str,
      "resolution": [w,h],
      "intrinsics": { "fx": float, "fy": float, "cx": float, "cy": float, "distortion_coeffs": [k1,k2,p1,p2,k3] },
      "K3x3": [[fx,0,cx],[0,fy,cy],[0,0,1]]?,
      ...
    }
    """
    data = _read_json(path)
    if isinstance(data, dict):
        return data
    return {}


def load_alignment(path: str) -> Dict[str, Any]:
    """Load PLY→OBJ alignment JSON. Fallback to identity matrix and defaults."""
    out = {
        'matrix': [1.0, 0.0, 0.0, 0.0,
                   0.0, 1.0, 0.0, 0.0,
                   0.0, 0.0, 1.0, 0.0,
                   0.0, 0.0, 0.0, 1.0],
        'floor_y': 0.0,
        'units': {'s_obj_to_m': 1.0}
    }
    data = _read_json(path)
    if isinstance(data, dict):
        mat = data.get('matrix') or data.get('align', {}).get('matrix')
        if isinstance(mat, list) and len(mat) == 16:
            out['matrix'] = [float(x) for x in mat]
        fy = data.get('floor_y') or data.get('align', {}).get('floor_y')
        if isinstance(fy, (int, float)):
            out['floor_y'] = float(fy)
        units = data.get('units') or data.get('align', {}).get('units')
        if isinstance(units, dict):
            s = units.get('s_obj_to_m')
            if isinstance(s, (int, float)):
                out['units'] = {'s_obj_to_m': float(s)}
    return out


def load_extrinsics(path: str) -> Dict[str, Any]:
    """Load camera_calibration.json containing per-camera extrinsics and optional align info.

    Expected structure:
    {
      "align": {"matrix":[16], "floor_y": float, "units": {"s_obj_to_m": float}},
      "cameras": {"cameraId": {"E": [16 floats]}}
    }
    """
    data = _read_json(path)
    if not isinstance(data, dict):
        return {'align': {}, 'cameras': {}}
    out = {
        'align': data.get('align', {}) or {},
        'cameras': {}
    }
    cams = data.get('cameras') or {}
    if isinstance(cams, dict):
        for cam_id, entry in cams.items():
            if not isinstance(entry, dict):
                continue
            E = entry.get('E')
            if isinstance(E, list) and len(E) == 16:
                out['cameras'][cam_id] = {'E': [float(x) for x in E]}
    return out


def save_extrinsics(path: str, camera_id: str, E_col_major_16: list) -> bool:
    """Persist/update extrinsics for a camera in camera_calibration.json."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        current = _read_json(path) or {}
        if 'cameras' not in current or not isinstance(current['cameras'], dict):
            current['cameras'] = {}
        current['cameras'][camera_id] = {'E': [float(x) for x in list(E_col_major_16)]}
        with open(path, 'w') as f:
            json.dump(current, f, indent=2)
        return True
    except Exception:
        return False


def _resolution_from_spec(spec: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    if not isinstance(spec, dict):
        return None
    res = spec.get('resolution')
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        try:
            w = int(res[0]); h = int(res[1])
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
    width = spec.get('width'); height = spec.get('height')
    try:
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            w = int(width); h = int(height)
            if w > 0 and h > 0:
                return w, h
    except Exception:
        pass
    return None


def _derive_k_from_specs(spec: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    res = _resolution_from_spec(spec)
    if not res:
        return None
    width, height = res
    try:
        hfov = float(spec.get('hfov_deg')) if spec.get('hfov_deg') is not None else None
    except Exception:
        hfov = None
    try:
        vfov = float(spec.get('vfov_deg')) if spec.get('vfov_deg') is not None else None
    except Exception:
        vfov = None
    if hfov is None and vfov is None:
        return None
    if hfov is None and vfov is not None:
        # derive horizontal FOV from vertical and aspect ratio
        vfov_rad = math.radians(vfov)
        hfov = math.degrees(2.0 * math.atan(math.tan(vfov_rad * 0.5) * (width / max(1.0, float(height)))))
    hfov_rad = math.radians(float(hfov))
    if vfov is None:
        vfov = math.degrees(2.0 * math.atan(math.tan(hfov_rad * 0.5) * (height / max(1.0, float(width)))))
    vfov_rad = math.radians(float(vfov))
    try:
        fx = (width / 2.0) / math.tan(max(1e-6, hfov_rad / 2.0))
        fy = (height / 2.0) / math.tan(max(1e-6, vfov_rad / 2.0))
        cx = width / 2.0
        cy = height / 2.0
        return float(fx), float(fy), float(cx), float(cy)
    except Exception:
        return None


def _derive_k_from_intrinsics_model(model_key: Optional[str], intrinsics_models: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    if not model_key or model_key not in intrinsics_models:
        return None
    model = intrinsics_models.get(model_key)
    if not isinstance(model, dict):
        return None
    intr = model.get('intrinsics') or {}
    if isinstance(intr, dict) and all(k in intr for k in ('fx', 'fy', 'cx', 'cy')):
        try:
            return (float(intr['fx']), float(intr['fy']), float(intr['cx']), float(intr['cy']))
        except Exception:
            return None
    K3 = intr.get('K3x3') or model.get('K3x3')
    if isinstance(K3, list) and len(K3) == 9:
        try:
            fx = float(K3[0]); fy = float(K3[4]); cx = float(K3[2]); cy = float(K3[5])
            return fx, fy, cx, cy
        except Exception:
            return None
    return None


def assemble_calibration_bundle(
    camera_ids: list,
    intrinsics_models: Dict[str, Any],
    model_map: Dict[str, str],
    extrinsics_data: Dict[str, Any],
    align_data: Dict[str, Any],
    camera_specs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Assemble a runtime calibration bundle to broadcast to clients."""
    k_table: Dict[str, list] = {}
    e_table: Dict[str, list] = {}
    for cam_id in camera_ids:
        spec = (camera_specs or {}).get(cam_id) if isinstance(camera_specs, dict) else None
        k_tuple = _derive_k_from_specs(spec or {}) if spec else None
        if k_tuple is None:
            k_tuple = _derive_k_from_intrinsics_model(model_map.get(cam_id), intrinsics_models or {})
        if k_tuple is not None:
            fx, fy, cx, cy = k_tuple
            k_table[cam_id] = [fx, fy, cx, cy]

        ext_entry = (extrinsics_data.get('cameras') or {}).get(cam_id) if isinstance(extrinsics_data, dict) else None
        E = ext_entry.get('E') if isinstance(ext_entry, dict) else None
        if isinstance(E, list) and len(E) == 16:
            e_table[cam_id] = [float(x) for x in E]

    align_dict = align_data if isinstance(align_data, dict) else {}
    matrix_vals = align_dict.get('matrix') if isinstance(align_dict, dict) else None
    if isinstance(matrix_vals, list) and len(matrix_vals) == 16:
        align_matrix = [float(x) for x in matrix_vals]
    else:
        align_matrix = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
    units_node = align_dict.get('units') if isinstance(align_dict, dict) else None
    if isinstance(units_node, dict) and 's_obj_to_m' in units_node and isinstance(units_node['s_obj_to_m'], (int, float)):
        units_dict = {'s_obj_to_m': float(units_node['s_obj_to_m'])}
    else:
        units_dict = {'s_obj_to_m': 1.0}
    floor_y = align_dict.get('floor_y', 0.0) if isinstance(align_dict, dict) else 0.0

    bundle = {
        'align': {
            'matrix': align_matrix,
            'floor_y': floor_y,
            'units': units_dict,
        },
        'cameras': {
            'K': k_table,
            'E': e_table,
            'pose_confidence': {},
        },
        'meta': {
            'version': 2,
            'conventions': {'E': 'world→camera', 'handedness': 'RH', 'up': 'Y'},
        }
    }
    bundle['metric_scale'] = 1.0
    if isinstance(camera_specs, dict) and camera_specs:
        bundle['meta']['camera_specs'] = camera_specs
    return bundle


def save_alignment(path: str, align_data: Dict[str, Any]) -> bool:
    """Persist/update alignment data in ply_alignment.json."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        current = _read_json(path) or {'matrix': [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], 'floor_y': 0.0, 'units': {'s_obj_to_m': 1.0}}
        if isinstance(align_data, dict):
            if 'matrix' in align_data and isinstance(align_data['matrix'], list) and len(align_data['matrix']) == 16:
                current['matrix'] = [float(x) for x in align_data['matrix']]
            if 'floor_y' in align_data and isinstance(align_data['floor_y'], (int, float)):
                current['floor_y'] = float(align_data['floor_y'])
            if 'units' in align_data and isinstance(align_data['units'], dict):
                if 's_obj_to_m' in align_data['units'] and isinstance(align_data['units']['s_obj_to_m'], (int, float)):
                    current['units'] = {'s_obj_to_m': float(align_data['units']['s_obj_to_m'])}
        with open(path, 'w') as f:
            json.dump(current, f, indent=2)
        return True
    except Exception:
        return False

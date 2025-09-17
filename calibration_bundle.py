import json
import os
from typing import Dict, Any, Optional


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


def assemble_calibration_bundle(
    camera_ids: list,
    intrinsics_models: Dict[str, Any],
    model_map: Dict[str, str],
    extrinsics_data: Dict[str, Any],
    align_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Assemble a runtime calibration bundle to broadcast to clients."""
    cams: Dict[str, Any] = {}
    for cam_id in camera_ids:
        model_key = model_map.get(cam_id)
        intr = {}
        if model_key and model_key in intrinsics_models:
            m = intrinsics_models[model_key]
            if isinstance(m, dict):
                ii = m.get('intrinsics') or {}
                if all(k in ii for k in ('fx', 'fy', 'cx', 'cy')):
                    intr = {
                        'fx': float(ii['fx']), 'fy': float(ii['fy']),
                        'cx': float(ii['cx']), 'cy': float(ii['cy'])
                    }
                K3 = m.get('K3x3') or ii.get('K3x3')
                if isinstance(K3, list):
                    intr['K3x3'] = K3
                dist = ii.get('distortion_coeffs')
                if isinstance(dist, list):
                    intr['distortion'] = dist
        ext_entry = (extrinsics_data.get('cameras') or {}).get(cam_id) if isinstance(extrinsics_data, dict) else None
        E = ext_entry.get('E') if isinstance(ext_entry, dict) else None
        cams[cam_id] = {
            'intrinsics': intr,
            'extrinsics': {'E': E} if isinstance(E, list) and len(E) == 16 else {}
        }
    bundle = {
        'align': {
            'matrix': align_data.get('matrix'),
            'floor_y': align_data.get('floor_y', 0.0),
            'units': align_data.get('units', {'s_obj_to_m': 1.0}),
        },
        'cameras': cams,
        'meta': {
            'version': 1,
            'conventions': {'E': 'world→camera', 'handedness': 'RH', 'up': 'Y'}
        }
    }
    return bundle

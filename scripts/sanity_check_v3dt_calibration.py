#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def resolve_like_noesis(yaml_path: Path, raw: str) -> Path:
    value = str(raw or '').strip()
    if not value:
        return Path('')
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    base_dir = yaml_path.parent.resolve()
    repo_root = base_dir.parent
    if value.startswith(('config/', 'models/', 'pipelines/')):
        return (repo_root / candidate).resolve()
    return (base_dir / candidate).resolve()


def axis_map_from_spec(spec: str) -> Optional[np.ndarray]:
    s = str(spec or "").strip().lower()
    if not s or s == "xyz":
        return None

    tokens = [tok for tok in s.replace(",", " ").split() if tok]
    if len(tokens) == 1 and len(tokens[0]) == 3 and all(ch in "xyz" for ch in tokens[0]):
        tokens = list(tokens[0])

    if len(tokens) != 3:
        return None

    basis = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    used = set()
    cols: List[np.ndarray] = []
    for tok in tokens:
        sign = -1.0 if tok.startswith("-") else 1.0
        axis = tok.lstrip("+-")
        if axis not in basis or axis in used:
            return None
        used.add(axis)
        cols.append(sign * basis[axis])
    return np.stack(cols, axis=1)


def rigid_inverse_world_to_cam(E_wc_col_major: Sequence[float]) -> np.ndarray:
    E = np.array(E_wc_col_major, dtype=np.float64).reshape((4, 4), order="F")
    R = E[:3, :3]
    t = E[:3, 3]
    Einv = np.eye(4, dtype=np.float64)
    Einv[:3, :3] = R.T
    Einv[:3, 3] = -R.T @ t
    return Einv


def camera_center_from_E_world_to_cam(E_wc: np.ndarray) -> np.ndarray:
    R = E_wc[:3, :3]
    t = E_wc[:3, 3]
    return -R.T @ t


def scaled_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    target_w: int,
    target_h: int,
    base_res: Optional[Tuple[int, int]],
) -> Tuple[float, float, float, float]:
    if base_res:
        base_w = float(base_res[0])
        base_h = float(base_res[1])
    else:
        base_w = 2.0 * float(cx) if cx else 0.0
        base_h = 2.0 * float(cy) if cy else 0.0

    sx = float(target_w) / base_w if base_w else 1.0
    sy = float(target_h) / base_h if base_h else 1.0
    return fx * sx, fy * sy, cx * sx, cy * sy


def best_scale_match_error(A: np.ndarray, B: np.ndarray) -> Tuple[float, float]:
    """Return (relative_error, alpha) minimizing ||alpha*A - B|| in least squares."""
    a = A.reshape(-1)
    b = B.reshape(-1)
    denom = float(np.dot(a, a))
    if denom <= 1e-12:
        return float("inf"), 1.0
    alpha = float(np.dot(a, b) / denom)
    err = np.linalg.norm(alpha * a - b)
    rel = float(err / (np.linalg.norm(b) + 1e-12))
    return rel, alpha


def nullspace_camera_center(P: np.ndarray) -> Optional[np.ndarray]:
    """Return inhomogeneous camera center from P's right nullspace (3D)."""
    try:
        _, _, vh = np.linalg.svd(P)
        v = vh[-1, :]
        if abs(float(v[3])) < 1e-12:
            return None
        return (v[:3] / v[3]).astype(np.float64)
    except Exception:
        return None


@dataclass
class Candidate:
    invert_e: bool
    world_axes: str
    y_flip: bool
    world_scale: float
    rel_err: float
    alpha: float


def build_projection_candidate(
    K: np.ndarray,
    E_col_major: Sequence[float],
    target_h: int,
    invert_e: bool,
    world_axes: str,
    y_flip: bool,
    world_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (P, E_eff_wc). E_eff_wc is the world->camera used to build P."""
    E = np.array(E_col_major, dtype=np.float64).reshape((4, 4), order="F")
    if invert_e:
        E = rigid_inverse_world_to_cam(E_col_major)

    A = axis_map_from_spec(world_axes)
    if A is not None:
        A4 = np.eye(4, dtype=np.float64)
        A4[:3, :3] = A
        E = E @ A4

    E = E.copy()
    E[:3, 3] *= float(world_scale)

    P = K @ E[:3, :]

    if y_flip:
        P_orig = P.copy()
        P[1, :] = -P_orig[1, :] + float(target_h) * P_orig[2, :]

    return P, E


def parse_caminfo(path: Path) -> Tuple[np.ndarray, str]:
    data = load_yaml(path)
    if "projectionMatrix_3x4_w2p" in data:
        key = "projectionMatrix_3x4_w2p"
    elif "projectionMatrix_3x4" in data:
        key = "projectionMatrix_3x4"
    else:
        raise ValueError(f"Missing projectionMatrix in {path}")

    arr = data.get(key)
    if not isinstance(arr, list) or len(arr) != 12:
        raise ValueError(f"{path}: {key} must be 12 floats")

    P = np.array([float(x) for x in arr], dtype=np.float64).reshape((3, 4), order="C")
    return P, key


def find_camera_intrinsics(cameras_yaml: Dict[str, Any], camera_name: str) -> Tuple[Dict[str, Any], Optional[Tuple[int, int]]]:
    models = cameras_yaml.get("intrinsics_models") or {}
    cams = cameras_yaml.get("cameras") or {}

    model_name: Optional[str] = None
    for _, entry in (cams.items() if isinstance(cams, dict) else []):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("name")) == camera_name:
            model_name = str(entry.get("model") or entry.get("intrinsics_model") or "") or None
            break

    if not model_name:
        raise KeyError(f"Camera '{camera_name}' not found in cameras.yaml")

    model = models.get(model_name)
    if not isinstance(model, dict):
        raise KeyError(f"Intrinsics model '{model_name}' not found")

    intr = model.get("intrinsics") or {}
    fx = float(intr.get("fx"))
    fy = float(intr.get("fy"))
    cx = float(intr.get("cx"))
    cy = float(intr.get("cy"))

    base_res = None
    res = model.get("resolution")
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        base_res = (int(res[0]), int(res[1]))
    else:
        res2 = intr.get("resolution")
        if isinstance(res2, (list, tuple)) and len(res2) >= 2:
            base_res = (int(res2[0]), int(res2[1]))

    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "model": model_name}, base_res


def parse_tracker_caminfo_list(tracker_yaml: Dict[str, Any]) -> List[str]:
    omp = tracker_yaml.get("ObjectModelProjection") or {}
    cmf = omp.get("cameraModelFilepath")
    if not isinstance(cmf, list) or not cmf:
        raise KeyError("tracker yaml: ObjectModelProjection.cameraModelFilepath missing")
    out: List[str] = []
    for p in cmf:
        if not isinstance(p, str):
            continue
        out.append(p)
    if not out:
        raise KeyError("tracker yaml: cameraModelFilepath had no usable entries")
    return out


def name_from_caminfo_path(p: str) -> str:
    # expects camInfo_<name>.yml
    stem = Path(p).name
    if stem.startswith("camInfo_"):
        return stem[len("camInfo_") :].rsplit(".", 1)[0]
    # fallback
    return Path(p).stem


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanity-check SV3DT camInfo projection matrices")
    ap.add_argument("--pipeline-config", required=True, type=Path)
    ap.add_argument("--cameras-config", required=True, type=Path)
    ap.add_argument("--calibration", required=True, type=Path)
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    pipe_path = args.pipeline_config.resolve()
    cam_path = args.cameras_config.resolve()
    cal_path = args.calibration.resolve()

    pipe = load_yaml(pipe_path)
    streammux = pipe.get("streammux") or {}
    target_w = int(streammux.get("width", 1920) or 1920)
    target_h = int(streammux.get("height", 1080) or 1080)

    tracker_cfg = (pipe.get("tracker") or {}).get("config-file")
    if not tracker_cfg:
        print("FAIL: pipeline yaml missing tracker.config-file")
        return 2

    tracker_path = resolve_like_noesis(pipe_path, str(tracker_cfg))
    tracker = load_yaml(tracker_path)
    caminfo_list = parse_tracker_caminfo_list(tracker)

    cameras_yaml = load_yaml(cam_path)
    calib_root = load_json(cal_path)
    calib_cams = (calib_root.get("cameras") or {}) if isinstance(calib_root, dict) else {}

    print("=== SV3DT camInfo sanity check ===")
    print(f"pipeline:  {pipe_path}")
    print(f"tracker:   {tracker_path}")
    print(f"cameras:   {cam_path}")
    print(f"calib:     {cal_path}")
    print(f"streammux: {target_w}x{target_h}")
    print(f"cams:      {len(caminfo_list)}")
    print()

    failures = 0

    # Candidate search space.
    invert_opts = [False, True]
    axes_opts = ["xyz", "xzy"]
    yflip_opts = [False, True]
    scale_opts = [1.0, 100.0]

    for caminfo_rel in caminfo_list:
        cam_name = name_from_caminfo_path(caminfo_rel)
        caminfo_path = resolve_like_noesis(pipe_path, caminfo_rel)

        print(f"--- {cam_name} ---")
        if not caminfo_path.exists():
            print(f"FAIL: camInfo missing: {caminfo_path}")
            failures += 1
            if args.fail_fast:
                return 1
            continue

        try:
            P_caminfo, key = parse_caminfo(caminfo_path)
        except Exception as exc:
            print(f"FAIL: unable to parse camInfo: {exc}")
            failures += 1
            if args.fail_fast:
                return 1
            continue

        if not np.all(np.isfinite(P_caminfo)):
            print("FAIL: camInfo projection contains NaN/Inf")
            failures += 1
            if args.fail_fast:
                return 1
            continue

        try:
            intr, base_res = find_camera_intrinsics(cameras_yaml, cam_name)
        except Exception as exc:
            print(f"FAIL: intrinsics not found: {exc}")
            failures += 1
            if args.fail_fast:
                return 1
            continue

        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        fx_s, fy_s, cx_s, cy_s = scaled_intrinsics(fx, fy, cx, cy, target_w, target_h, base_res)

        # Determine whether camInfo expects w2p K (with principal point) or centered PP.
        use_w2p = key.endswith("_w2p")
        cx_use = float(cx_s) if use_w2p else 0.0
        cy_use = float(cy_s) if use_w2p else 0.0
        K = np.array([[float(fx_s), 0.0, cx_use], [0.0, float(fy_s), cy_use], [0.0, 0.0, 1.0]], dtype=np.float64)

        calib_entry = calib_cams.get(cam_name)
        if not isinstance(calib_entry, dict) or not isinstance(calib_entry.get("E"), list) or len(calib_entry.get("E")) != 16:
            print("FAIL: calibration missing E for this camera")
            failures += 1
            if args.fail_fast:
                return 1
            continue
        E_col = [float(x) for x in calib_entry["E"]]

        # Find best matching candidate.
        best: Optional[Candidate] = None
        best_P = None
        best_Eeff = None

        for inv in invert_opts:
            for axes in axes_opts:
                for yf in yflip_opts:
                    for sc in scale_opts:
                        P_cand, E_eff = build_projection_candidate(K, E_col, target_h, inv, axes, yf, sc)
                        rel, alpha = best_scale_match_error(P_cand, P_caminfo)
                        if best is None or rel < best.rel_err:
                            best = Candidate(inv, axes, yf, sc, rel, alpha)
                            best_P = P_cand
                            best_Eeff = E_eff

        assert best is not None and best_P is not None and best_Eeff is not None

        # Print core facts.
        print(f"camInfo: {caminfo_path}")
        print(f"matrix:  {key}  (assume w2p={use_w2p})")
        print(f"intrinsics model: {intr['model']}  base_res={base_res if base_res else 'inferred'}")
        print(f"K@{target_w}x{target_h}: fx={fx_s:.3f} fy={fy_s:.3f} cx={cx_use:.3f} cy={cy_use:.3f}")
        print(
            "best-match toggles: "
            f"invert_e={int(best.invert_e)} world_axes={best.world_axes} y_flip={int(best.y_flip)} world_scale={best.world_scale:g} "
            f"rel_err={best.rel_err:.3e}"
        )

        # Camera center consistency check.
        C_from_Eeff = camera_center_from_E_world_to_cam(best_Eeff)
        C_from_P = nullspace_camera_center(P_caminfo)

        if C_from_P is None:
            print("WARN: could not compute camera center from P nullspace")
        else:
            dist = float(np.linalg.norm(C_from_P - C_from_Eeff))
            print(f"camera_center(E_eff) = {C_from_Eeff.tolist()}")
            print(f"camera_center(P)     = {C_from_P.tolist()}")
            print(f"center mismatch |d|  = {dist:.6f} (in SV3DT world units)")

        # Decide pass/fail with actionable advice.
        ok = True
        actions: List[str] = []

        # P match quality
        if best.rel_err > 1e-3:
            ok = False
            actions.append(
                "Projection matrix in camInfo does not match calibration within tolerance. "
                "Likely wrong camInfo directory, wrong calibration JSON, or wrong cameras YAML. "
                "Fix: regenerate camInfo from the exact same (pipeline, cameras, calibration) triple, "
                "then point nvtracker_sv3dt.yml to that directory."
            )

        # Strongly flag inverted-E usage (common root cause)
        if best.invert_e:
            ok = False
            actions.append(
                "camInfo appears to have been generated with INVERT_E=1 (treating E as if it must be inverted). "
                "But your calibration JSON stores E as world->camera (col-major). For SV3DT, camInfo should use E directly. "
                "Fix: set NOESIS_V3DT_CAMINFO_INVERT_E=0, regenerate camInfo, and prevent ds8_runtime autogen from reintroducing it."
            )

        # World axes
        if best.world_axes != "xzy":
            # Not always wrong, but for your stated convention (Y up) SV3DT reference is Z-up.
            actions.append(
                "NOTE: best match used world_axes=xyz. In your Noesis convention (Y is up, X/Z are floor), "
                "SV3DT reference configs are Z-up. Common fix is world_axes=xzy (swap Y/Z) so SV3DT sees Z-up."
            )

        # Unit scale
        if abs(best.world_scale - 100.0) < 1e-9:
            actions.append(
                "NOTE: camInfo best match uses world_scale=100 (centimeters). Your nvtracker SV3DT config comments indicate it's tuned for METERS. "
                "If you want centimeters, multiply world-space variance params by 1e4; otherwise regenerate camInfo with world_scale=1."
            )

        if C_from_P is not None:
            if float(np.linalg.norm(C_from_P - C_from_Eeff)) > (0.10 * max(1.0, best.world_scale)):
                # Scale-aware threshold: 10cm (or 10 in cm mode)
                ok = False
                actions.append(
                    "Camera center derived from camInfo P does not match camera center implied by E. "
                    "This almost always means: wrong invert_e, wrong world_axes, or wrong y_flip when generating camInfo."
                )

        print("RESULT:", "PASS" if ok else "FAIL")
        if actions:
            print("WHY / WHAT TO DO:")
            for a in actions:
                print("  -", a)
        print()

        if not ok:
            failures += 1
            if args.fail_fast:
                return 1

    if failures:
        print(f"SUMMARY: {failures} camera(s) failed.")
        print(
            "Most common quick fix (if you see invert_e=1 above):\n"
            "  export NOESIS_V3DT_CAMINFO_INVERT_E=0\n"
            "  export NOESIS_V3DT_CAMINFO_WORLD_AXES=xzy\n"
            "  export NOESIS_V3DT_CAMINFO_Y_FLIP=1\n"
            "  export NOESIS_V3DT_CAMINFO_WORLD_SCALE=1\n"
            "  python3 scripts/generate_v3dt_caminfo.py --pipeline-config <...> --cameras-config <...> --calibration <...> --output-dir <...>\n"
            "  export NOESIS_V3DT_AUTOGEN_CAMINFO=0   # stop ds8_runtime from overwriting\n"
        )
        return 1

    print("SUMMARY: all cameras PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

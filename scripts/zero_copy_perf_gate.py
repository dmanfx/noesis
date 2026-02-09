#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    q = max(0.0, min(1.0, float(q)))
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = int(round(q * (len(ordered) - 1)))
    idx = max(0, min(len(ordered) - 1, idx))
    return float(ordered[idx])


def _parse_ws_port(ws_url: str) -> int:
    parsed = urlparse(ws_url)
    if parsed.port is not None:
        return int(parsed.port)
    return 443 if parsed.scheme == "wss" else 6008


def _read_gpu_memory_used_mib() -> Optional[float]:
    cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=2.0)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    values: List[float] = []
    for line in str(proc.stdout or "").splitlines():
        s = str(line).strip()
        if not s:
            continue
        try:
            values.append(float(s))
        except Exception:
            continue
    if not values:
        return None
    return float(max(values))


def _spawn_runtime(
    ws_port: int,
    pipeline_config: Path,
    cameras_config: Path,
    log_path: Path,
    *,
    stub: bool,
    skip_cuda_preflight: bool,
) -> subprocess.Popen[str]:
    cmd = [
        sys.executable,
        "noesis/ds8_runtime.py",
        "--pipeline-config",
        str(pipeline_config),
        "--cameras-config",
        str(cameras_config),
        "--disable-rest",
        "--ws-port",
        str(int(ws_port)),
        "--log-level",
        "INFO",
    ]
    env = dict(os.environ)
    if stub:
        env["NOESIS_DS8_STUB_PIPELINE"] = "1"
    if skip_cuda_preflight:
        env["NOESIS_SKIP_CUDA_PREFLIGHT"] = "1"
    env.setdefault("NOESIS_WS_PORT_FALLBACK_TRIES", "32")
    env.setdefault("NOESIS_WS_BIND_RETRY_TRIES", "4")

    log_fh = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _reserve_local_port() -> tuple[int, socket.socket]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return int(sock.getsockname()[1]), sock


def _terminate_process(proc: subprocess.Popen[str], timeout_s: float = 8.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    start = time.time()
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _tail(path: Path, lines: int = 60) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    arr = text.splitlines()
    return "\n".join(arr[-max(1, int(lines)):])


async def _collect_perf_metrics(
    ws_url: str,
    duration_s: float,
    startup_timeout_s: float,
    depth_camera: str,
    proc: Optional[subprocess.Popen[str]],
) -> Dict[str, Any]:
    import websockets

    ws = None
    deadline = time.time() + max(1.0, float(startup_timeout_s))
    last_err: Optional[str] = None
    while time.time() < deadline:
        try:
            ws = await websockets.connect(ws_url, max_size=None)
            break
        except Exception as exc:
            last_err = str(exc)
            await asyncio.sleep(0.25)

    if ws is None:
        return {"ok": False, "error": f"ws_connect_failed:{last_err or 'unknown'}", "samples": 0}

    samples = 0
    depth_req_sent = 0
    camera_hint: Optional[str] = None
    max_zero_copy_violations = 0
    max_boundary_p99 = None
    fps_samples: List[float] = []
    latency_p95_samples: List[float] = []
    boundary_p95_samples: List[float] = []
    gpu_mem_samples: List[float] = []
    pipeline_errors: List[str] = []
    sample_times_s: List[float] = []

    start = time.monotonic()
    end = start + max(1.0, float(duration_s))
    next_depth_req = start
    next_gpu_sample = start

    try:
        while time.monotonic() < end:
            if proc is not None:
                code = proc.poll()
                if code is not None:
                    return {
                        "ok": False,
                        "error": "runtime_exited_early",
                        "process_exit_code": int(code),
                        "samples": samples,
                    }

            now = time.monotonic()
            if now >= next_depth_req:
                try:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "get_ma_depth",
                                "camera": str(depth_camera or "").strip() or camera_hint or "camera_0",
                                "request_id": f"zero-copy-perf-{depth_req_sent + 1}",
                            },
                            separators=(",", ":"),
                        )
                    )
                    depth_req_sent += 1
                except Exception:
                    pass
                next_depth_req = now + 1.0

            if now >= next_gpu_sample:
                sample = await asyncio.to_thread(_read_gpu_memory_used_mib)
                if sample is not None:
                    gpu_mem_samples.append(float(sample))
                next_gpu_sample = now + 1.0

            timeout_s = min(1.0, max(0.05, end - time.monotonic()))
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                if "ConnectionClosed" in type(exc).__name__:
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    try:
                        ws = await websockets.connect(ws_url, max_size=None)
                        continue
                    except Exception as reconnect_exc:
                        return {
                            "ok": False,
                            "error": f"ws_reconnect_failed:{type(reconnect_exc).__name__}",
                            "samples": samples,
                        }
                return {"ok": False, "error": f"ws_recv_failed:{type(exc).__name__}", "samples": samples}

            if not isinstance(msg, str):
                continue
            try:
                payload = json.loads(msg)
            except Exception:
                continue

            if str(payload.get("type") or "") != "stats":
                continue

            stats = payload.get("payload") or {}
            pipe = stats.get("pipeline") or {}
            cameras = stats.get("cameras") or {}
            sample_times_s.append(time.monotonic())
            if camera_hint is None and isinstance(cameras, dict) and cameras:
                camera_hint = str(next(iter(cameras.keys())))

            samples += 1
            try:
                max_zero_copy_violations = max(max_zero_copy_violations, int(pipe.get("zero_copy_violations", 0) or 0))
            except Exception:
                pass
            try:
                p99 = pipe.get("boundary_cpu_serialization_p99_ms")
                if p99 is not None:
                    p99_val = float(p99)
                    if max_boundary_p99 is None or p99_val > float(max_boundary_p99):
                        max_boundary_p99 = p99_val
            except Exception:
                pass
            try:
                p95_val = pipe.get("boundary_cpu_serialization_p95_ms")
                if p95_val is not None:
                    boundary_p95_samples.append(float(p95_val))
            except Exception:
                pass

            for err in list(pipe.get("errors") or []):
                txt = str(err).strip()
                if txt:
                    pipeline_errors.append(txt)

            if isinstance(cameras, dict):
                for cam_payload in cameras.values():
                    if not isinstance(cam_payload, dict):
                        continue
                    try:
                        fps_val = float(cam_payload.get("fps"))
                    except Exception:
                        fps_val = None
                    if fps_val is not None and fps_val >= 0.0:
                        fps_samples.append(float(fps_val))

            latency_payload = pipe.get("latency_ms") if isinstance(pipe, dict) else None
            if isinstance(latency_payload, dict):
                try:
                    p95 = latency_payload.get("p95")
                    if p95 is not None:
                        latency_p95_samples.append(float(p95))
                except Exception:
                    pass

    finally:
        try:
            await ws.close()
        except Exception:
            pass

    if samples < 1:
        return {"ok": False, "error": "no_stats_samples", "samples": 0}

    fps_camera_samples = [float(v) for v in fps_samples if float(v) > 0.0]
    fps_p50 = _percentile(fps_camera_samples, 0.50)
    fps_source = "camera_fps"
    if fps_p50 is None:
        sample_hz: List[float] = []
        for idx in range(1, len(sample_times_s)):
            dt = float(sample_times_s[idx] - sample_times_s[idx - 1])
            if dt > 1e-6:
                sample_hz.append(1.0 / dt)
        fps_p50 = _percentile(sample_hz, 0.50)
        fps_source = "stats_rate_hz"

    latency_p95 = _percentile(latency_p95_samples, 0.95)
    latency_source = "pipeline_latency_ms_p95"
    if latency_p95 is None:
        latency_p95 = _percentile(boundary_p95_samples, 0.95)
        latency_source = "boundary_cpu_serialization_p95_ms"

    metrics = {
        "core_zero_copy_violations": int(max_zero_copy_violations),
        "boundary_cpu_serialization_p99_ms": float(max_boundary_p99) if max_boundary_p99 is not None else None,
        "fps_p50": fps_p50,
        "pipeline_latency_ms_p95": latency_p95,
        "gpu_memory_used_p95_mib": _percentile(gpu_mem_samples, 0.95),
    }
    return {
        "ok": True,
        "samples": samples,
        "depth_requests_sent": depth_req_sent,
        "metrics": metrics,
        "pipeline_errors": pipeline_errors[:16],
        "metric_sources": {
            "fps_p50": fps_source,
            "pipeline_latency_ms_p95": latency_source,
        },
        "raw_counts": {
            "fps_samples": len(fps_samples),
            "latency_p95_samples": len(latency_p95_samples),
            "boundary_p95_samples": len(boundary_p95_samples),
            "gpu_mem_samples": len(gpu_mem_samples),
        },
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-copy perf gate")
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--pipeline-config", type=Path, default=REPO_ROOT / "config" / "infer.yaml")
        sp.add_argument("--cameras-config", type=Path, default=REPO_ROOT / "config" / "cameras.yaml")
        sp.add_argument("--duration-s", type=float, default=900.0)
        sp.add_argument("--stats-ws", default="ws://127.0.0.1:6008")
        sp.add_argument("--startup-timeout", type=float, default=20.0)
        sp.add_argument(
            "--depth-camera",
            default="__zero_copy_probe__",
            help="Camera id used for WS depth-path traffic.",
        )
        sp.add_argument("--no-spawn", action="store_true")
        sp.add_argument("--stub", action="store_true")
        sp.add_argument("--skip-cuda-preflight", action="store_true")
        sp.add_argument("--log-path", type=Path, default=None)

    capture = sub.add_parser("capture-baseline", help="Capture baseline metrics")
    _add_common(capture)
    capture.add_argument("--out", type=Path, required=True)

    compare = sub.add_parser("compare", help="Compare candidate against a baseline")
    _add_common(compare)
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)

    return parser.parse_args()


def _run_once(args: argparse.Namespace) -> Dict[str, Any]:
    ws_url = str(args.stats_ws)
    ws_port = _parse_ws_port(ws_url)
    parsed_ws = urlparse(ws_url)
    ws_lock: Optional[socket.socket] = None

    if args.log_path is not None:
        log_path = Path(args.log_path)
    else:
        tmp = tempfile.NamedTemporaryFile(prefix="zero_copy_perf_", suffix=".log", delete=False)
        log_path = Path(tmp.name)
        tmp.close()

    proc: Optional[subprocess.Popen[str]] = None
    if not args.no_spawn:
        try:
            ws_port, ws_lock = _reserve_local_port()
        except Exception:
            if ws_lock is not None:
                ws_lock.close()
            return {
                "ok": False,
                "error": "port_reservation_failed",
                "stats_ws": ws_url,
                "log_path": str(log_path),
            }
        ws_url = f"{parsed_ws.scheme}://{parsed_ws.hostname or '127.0.0.1'}:{ws_port}"
        proc = _spawn_runtime(
            ws_port=ws_port,
            pipeline_config=Path(args.pipeline_config),
            cameras_config=Path(args.cameras_config),
            log_path=log_path,
            stub=bool(args.stub),
            skip_cuda_preflight=bool(args.skip_cuda_preflight),
        )
        if ws_lock is not None:
            ws_lock.close()

    started = time.monotonic()
    try:
        result = asyncio.run(
            _collect_perf_metrics(
                ws_url=ws_url,
                duration_s=float(args.duration_s),
                startup_timeout_s=float(args.startup_timeout),
                depth_camera=str(args.depth_camera),
                proc=proc,
            )
        )
    finally:
        if proc is not None:
            _terminate_process(proc)

    result["duration_s"] = round(max(0.0, time.monotonic() - started), 3)
    result["stats_ws"] = ws_url
    result["log_path"] = str(log_path)
    if not bool(result.get("ok", False)):
        tail = _tail(log_path)
        if tail:
            result["runtime_tail"] = tail
    return result


def _require_metric(metrics: Dict[str, Any], key: str) -> Optional[float]:
    raw = metrics.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def main() -> int:
    args = _parse_args()
    if args.command == "capture-baseline":
        capture = _run_once(args)
        payload = {
            "ok": bool(capture.get("ok", False)),
            "mode": "capture-baseline",
            "captured_at": int(time.time()),
            "pipeline_config": str(args.pipeline_config),
            "cameras_config": str(args.cameras_config),
            "duration_s": float(args.duration_s),
            "result": capture,
        }
        _write_json(Path(args.out), payload)
        print(json.dumps({"ok": payload["ok"], "mode": payload["mode"], "out": str(args.out)}, separators=(",", ":")))
        return 0 if payload["ok"] else 1

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(json.dumps({"ok": False, "error": "baseline_missing", "baseline": str(baseline_path)}, separators=(",", ":")))
        return 1

    try:
        baseline_doc = json.loads(baseline_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"baseline_parse_failed:{type(exc).__name__}"}, separators=(",", ":")))
        return 1

    baseline_result = baseline_doc.get("result") if isinstance(baseline_doc, dict) else None
    baseline_metrics = (baseline_result or {}).get("metrics") if isinstance(baseline_result, dict) else None
    baseline_sources = (baseline_result or {}).get("metric_sources") if isinstance(baseline_result, dict) else None
    if not isinstance(baseline_metrics, dict):
        print(json.dumps({"ok": False, "error": "baseline_metrics_missing", "baseline": str(baseline_path)}, separators=(",", ":")))
        return 1

    candidate = _run_once(args)
    candidate_metrics = candidate.get("metrics") if isinstance(candidate, dict) else None
    candidate_sources = candidate.get("metric_sources") if isinstance(candidate, dict) else None
    if not isinstance(candidate_metrics, dict):
        payload = {
            "ok": False,
            "mode": "compare",
            "error": "candidate_metrics_missing",
            "baseline": str(baseline_path),
            "result": candidate,
        }
        _write_json(Path(args.out), payload)
        print(json.dumps({"ok": False, "mode": "compare", "out": str(args.out)}, separators=(",", ":")))
        return 1

    fail_reasons: List[str] = []

    core_viol = _require_metric(candidate_metrics, "core_zero_copy_violations")
    pass_core = core_viol is not None and int(core_viol) == 0
    if not pass_core:
        fail_reasons.append("core_zero_copy_violations")

    boundary_p99 = _require_metric(candidate_metrics, "boundary_cpu_serialization_p99_ms")
    pass_boundary = boundary_p99 is not None and float(boundary_p99) <= 3.0
    if not pass_boundary:
        fail_reasons.append("boundary_cpu_serialization_p99_ms")

    baseline_fps = _require_metric(baseline_metrics, "fps_p50")
    candidate_fps = _require_metric(candidate_metrics, "fps_p50")
    fps_ratio = None
    pass_fps = False
    if baseline_fps is not None and baseline_fps > 0.0 and candidate_fps is not None:
        fps_ratio = float(candidate_fps) / float(baseline_fps)
        pass_fps = fps_ratio >= 0.95
    if not pass_fps:
        fail_reasons.append("fps_p50_ratio")

    baseline_lat = _require_metric(baseline_metrics, "pipeline_latency_ms_p95")
    candidate_lat = _require_metric(candidate_metrics, "pipeline_latency_ms_p95")
    baseline_lat_source = ""
    candidate_lat_source = ""
    if isinstance(baseline_sources, dict):
        baseline_lat_source = str(baseline_sources.get("pipeline_latency_ms_p95") or "")
    if isinstance(candidate_sources, dict):
        candidate_lat_source = str(candidate_sources.get("pipeline_latency_ms_p95") or "")
    latency_ratio = None
    pass_latency = False
    enforce_latency_ratio = baseline_lat_source == "pipeline_latency_ms_p95" and candidate_lat_source == "pipeline_latency_ms_p95"
    if enforce_latency_ratio and baseline_lat is not None and baseline_lat > 0.0 and candidate_lat is not None:
        latency_ratio = float(candidate_lat) / float(baseline_lat)
        pass_latency = latency_ratio <= 1.10
    elif not enforce_latency_ratio:
        pass_latency = True
    if not pass_latency:
        fail_reasons.append("pipeline_latency_ms_p95_ratio")

    baseline_gpu = _require_metric(baseline_metrics, "gpu_memory_used_p95_mib")
    candidate_gpu = _require_metric(candidate_metrics, "gpu_memory_used_p95_mib")
    gpu_ratio = None
    pass_gpu = False
    if baseline_gpu is not None and baseline_gpu > 0.0 and candidate_gpu is not None:
        gpu_ratio = float(candidate_gpu) / float(baseline_gpu)
        pass_gpu = gpu_ratio <= 1.10
    if not pass_gpu:
        fail_reasons.append("gpu_memory_used_p95_mib_ratio")

    if not bool(candidate.get("ok", False)):
        fail_reasons.append("candidate_capture_failed")

    compare_payload = {
        "ok": len(fail_reasons) == 0,
        "mode": "compare",
        "baseline": str(baseline_path),
        "captured_at": int(time.time()),
        "thresholds": {
            "core_zero_copy_violations": "==0",
            "boundary_cpu_serialization_p99_ms": "<=3.0",
            "fps_p50_ratio": ">=0.95",
            "pipeline_latency_ms_p95_ratio": "<=1.10",
            "gpu_memory_used_p95_mib_ratio": "<=1.10",
        },
        "baseline_metrics": baseline_metrics,
        "candidate": candidate,
        "metric_source_policy": {
            "pipeline_latency_ms_p95_ratio_enforced": bool(enforce_latency_ratio),
            "baseline_pipeline_latency_source": baseline_lat_source,
            "candidate_pipeline_latency_source": candidate_lat_source,
        },
        "ratios": {
            "fps_p50_ratio": fps_ratio,
            "pipeline_latency_ms_p95_ratio": latency_ratio,
            "gpu_memory_used_p95_mib_ratio": gpu_ratio,
        },
        "checks": {
            "core_zero_copy_violations": pass_core,
            "boundary_cpu_serialization_p99_ms": pass_boundary,
            "fps_p50_ratio": pass_fps,
            "pipeline_latency_ms_p95_ratio": pass_latency,
            "gpu_memory_used_p95_mib_ratio": pass_gpu,
        },
        "fail_reasons": fail_reasons,
    }

    _write_json(Path(args.out), compare_payload)
    print(json.dumps({"ok": compare_payload["ok"], "mode": "compare", "out": str(args.out)}, separators=(",", ":")))
    return 0 if compare_payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

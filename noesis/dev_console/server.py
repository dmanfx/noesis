from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from noesis.ds8_preflight import REPO_ROOT, run_preflight
from noesis.dev_console.activity import list_activity, record_activity_safe
from noesis.dev_console.config_editor import model_patch
from noesis.dev_console.diagnostics import diagnostics_snapshot
from noesis.dev_console.gate_catalog import build_gate_catalog
from noesis.dev_console.launch_candidates import build_launch_candidates
from noesis.dev_console.launch_decision import build_launch_decision
from noesis.dev_console.launch_diff import build_launch_diff
from noesis.dev_console.launch_spec import LaunchSpec, parse_env_lines
from noesis.dev_console.launch_plan import build_launch_plan
from noesis.dev_console.live_health import summarize_live_health
from noesis.dev_console.live_probe import probe_live_ws, send_live_control
from noesis.dev_console.log_intelligence import analyze_runtime_logs
from noesis.dev_console.manifest import load_knobs
from noesis.dev_console.materialize import materialize_launch_pipeline
from noesis.dev_console.metadata_compat import analyze_pipeline
from noesis.dev_console.model_matrix import build_model_matrix
from noesis.dev_console.pipeline_flow import describe_flow
from noesis.dev_console.presets import Preset, get_preset, list_presets
from noesis.dev_console.profiles import delete_profile, list_profiles, load_profile, save_profile
from noesis.dev_console.remediation import build_remediation
from noesis.dev_console.runtime_proxy import depth_refresh, runtime_health
from noesis.dev_console.source_probe import build_source_readiness
from noesis.dev_console.supervisor import RuntimeSupervisor
from noesis.dev_console.support_bundle import list_support_bundles, read_support_bundle, write_support_bundle
from noesis.dev_console.validator import validate_launch


STATIC_DIR = Path(__file__).resolve().parent / "static"


def _preset_spec(preset: Preset) -> Dict[str, Any]:
    return {
        "preset_id": preset.id,
        "pipeline_config": preset.pipeline_config,
        "cameras_config": preset.cameras_config,
        "pgie_profile": preset.pgie_profile,
        "size": preset.size,
        "tracking_mode": preset.tracking_mode,
        "runtime_entry": preset.runtime_entry,
        "enable_rest": preset.enable_rest,
        "rtsp_port": 8554,
        "env": preset.env,
    }


def _spec_from_payload(payload: Optional[Mapping[str, Any]]) -> LaunchSpec:
    data: Dict[str, Any] = dict(payload or {})
    preset_id = str(data.get("preset_id") or "").strip()
    if preset_id:
        preset = get_preset(preset_id)
        if preset is None:
            raise HTTPException(status_code=404, detail=f"Unknown preset: {preset_id}")
        base = _preset_spec(preset)
        base.update({key: value for key, value in data.items() if value not in (None, "") or key in {"size"}})
        data = base
    env_lines = data.pop("env_lines", "")
    if env_lines:
        env = dict(data.get("env") or {})
        env.update(parse_env_lines(str(env_lines)))
        data["env"] = env
    return LaunchSpec.from_mapping(data)


def _activity_spec(spec: LaunchSpec) -> Dict[str, Any]:
    return {
        "launch_id": spec.launch_id,
        "pipeline_config": spec.pipeline_config,
        "cameras_config": spec.cameras_config,
        "pgie_profile": spec.pgie_profile,
        "size": spec.size,
        "tracking_mode": spec.tracking_mode,
        "depth_enable_seconds": spec.depth_enable_seconds,
        "strict_baseline": spec.strict_baseline,
        "ports": {"ws": spec.ws_port, "rest": spec.rest_port, "rtsp": spec.rtsp_port},
        "env_count": len(spec.env or {}),
    }


def _activity_validation(validation: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    raw = validation if isinstance(validation, Mapping) else {}
    nested = raw.get("validation") if isinstance(raw.get("validation"), Mapping) else None
    payload = nested if nested is not None else raw
    counts = payload.get("counts") if isinstance(payload.get("counts"), Mapping) else {}
    return {
        "blocking": bool(payload.get("blocking")),
        "counts": {
            "block": int(counts.get("block", 0) or 0),
            "warn": int(counts.get("warn", 0) or 0),
            "info": int(counts.get("info", 0) or 0),
        },
    }


def _activity_validation_severity(validation: Optional[Mapping[str, Any]]) -> str:
    summary = _activity_validation(validation)
    counts = summary["counts"]
    if summary["blocking"] or counts["block"]:
        return "block"
    if counts["warn"]:
        return "warn"
    return "ok"


def _activity_validation_detail(validation: Optional[Mapping[str, Any]]) -> str:
    counts = _activity_validation(validation)["counts"]
    return f"{counts['block']} block / {counts['warn']} warn / {counts['info']} info"


def _activity_runtime(status: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "running": bool(status.get("running")),
        "pid": status.get("pid"),
        "returncode": status.get("returncode"),
        "uptime_s": status.get("uptime_s"),
        "launch_id": status.get("launch_id"),
        "materialized_pipeline": status.get("materialized_pipeline"),
        "log_path": status.get("log_path"),
    }


def _record_activity(event_type: str, title: str, *, severity: str = "info", detail: str = "", payload: Optional[Mapping[str, Any]] = None) -> None:
    record_activity_safe(event_type, title, severity=severity, detail=detail, payload=payload)


def create_app(supervisor: Optional[RuntimeSupervisor] = None) -> FastAPI:
    runtime = supervisor or RuntimeSupervisor()
    app = FastAPI(title="Noesis DS8 Dev Console", version="1.0.0")
    app.state.supervisor = runtime

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "repo_root": str(REPO_ROOT), "runtime": runtime.status()}

    @app.get("/api/summary")
    def summary() -> Dict[str, Any]:
        default_spec = LaunchSpec()
        pipeline_path = REPO_ROOT / default_spec.pipeline_config
        analysis: Dict[str, Any] = {}
        if pipeline_path.exists():
            analysis = analyze_pipeline(pipeline_path, tracking_mode=default_spec.tracking_mode)
        return {
            "repo_root": str(REPO_ROOT),
            "runtime": runtime.status(),
            "presets": [preset.to_dict() for preset in list_presets()],  # type: ignore[union-attr]
            "analysis": analysis,
            "knob_count": len(load_knobs()),
        }

    @app.get("/api/presets")
    def presets(runnable_only: bool = False) -> Dict[str, Any]:
        raw = list_presets(runnable_only=runnable_only)
        items = [item if isinstance(item, dict) else item.to_dict() for item in raw]
        return {"items": items}

    @app.get("/api/knobs")
    def knobs() -> Dict[str, Any]:
        return {"items": load_knobs()}

    @app.get("/api/profiles")
    def profiles() -> Dict[str, Any]:
        return {"items": list_profiles()}

    @app.get("/api/activity")
    def activity(limit: int = 100) -> Dict[str, Any]:
        return list_activity(limit=limit)

    @app.post("/api/gates")
    def gates(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = _spec_from_payload(payload)
        catalog = build_gate_catalog(spec)
        if payload.get("record_activity"):
            summary_payload = catalog.get("summary") if isinstance(catalog.get("summary"), Mapping) else {}
            overrides = [
                {"label": item.get("label"), "target": item.get("target"), "value": item.get("effective_value")}
                for item in catalog.get("controls", [])
                if isinstance(item, Mapping) and item.get("explicit") and item.get("target_type") == "env"
            ][:12]
            _record_activity(
                "gates.evaluate",
                "Updated gate deck",
                severity="ok",
                detail=(
                    f"{summary_payload.get('active', 0)} active / "
                    f"{summary_payload.get('explicit_overrides', 0)} env override"
                ),
                payload={"spec": _activity_spec(spec), "summary": summary_payload, "overrides": overrides},
            )
        return catalog

    @app.post("/api/profiles/save")
    def profile_save(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec_payload = payload.get("spec") if isinstance(payload.get("spec"), Mapping) else payload
        spec = _spec_from_payload(spec_payload)
        try:
            result = save_profile(
                name=str(payload.get("name") or ""),
                notes=str(payload.get("notes") or ""),
                spec=spec,
                profile_id=str(payload.get("profile_id") or "") or None,
            )
        except Exception as exc:
            _record_activity(
                "profile.save_failed",
                "Profile save failed",
                severity="warn",
                detail=str(exc),
                payload={"spec": _activity_spec(spec)},
            )
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        summary = result.get("summary") if isinstance(result.get("summary"), Mapping) else {}
        _record_activity(
            "profile.save",
            f"Saved profile {summary.get('name') or summary.get('id') or 'launch profile'}",
            severity="ok",
            payload={"profile": summary, "spec": _activity_spec(spec)},
        )
        return result

    @app.post("/api/profiles/load")
    def profile_load(payload: Dict[str, Any]) -> Dict[str, Any]:
        profile_id = str(payload.get("profile_id") or "")
        try:
            result = load_profile(profile_id)
        except FileNotFoundError as exc:
            _record_activity("profile.load_failed", "Profile load failed", severity="warn", detail=str(exc), payload={"profile_id": profile_id})
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            _record_activity("profile.load_failed", "Profile load failed", severity="warn", detail=str(exc), payload={"profile_id": profile_id})
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        summary = result.get("summary") if isinstance(result.get("summary"), Mapping) else {}
        _record_activity(
            "profile.load",
            f"Loaded profile {summary.get('name') or profile_id}",
            severity="info",
            payload={"profile": summary},
        )
        return result

    @app.post("/api/profiles/delete")
    def profile_delete(payload: Dict[str, Any]) -> Dict[str, Any]:
        profile_id = str(payload.get("profile_id") or "")
        try:
            result = delete_profile(profile_id)
        except FileNotFoundError as exc:
            _record_activity("profile.delete_failed", "Profile delete failed", severity="warn", detail=str(exc), payload={"profile_id": profile_id})
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        _record_activity("profile.delete", f"Removed profile {result.get('profile_id') or profile_id}", severity="info", payload=result)
        return result

    @app.post("/api/launch/preview")
    def launch_preview(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = _spec_from_payload(payload)
        materialized = materialize_launch_pipeline(spec, dry_run=True)
        validation = validate_launch(spec)
        _record_activity(
            "launch.preview",
            f"Previewed {spec.pgie_profile} {spec.tracking_mode}",
            severity=_activity_validation_severity(validation),
            detail=_activity_validation_detail(validation),
            payload={
                "spec": _activity_spec(spec),
                "validation": _activity_validation(validation),
                "materialized_pipeline": str(materialized),
            },
        )
        return {
            "spec": spec.to_dict(),
            "materialized_pipeline": str(materialized),
            "validation": validation,
        }

    @app.post("/api/launch/plan")
    def launch_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = _spec_from_payload(payload)
        plan = build_launch_plan(spec)
        if payload.get("record_activity"):
            artifacts = plan.get("artifacts") if isinstance(plan.get("artifacts"), Mapping) else {}
            missing = int(artifacts.get("missing", 0) or 0)
            artifact_summary = {
                "total": artifacts.get("total", 0),
                "missing": missing,
                "ready": artifacts.get("ready"),
                "missing_labels": [
                    item.get("label")
                    for item in artifacts.get("items", [])
                    if isinstance(item, Mapping) and not item.get("exists")
                ][:12],
            }
            _record_activity(
                "launch.plan",
                f"Refreshed launch plan for {spec.pgie_profile}",
                severity="warn" if missing else "ok",
                detail=f"{plan.get('change_count', 0)} tracked changes / {missing} missing artifact",
                payload={"spec": _activity_spec(spec), "change_count": plan.get("change_count"), "artifacts": artifact_summary},
            )
        return plan

    @app.post("/api/launch/diff")
    def launch_diff(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = _spec_from_payload(payload)
        diff = build_launch_diff(spec)
        if payload.get("record_activity"):
            summary_payload = diff.get("summary") if isinstance(diff.get("summary"), Mapping) else {}
            _record_activity(
                "launch.diff",
                f"Compared launch diff for {spec.pgie_profile}",
                severity="warn" if summary_payload.get("high_impact") else "ok",
                detail=(
                    f"{summary_payload.get('total', 0)} changes / "
                    f"{summary_payload.get('high_impact', 0)} high impact"
                ),
                payload={"spec": _activity_spec(spec), "summary": summary_payload},
            )
        return diff

    @app.post("/api/launch/validate")
    def launch_validate(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = _spec_from_payload(payload)
        validation = validate_launch(spec)
        _record_activity(
            "launch.validate",
            f"Validated {spec.pgie_profile} {spec.tracking_mode}",
            severity=_activity_validation_severity(validation),
            detail=_activity_validation_detail(validation),
            payload={"spec": _activity_spec(spec), "validation": _activity_validation(validation)},
        )
        return validation

    @app.post("/api/model-matrix")
    def model_matrix(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = _spec_from_payload(payload)
        matrix = build_model_matrix(spec)
        if payload.get("record_activity"):
            summary_payload = matrix.get("summary") if isinstance(matrix.get("summary"), Mapping) else {}
            _record_activity(
                "model.matrix",
                "Refreshed model matrix",
                severity="warn" if summary_payload.get("blocked") else "ok",
                detail=(
                    f"{summary_payload.get('ready', 0)} ready / "
                    f"{summary_payload.get('warn', 0)} warn / "
                    f"{summary_payload.get('blocked', 0)} blocked"
                ),
                payload={
                    "spec": _activity_spec(spec),
                    "summary": summary_payload,
                    "active_id": matrix.get("active_id"),
                },
            )
        return matrix

    @app.post("/api/sources")
    def sources(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = _spec_from_payload(payload)
        source_payload = build_source_readiness(
            spec,
            probe_network=bool(payload.get("probe_network", True)),
            timeout_s=float(payload.get("timeout_s", 0.35)),
        )
        if payload.get("record_activity"):
            summary_payload = source_payload.get("summary") if isinstance(source_payload.get("summary"), Mapping) else {}
            _record_activity(
                "sources.scan",
                "Scanned source readiness",
                severity="block" if summary_payload.get("blocked") else ("warn" if summary_payload.get("warn") else "ok"),
                detail=(
                    f"{summary_payload.get('ready', 0)} ready / "
                    f"{summary_payload.get('warn', 0)} warn / "
                    f"{summary_payload.get('blocked', 0)} blocked"
                ),
                payload={"spec": _activity_spec(spec), "summary": summary_payload},
            )
        return source_payload

    @app.post("/api/launch/decision")
    def launch_decision(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = _spec_from_payload(payload)
        runtime_status = runtime.status()
        managed_pid = runtime_status.get("pid") if isinstance(runtime_status.get("pid"), int) else None
        ignore_managed_ports = bool(runtime_status.get("running") and managed_pid is not None)
        decision = build_launch_decision(
            spec,
            runtime_status=runtime_status,
            managed_pid=managed_pid,
            ignore_managed_runtime_ports=ignore_managed_ports,
            probe_network=bool(payload.get("probe_network", True)),
            timeout_s=float(payload.get("timeout_s", 0.35)),
        )
        if payload.get("record_activity"):
            _record_activity(
                "launch.decision",
                f"Evaluated launch decision: {decision.get('status')}",
                severity="block" if decision.get("status") == "blocked" else ("warn" if decision.get("status") == "attention" else "ok"),
                detail=str(decision.get("summary") or ""),
                payload={
                    "spec": _activity_spec(spec),
                    "status": decision.get("status"),
                    "score": decision.get("score"),
                    "start_allowed": decision.get("start_allowed"),
                },
            )
        return decision

    @app.post("/api/launch/candidates")
    def launch_candidates(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = _spec_from_payload(payload)
        runtime_status = runtime.status()
        managed_pid = runtime_status.get("pid") if isinstance(runtime_status.get("pid"), int) else None
        candidates = build_launch_candidates(
            spec,
            runtime_status=runtime_status,
            managed_pid=managed_pid,
            probe_network=bool(payload.get("probe_network", True)),
            timeout_s=float(payload.get("timeout_s", 0.35)),
        )
        if payload.get("record_activity"):
            summary_payload = candidates.get("summary") if isinstance(candidates.get("summary"), Mapping) else {}
            _record_activity(
                "launch.candidates",
                "Compared launch candidates",
                severity="ok" if summary_payload.get("ready") else ("warn" if summary_payload.get("attention") else "block"),
                detail=(
                    f"{summary_payload.get('ready', 0)} ready / "
                    f"{summary_payload.get('attention', 0)} attention / "
                    f"{summary_payload.get('blocked', 0)} blocked"
                ),
                payload={"spec": _activity_spec(spec), "summary": summary_payload},
            )
        return candidates

    @app.post("/api/runtime/start")
    def runtime_start(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = _spec_from_payload(payload)
        runtime_status = runtime.status()
        managed_pid = runtime_status.get("pid") if isinstance(runtime_status.get("pid"), int) else None
        ignore_managed_ports = bool(runtime_status.get("running") and managed_pid is not None)
        decision = build_launch_decision(
            spec,
            runtime_status=runtime_status,
            managed_pid=managed_pid,
            ignore_managed_runtime_ports=ignore_managed_ports,
            probe_network=bool(payload.get("probe_network", True)),
            timeout_s=float(payload.get("timeout_s", 0.35)),
        )
        if not decision.get("start_allowed"):
            validation = decision.get("validation") if isinstance(decision.get("validation"), Mapping) else None
            detail = {
                "error": str(decision.get("summary") or "Launch decision is not start-ready"),
                "decision": decision,
                "validation": validation,
            }
            _record_activity(
                "runtime.start_blocked",
                "Runtime start blocked by launch decision",
                severity="block" if decision.get("status") == "blocked" else "warn",
                detail=str(decision.get("summary") or ""),
                payload={
                    "spec": _activity_spec(spec),
                    "status": decision.get("status"),
                    "score": decision.get("score"),
                    "start_allowed": decision.get("start_allowed"),
                    "validation": _activity_validation(validation),
                },
            )
            raise HTTPException(status_code=409, detail=detail)
        try:
            result = runtime.start(spec)
        except Exception as exc:
            detail = {"error": str(exc), "validation": runtime.last_validation}
            _record_activity(
                "runtime.start_blocked",
                "Runtime start blocked",
                severity="block",
                detail=str(exc),
                payload={"spec": _activity_spec(spec), "validation": _activity_validation(runtime.last_validation)},
            )
            raise HTTPException(status_code=409, detail=detail) from exc
        _record_activity(
            "runtime.start",
            f"Started runtime {result.get('launch_id') or spec.launch_id}",
            severity="ok",
            payload={"spec": _activity_spec(spec), "runtime": _activity_runtime(result), "validation": _activity_validation(runtime.last_validation)},
        )
        return result

    @app.post("/api/runtime/stop")
    def runtime_stop() -> Dict[str, Any]:
        result = runtime.stop()
        _record_activity("runtime.stop", "Stop signal sent", severity="info", payload={"runtime": _activity_runtime(result)})
        return result

    @app.post("/api/runtime/restart")
    def runtime_restart(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = _spec_from_payload(payload)
        runtime_status = runtime.status()
        managed_pid = runtime_status.get("pid") if isinstance(runtime_status.get("pid"), int) else None
        decision_runtime_status = dict(runtime_status)
        ignore_managed_ports = bool(runtime_status.get("running") and managed_pid is not None)
        if ignore_managed_ports:
            decision_runtime_status["running"] = False
        decision = build_launch_decision(
            spec,
            runtime_status=decision_runtime_status,
            managed_pid=managed_pid,
            ignore_managed_runtime_ports=ignore_managed_ports,
            probe_network=bool(payload.get("probe_network", True)),
            timeout_s=float(payload.get("timeout_s", 0.35)),
        )
        if not decision.get("start_allowed"):
            validation = decision.get("validation") if isinstance(decision.get("validation"), Mapping) else None
            detail = {
                "error": str(decision.get("summary") or "Launch decision is not restart-ready"),
                "decision": decision,
                "validation": validation,
            }
            _record_activity(
                "runtime.restart_blocked",
                "Runtime restart blocked by launch decision",
                severity="block" if decision.get("status") == "blocked" else "warn",
                detail=str(decision.get("summary") or ""),
                payload={
                    "spec": _activity_spec(spec),
                    "status": decision.get("status"),
                    "score": decision.get("score"),
                    "start_allowed": decision.get("start_allowed"),
                    "validation": _activity_validation(validation),
                },
            )
            raise HTTPException(status_code=409, detail=detail)
        try:
            result = runtime.restart(spec)
        except Exception as exc:
            _record_activity(
                "runtime.restart_blocked",
                "Runtime restart blocked",
                severity="block",
                detail=str(exc),
                payload={"spec": _activity_spec(spec), "validation": _activity_validation(runtime.last_validation)},
            )
            raise HTTPException(status_code=409, detail={"error": str(exc), "validation": runtime.last_validation}) from exc
        _record_activity(
            "runtime.restart",
            f"Restarted runtime {result.get('launch_id') or spec.launch_id}",
            severity="ok",
            payload={"spec": _activity_spec(spec), "runtime": _activity_runtime(result), "validation": _activity_validation(runtime.last_validation)},
        )
        return result

    @app.get("/api/runtime/status")
    def runtime_status() -> Dict[str, Any]:
        return runtime.status()

    @app.get("/api/runtime/logs")
    def runtime_logs(lines: int = 240) -> Dict[str, Any]:
        return runtime.tail_log(lines=lines)

    @app.get("/api/runtime/log-insights")
    def runtime_log_insights(lines: int = 600) -> Dict[str, Any]:
        runtime_status = runtime.status()
        log_payload = runtime.tail_log(lines=lines)
        return analyze_runtime_logs(
            log_payload.get("lines", []),
            path=log_payload.get("path"),
            runtime_status=runtime_status,
        )

    @app.post("/api/runtime/depth-refresh")
    def runtime_depth_refresh(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec_payload = payload.get("spec") if isinstance(payload.get("spec"), Mapping) else {}
        spec = _spec_from_payload(spec_payload)
        seconds = int(payload.get("seconds", 10))
        result = depth_refresh(rest_host=spec.rest_host, rest_port=spec.rest_port, seconds=seconds)
        _record_activity(
            "runtime.depth_refresh",
            f"Requested depth burst for {seconds}s",
            severity="ok" if result.get("ok") else "warn",
            detail=str(result.get("error") or result.get("status") or ""),
            payload={"spec": _activity_spec(spec), "seconds": seconds, "status": result.get("status"), "ok": result.get("ok")},
        )
        return result

    @app.post("/api/runtime/health")
    def proxied_runtime_health(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = _spec_from_payload(payload)
        result = runtime_health(rest_host=spec.rest_host, rest_port=spec.rest_port)
        _record_activity(
            "runtime.health",
            "Sampled runtime health",
            severity="ok" if result.get("ok") else "warn",
            detail=str(result.get("error") or result.get("status") or ""),
            payload={"spec": _activity_spec(spec), "status": result.get("status"), "ok": result.get("ok")},
        )
        return result

    @app.post("/api/pipeline/flow")
    def pipeline_flow(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = _spec_from_payload(payload)
        return describe_flow(
            pipeline_config=spec.pipeline_config,
            pgie_profile=spec.pgie_profile,
            size=spec.size,
            tracking_mode=spec.tracking_mode,
            rtsp_port=spec.rtsp_port,
            depth_enable_seconds=spec.depth_enable_seconds,
            env=spec.env,
        )

    @app.post("/api/config/model-patch")
    def config_model_patch(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = _spec_from_payload(payload)
        patch = model_patch(
            REPO_ROOT / spec.pipeline_config,
            pgie_profile=spec.pgie_profile,
            size=spec.size,
            reid_enable=payload.get("reid_enable"),
            pose_enable=payload.get("pose_enable"),
            mapanything_enable=payload.get("mapanything_enable"),
            tracking_mode=spec.tracking_mode,
        )
        return {"config": patch}

    @app.post("/api/preflight/raw")
    def preflight_raw(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = _spec_from_payload(payload)
        results = run_preflight(
            pipeline_path=REPO_ROOT / spec.pipeline_config,
            tracking_mode=spec.tracking_mode,
            strict_baseline=spec.strict_baseline,
            include_cuda=bool(payload.get("include_cuda", False)),
        )
        return {"results": [result.to_dict() for result in results]}

    @app.post("/api/diagnostics")
    def diagnostics(payload: Dict[str, Any]) -> Dict[str, Any]:
        runtime_status = runtime.status()
        managed_pid = runtime_status.get("pid") if isinstance(runtime_status.get("pid"), int) else None
        return diagnostics_snapshot(_spec_from_payload(payload), managed_pid=managed_pid)

    @app.post("/api/remediation")
    def remediation(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = _spec_from_payload(payload)
        runtime_status = runtime.status()
        managed_pid = runtime_status.get("pid") if isinstance(runtime_status.get("pid"), int) else None
        validation = validate_launch(spec)
        diagnostics_payload = diagnostics_snapshot(spec, managed_pid=managed_pid)
        plan = build_launch_plan(spec)
        return build_remediation(spec, validation=validation, diagnostics=diagnostics_payload, launch_plan=plan)

    @app.post("/api/support/bundle")
    def support_bundle(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = _spec_from_payload(payload)
        result = write_support_bundle(spec, runtime.status())
        _record_activity(
            "support.bundle",
            f"Saved support bundle {result.get('launch_id') or spec.launch_id}",
            severity="block" if result.get("blocking") else "ok",
            detail=f"{result.get('process_count', 0)} observed process owner(s)",
            payload={
                "spec": _activity_spec(spec),
                "bundle_path": result.get("bundle_path"),
                "summary_path": result.get("summary_path"),
                "counts": result.get("counts"),
                "live_connected": result.get("live_connected"),
            },
        )
        return result

    @app.get("/api/support/bundles")
    def support_bundles(limit: int = 30) -> Dict[str, Any]:
        return list_support_bundles(limit=limit)

    @app.get("/api/support/bundles/{bundle_id}")
    def support_bundle_detail(bundle_id: str) -> Dict[str, Any]:
        try:
            return read_support_bundle(bundle_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/live/probe")
    def live_probe(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = _spec_from_payload(payload)
        result = probe_live_ws(spec.ws_host, int(spec.ws_port), timeout_s=float(payload.get("timeout_s", 3.0)))
        if payload.get("record_activity"):
            _record_activity(
                "live.probe",
                "Probed live WebSocket",
                severity="ok" if result.get("connected") else "warn",
                detail=str(result.get("error") or result.get("url") or ""),
                payload={
                    "spec": _activity_spec(spec),
                    "connected": result.get("connected"),
                    "pong": result.get("pong"),
                    "message_types": result.get("message_types"),
                },
            )
        return result

    @app.post("/api/live/health")
    def live_health(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec = _spec_from_payload(payload)
        probe = probe_live_ws(spec.ws_host, int(spec.ws_port), timeout_s=float(payload.get("timeout_s", 3.0)))
        result = summarize_live_health(probe)
        if payload.get("record_activity"):
            _record_activity(
                "live.health",
                f"Sampled live health: {result.get('status', 'unknown')}",
                severity="ok" if result.get("status") == "healthy" else "warn",
                detail=str(result.get("summary") or ""),
                payload={
                    "spec": _activity_spec(spec),
                    "status": result.get("status"),
                    "score": result.get("score"),
                    "camera_count": len(result.get("cameras") or []),
                },
            )
        return result

    @app.post("/api/live/control")
    def live_control(payload: Dict[str, Any]) -> Dict[str, Any]:
        spec_payload = payload.get("spec") if isinstance(payload.get("spec"), Mapping) else payload
        spec = _spec_from_payload(spec_payload)
        action = str(payload.get("action", ""))
        result = send_live_control(
            spec.ws_host,
            int(spec.ws_port),
            action=action,
            enabled=payload.get("enabled") if isinstance(payload.get("enabled"), bool) else None,
        )
        _record_activity(
            "live.control",
            f"Sent live control {action or 'unknown'}",
            severity="ok" if result.get("ok") else "warn",
            detail=str(result.get("error") or result.get("status") or ""),
            payload={
                "spec": _activity_spec(spec),
                "action": action,
                "enabled": payload.get("enabled") if isinstance(payload.get("enabled"), bool) else None,
                "ok": result.get("ok"),
                "status": result.get("status"),
            },
        )
        return result

    return app


app = create_app()

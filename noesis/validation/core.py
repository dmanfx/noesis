from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


class CheckStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class ResultLevel(str, Enum):
    VERIFIED = "verified"
    LIKELY = "likely"
    TENTATIVE = "tentative"
    UNTRUSTED = "untrusted"
    INVALID = "invalid"


class FailureType(str, Enum):
    CALIBRATION = "calibration_failure"
    TRANSFORM = "transform_failure"
    PROJECTION = "projection_failure"
    SCENE = "scene_failure"
    TEMPORAL = "temporal_failure"
    SEMANTIC = "semantic_failure"
    SYNC = "sync_failure"
    MODEL = "model_failure"
    DATA_QUALITY = "data_quality_failure"
    REGRESSION = "regression_failure"
    INFRASTRUCTURE = "infrastructure_failure"


STATUS_RANK: dict[CheckStatus, int] = {
    CheckStatus.PASS: 0,
    CheckStatus.SKIPPED: 1,
    CheckStatus.WARNING: 2,
    CheckStatus.BLOCKED: 3,
    CheckStatus.FAIL: 4,
}


def _coerce_status(value: CheckStatus | str) -> CheckStatus:
    if isinstance(value, CheckStatus):
        return value
    return CheckStatus(str(value))


def _coerce_level(value: ResultLevel | str) -> ResultLevel:
    if isinstance(value, ResultLevel):
        return value
    return ResultLevel(str(value))


def _coerce_failure_type(value: FailureType | str | None) -> FailureType | None:
    if value is None:
        return None
    if isinstance(value, FailureType):
        return value
    return FailureType(str(value))


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def worst_status(statuses: Sequence[CheckStatus | str]) -> CheckStatus:
    if not statuses:
        return CheckStatus.SKIPPED
    coerced = [_coerce_status(status) for status in statuses]
    return max(coerced, key=lambda status: STATUS_RANK[status])


def level_from_status(status: CheckStatus | str, *, confidence: float | None = None) -> ResultLevel:
    status_value = _coerce_status(status)
    if status_value == CheckStatus.FAIL:
        return ResultLevel.INVALID
    if status_value == CheckStatus.BLOCKED:
        return ResultLevel.UNTRUSTED
    if status_value == CheckStatus.WARNING:
        return ResultLevel.TENTATIVE
    if status_value == CheckStatus.SKIPPED:
        return ResultLevel.TENTATIVE
    if confidence is None:
        return ResultLevel.LIKELY
    if confidence >= 0.85:
        return ResultLevel.VERIFIED
    if confidence >= 0.65:
        return ResultLevel.LIKELY
    if confidence >= 0.45:
        return ResultLevel.TENTATIVE
    return ResultLevel.UNTRUSTED


@dataclass(frozen=True)
class ConfidenceScores:
    detection: float | None = None
    tracking: float | None = None
    reid: float | None = None
    projection: float | None = None
    geometry: float | None = None
    temporal: float | None = None
    semantic: float | None = None
    end_to_end: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "detection": self.detection,
            "tracking": self.tracking,
            "reid": self.reid,
            "projection": self.projection,
            "geometry": self.geometry,
            "temporal": self.temporal,
            "semantic": self.semantic,
            "end_to_end": self.end_to_end,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ConfidenceScores":
        if not isinstance(payload, Mapping):
            return cls()

        def _score(name: str) -> float | None:
            value = payload.get(name)
            if value is None:
                return None
            try:
                value_f = float(value)
            except Exception:
                return None
            if value_f < 0.0:
                return 0.0
            if value_f > 1.0:
                return 1.0
            return value_f

        return cls(
            detection=_score("detection"),
            tracking=_score("tracking"),
            reid=_score("reid"),
            projection=_score("projection"),
            geometry=_score("geometry"),
            temporal=_score("temporal"),
            semantic=_score("semantic"),
            end_to_end=_score("end_to_end"),
        )


@dataclass(frozen=True)
class SourceMetadata:
    repo: str = "Noesis_Devel"
    git_revision: str | None = None
    pipeline_config: str | None = None
    cameras_config: str | None = None
    menon_available: bool | None = None
    menon_revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "git_revision": self.git_revision,
            "pipeline_config": self.pipeline_config,
            "cameras_config": self.cameras_config,
            "menon_available": self.menon_available,
            "menon_revision": self.menon_revision,
        }


@dataclass(frozen=True)
class ValidationCheck:
    id: str
    domain: str
    name: str
    status: CheckStatus | str
    level: ResultLevel | str | None = None
    failure_type: FailureType | str | None = None
    severity: str | None = None
    camera: str | None = None
    room: str | None = None
    metric: Mapping[str, Any] = field(default_factory=dict)
    threshold: Mapping[str, Any] = field(default_factory=dict)
    evidence: Sequence[str] = field(default_factory=list)
    detail: str = ""
    suggested_next_diagnostic: str | None = None

    def __post_init__(self) -> None:
        status = _coerce_status(self.status)
        object.__setattr__(self, "status", status)
        if self.level is None:
            confidence = None
            for key in ("confidence", "score", "value"):
                raw = self.metric.get(key)
                try:
                    confidence = float(raw)
                    break
                except Exception:
                    continue
            object.__setattr__(self, "level", level_from_status(status, confidence=confidence))
        else:
            object.__setattr__(self, "level", _coerce_level(self.level))
        object.__setattr__(self, "failure_type", _coerce_failure_type(self.failure_type))
        if self.severity is None:
            severity = "error" if status == CheckStatus.FAIL else "warning" if status == CheckStatus.WARNING else status.value
            object.__setattr__(self, "severity", severity)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "domain": self.domain,
            "name": self.name,
            "status": _coerce_status(self.status).value,
            "level": _coerce_level(self.level or ResultLevel.TENTATIVE).value,
            "failure_type": self.failure_type.value if isinstance(self.failure_type, FailureType) else None,
            "severity": self.severity,
            "camera": self.camera,
            "room": self.room,
            "metric": dict(_json_safe(self.metric)),
            "threshold": dict(_json_safe(self.threshold)),
            "evidence": list(self.evidence),
            "detail": self.detail,
            "suggested_next_diagnostic": self.suggested_next_diagnostic,
        }
        return {key: value for key, value in payload.items() if value not in (None, [], {})}


@dataclass
class ValidationReport:
    run_id: str
    source: SourceMetadata = field(default_factory=SourceMetadata)
    scope: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    confidence: ConfidenceScores = field(default_factory=ConfidenceScores)
    checks: list[ValidationCheck] = field(default_factory=list)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def add_check(self, check: ValidationCheck) -> None:
        self.checks.append(check)

    @property
    def status(self) -> CheckStatus:
        return worst_status([check.status for check in self.checks])

    @property
    def level(self) -> ResultLevel:
        confidence = self.confidence.end_to_end
        return level_from_status(self.status, confidence=confidence)

    def summary(self) -> dict[str, Any]:
        counts = {status.value: 0 for status in CheckStatus}
        for check in self.checks:
            counts[_coerce_status(check.status).value] += 1
        return {
            "status": self.status.value,
            "level": self.level.value,
            "end_to_end_confidence": self.confidence.end_to_end,
            "failure_count": counts[CheckStatus.FAIL.value],
            "warning_count": counts[CheckStatus.WARNING.value],
            "blocked_count": counts[CheckStatus.BLOCKED.value],
            "pass_count": counts[CheckStatus.PASS.value],
            "skipped_count": counts[CheckStatus.SKIPPED.value],
            "check_count": len(self.checks),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "run_id": str(self.run_id),
            "created_at": str(self.created_at),
            "source": self.source.to_dict(),
            "scope": _json_safe(dict(self.scope)),
            "summary": self.summary(),
            "confidence": self.confidence.to_dict(),
            "checks": [check.to_dict() for check in self.checks],
            "artifacts": _json_safe(dict(self.artifacts)),
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

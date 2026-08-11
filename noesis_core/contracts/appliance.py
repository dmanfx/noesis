from __future__ import annotations

from datetime import datetime, timezone
import os
import re
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .base import ContractModel


_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9-]{5,63}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_IMAGE_ID_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
_GIT_OID_RE = re.compile(r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z$"
)
_STORE_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_CANONICAL_TEXT_RE = re.compile(r"^[^\x00-\x1f\x7f]+$")
_MAX_SAFE_INTEGER = (1 << 53) - 1

Identifier = Annotated[str, Field(pattern=_IDENTIFIER_RE.pattern)]
Sha256 = Annotated[str, Field(pattern=_SHA256_RE.pattern)]
GitOid = Annotated[str, Field(pattern=_GIT_OID_RE.pattern)]
ImageId = Annotated[str, Field(pattern=_IMAGE_ID_RE.pattern)]
SchemaVersion = Annotated[int, Field(strict=True, ge=1, le=1_000_000)]
BaselineBytes = Annotated[int, Field(strict=True, ge=0, le=128 * 1024 * 1024)]
BaselineFileCount = Annotated[int, Field(strict=True, ge=1, le=10_000_000)]
SafeNonNegativeInteger = Annotated[
    int, Field(strict=True, ge=0, le=_MAX_SAFE_INTEGER)
]
SafePositiveInteger = Annotated[int, Field(strict=True, ge=1, le=_MAX_SAFE_INTEGER)]


def _canonical_text(value: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("value must be a non-empty canonical string")
    if len(value) > maximum or _CANONICAL_TEXT_RE.fullmatch(value) is None:
        raise ValueError("value must be a bounded canonical string")
    return value


def _absolute_path(value: str) -> str:
    value = _canonical_text(value, maximum=4096)
    if (
        not os.path.isabs(value)
        or os.path.normpath(value) != value
        or "//" in value
    ):
        raise ValueError("path must be normalized and absolute")
    return value


def _timestamp(value: str) -> str:
    value = _canonical_text(value, maximum=24)
    if _TIMESTAMP_RE.fullmatch(value) is None:
        raise ValueError("timestamp must use UTC millisecond RFC3339 form")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError("timestamp is not a valid UTC instant") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z" != value:
        raise ValueError("timestamp is not canonical")
    return value


def _paths_overlap(left: str, right: str) -> bool:
    try:
        common = os.path.commonpath((left, right))
    except ValueError:
        return False
    return common in {left, right}


class CheckoutBinding(ContractModel):
    root: str
    snapshot_kind: Literal["menon-runtime-v1", "noesis-runtime-v1"]
    snapshot_sha256: Sha256
    software_revision: GitOid

    @field_validator("root")
    @classmethod
    def _root_is_absolute(cls, value: str) -> str:
        return _absolute_path(value)


class StateReleaseBinding(ContractModel):
    release_id: Identifier
    manifest_path: str
    manifest_sha256: Sha256

    @field_validator("manifest_path")
    @classmethod
    def _manifest_is_absolute(cls, value: str) -> str:
        return _absolute_path(value)


class ApplianceEndpoints(ContractModel):
    websocket: Literal["ws://127.0.0.1:6008"]
    rest: Literal["http://127.0.0.1:8080"]
    rtsp: Literal["rtsp://127.0.0.1:8554/mosaic"]


class ApplianceReadinessVersions(ContractModel):
    capability_contract_version: Literal[1]
    deployment_health_contract_version: Literal[1]
    websocket_health_contract_version: Literal[2]


class DS8RuntimeSelector(ContractModel):
    family: Literal["ds8"]
    pgie_profile: Literal[
        "yolo11",
        "yolo11_seg",
        "yolo26",
        "yolo26_seg",
        "rfdetr",
        "rfdetr_seg",
        "wholebody49",
    ]
    model_size: Literal["n", "s", "m", "l", "x"]
    tracking_mode: Literal["baseline", "v3dt"]

    @model_validator(mode="after")
    def _profile_size_is_approved(self) -> "DS8RuntimeSelector":
        sizes = {
            "yolo11": frozenset(("s", "m", "l")),
            "yolo11_seg": frozenset(("s", "m", "l")),
            "yolo26": frozenset(("n", "s", "m", "l", "x")),
            "yolo26_seg": frozenset(("n", "s", "m")),
            "rfdetr": frozenset(("n", "s", "m")),
            "rfdetr_seg": frozenset(("n", "s", "m")),
            "wholebody49": frozenset(("s", "x")),
        }
        if self.model_size not in sizes[self.pgie_profile]:
            raise ValueError("PGIE profile/model size is not an approved DS8 pair")
        return self

    @property
    def variant(self) -> str:
        return (
            f"ds8:{self.pgie_profile}:{self.model_size}:{self.tracking_mode}"
        )


class DS9RuntimeSelector(ContractModel):
    family: Literal["ds9"]
    lane: Literal["baseline", "v3dt", "wholebody49-s", "wholebody49-x"]
    supervisor_path: str
    runtime_image_id: ImageId
    build_image_id: ImageId
    docker_root: str
    artifact_root: str
    runtime_root: str
    asset_realization_sha256: Sha256
    ownership_matrix_sha256: Sha256

    @field_validator(
        "supervisor_path", "docker_root", "artifact_root", "runtime_root"
    )
    @classmethod
    def _paths_are_absolute(cls, value: str) -> str:
        return _absolute_path(value)

    @model_validator(mode="after")
    def _roots_are_disjoint(self) -> "DS9RuntimeSelector":
        roots = (self.docker_root, self.artifact_root, self.runtime_root)
        if any(
            _paths_overlap(left, right)
            for index, left in enumerate(roots)
            for right in roots[index + 1 :]
        ):
            raise ValueError("DS9 Docker, artifact, and runtime roots must be disjoint")
        return self

    @property
    def variant(self) -> str:
        return f"ds9:{self.lane}"


RuntimeSelector = Annotated[
    DS8RuntimeSelector | DS9RuntimeSelector,
    Field(discriminator="family"),
]


class DeploymentSelector(ContractModel):
    contract: Literal["noesis.appliance.deployment_selector"]
    contract_version: Literal[1]
    deployment_id: Identifier
    created_at: str
    bundle_sha256: Sha256
    menon_checkout: CheckoutBinding
    noesis_checkout: CheckoutBinding
    state_release: StateReleaseBinding
    endpoints: ApplianceEndpoints
    runtime: RuntimeSelector
    readiness: ApplianceReadinessVersions

    @field_validator("created_at")
    @classmethod
    def _created_at_is_canonical(cls, value: str) -> str:
        return _timestamp(value)

    @model_validator(mode="after")
    def _relational_contract_is_valid(self) -> "DeploymentSelector":
        if self.menon_checkout.snapshot_kind != "menon-runtime-v1":
            raise ValueError("Menon checkout must use menon-runtime-v1")
        if self.noesis_checkout.snapshot_kind != "noesis-runtime-v1":
            raise ValueError("Noesis checkout must use noesis-runtime-v1")
        if _paths_overlap(self.menon_checkout.root, self.noesis_checkout.root):
            raise ValueError("Menon and Noesis checkout roots must be disjoint")
        if isinstance(self.runtime, DS9RuntimeSelector):
            expected_supervisor = os.path.join(
                self.noesis_checkout.root,
                "DS9",
                "scripts",
                "run_canonical_runtime_container.py",
            )
            if self.runtime.supervisor_path != expected_supervisor:
                raise ValueError("DS9 supervisor path is not the canonical runtime supervisor")
        return self


class StateBaselineFile(ContractModel):
    sha256: Sha256
    bytes: BaselineBytes


class StateBaseline(ContractModel):
    contract: Literal["noesis.appliance.state_baseline"]
    contract_version: Literal[1]
    release_id: Identifier
    created_at: str
    files: dict[str, StateBaselineFile]

    @field_validator("created_at")
    @classmethod
    def _created_at_is_canonical(cls, value: str) -> str:
        return _timestamp(value)

    @field_validator("files")
    @classmethod
    def _files_are_closed_and_sorted(
        cls, value: dict[str, StateBaselineFile]
    ) -> dict[str, StateBaselineFile]:
        if not 1 <= len(value) <= 100_000:
            raise ValueError("baseline files must contain 1..100000 entries")
        normalized: dict[str, StateBaselineFile] = {}
        for path, row in sorted(value.items()):
            if (
                not isinstance(path, str)
                or not path
                or path != path.strip()
                or "\\" in path
                or path.startswith("/")
                or _CANONICAL_TEXT_RE.fullmatch(path) is None
                or len(path) > 4096
            ):
                raise ValueError("baseline path must be bounded canonical POSIX-relative text")
            components = path.split("/")
            if any(component in {"", ".", ".."} for component in components):
                raise ValueError("baseline paths must not contain empty or dot segments")
            normalized[path] = row
        return normalized


class StateMigration(ContractModel):
    mode: Literal["fresh", "clone", "migrate", "import_ds8"]
    tool_sha256: Sha256
    report_sha256: Sha256


class StateBaselineBinding(ContractModel):
    inventory_path: str
    inventory_sha256: Sha256
    file_count: BaselineFileCount
    byte_count: SafeNonNegativeInteger

    @field_validator("inventory_path")
    @classmethod
    def _inventory_is_absolute(cls, value: str) -> str:
        return _absolute_path(value)


class StateRelease(ContractModel):
    contract: Literal["noesis.appliance.state_release"]
    contract_version: Literal[1]
    release_id: Identifier
    parent_release_id: Identifier | None
    created_at: str
    root: str
    schemas: dict[str, SchemaVersion]
    migration: StateMigration
    baseline: StateBaselineBinding

    @field_validator("created_at")
    @classmethod
    def _created_at_is_canonical(cls, value: str) -> str:
        return _timestamp(value)

    @field_validator("root")
    @classmethod
    def _root_is_absolute(cls, value: str) -> str:
        return _absolute_path(value)

    @field_validator("schemas")
    @classmethod
    def _schemas_are_bounded_and_sorted(cls, value: dict[str, int]) -> dict[str, int]:
        if not 1 <= len(value) <= 64:
            raise ValueError("schemas must contain 1..64 entries")
        normalized: dict[str, int] = {}
        for name, version in sorted(value.items()):
            if _STORE_RE.fullmatch(name) is None:
                raise ValueError("state schema name is invalid")
            if isinstance(version, bool) or not 1 <= int(version) <= 1_000_000:
                raise ValueError("state schema version must be a positive bounded integer")
            normalized[name] = int(version)
        return normalized

    @model_validator(mode="after")
    def _baseline_is_inside_release(self) -> "StateRelease":
        common = os.path.commonpath((self.root, self.baseline.inventory_path))
        if common != self.root or self.baseline.inventory_path == self.root:
            raise ValueError("state baseline must be stored beneath the release root")
        return self


class DeploymentHealth(ContractModel):
    contract: Literal["noesis.appliance.deployment_health"]
    contract_version: Literal[1]
    deployment_id: Identifier
    selector_sha256: Sha256
    state_release_id: Identifier
    runtime_family: Literal["ds8", "ds9"]
    runtime_variant: str = Field(min_length=1, max_length=128)
    instance_id: str = Field(min_length=1, max_length=160)
    run_id: str = Field(min_length=1, max_length=160)
    boot_id: Identifier
    software_revision: GitOid
    generated_at_us: SafePositiveInteger
    ready: Literal[True]

    @field_validator("runtime_variant", "instance_id", "run_id")
    @classmethod
    def _identity_text_is_canonical(cls, value: str) -> str:
        return _canonical_text(value, maximum=160)

    @model_validator(mode="after")
    def _variant_matches_family(self) -> "DeploymentHealth":
        if not self.runtime_variant.startswith(f"{self.runtime_family}:"):
            raise ValueError("runtime variant must match the runtime family")
        return self


class RuntimeDeploymentContext(ContractModel):
    """Supervisor-to-runtime identity handoff after host-side selector proof."""

    contract: Literal["noesis.appliance.runtime_context"]
    contract_version: Literal[1]
    deployment_id: Identifier
    selector_sha256: Sha256
    state_release_id: Identifier
    runtime_family: Literal["ds8", "ds9"]
    runtime_variant: str = Field(min_length=1, max_length=128)
    boot_id: Identifier
    software_revision: GitOid

    @field_validator("runtime_variant")
    @classmethod
    def _variant_is_canonical(cls, value: str) -> str:
        return _canonical_text(value, maximum=128)

    @model_validator(mode="after")
    def _variant_matches_family(self) -> "RuntimeDeploymentContext":
        if not self.runtime_variant.startswith(f"{self.runtime_family}:"):
            raise ValueError("runtime variant must match the runtime family")
        return self


class WebSocketDeploymentHealth(ContractModel):
    type: Literal["health"]
    contract: Literal["noesis.ws.health"]
    contract_version: Literal[2]
    deployment_id: Identifier
    selector_sha256: Sha256
    state_release_id: Identifier
    runtime_family: Literal["ds8", "ds9"]
    runtime_variant: str = Field(min_length=1, max_length=128)
    instance_id: str = Field(min_length=1, max_length=160)
    run_id: str = Field(min_length=1, max_length=160)
    boot_id: Identifier
    software_revision: GitOid
    generated_at_us: SafePositiveInteger
    ready: Literal[True]

    @field_validator("runtime_variant", "instance_id", "run_id")
    @classmethod
    def _identity_text_is_canonical(cls, value: str) -> str:
        return _canonical_text(value, maximum=160)

    @model_validator(mode="after")
    def _variant_matches_family(self) -> "WebSocketDeploymentHealth":
        if not self.runtime_variant.startswith(f"{self.runtime_family}:"):
            raise ValueError("runtime variant must match the runtime family")
        return self


__all__ = [
    "ApplianceEndpoints",
    "ApplianceReadinessVersions",
    "CheckoutBinding",
    "DeploymentHealth",
    "DeploymentSelector",
    "DS8RuntimeSelector",
    "DS9RuntimeSelector",
    "StateBaseline",
    "StateBaselineBinding",
    "StateBaselineFile",
    "StateMigration",
    "StateRelease",
    "StateReleaseBinding",
    "RuntimeDeploymentContext",
    "WebSocketDeploymentHealth",
]

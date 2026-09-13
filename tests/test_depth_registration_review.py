from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from noesis.calibration.depth_registration import (
    REGISTRATION_SCOPE, SOURCE_SPACE, TARGET_SPACE, TRANSFORM_TYPE_PIECEWISE,
    DepthRegistrationEntry,
)
from noesis.calibration.depth_registration_builder import build_registration_entry
from noesis.calibration.depth_registration_review import (
    SAMPLE_CONTRACT, ReviewPolicy, main, mapping_review, review_registration,
)


def _entry() -> DepthRegistrationEntry:
    return DepthRegistrationEntry(
        camera_id="room", created_ts_us=1, transform_type=TRANSFORM_TYPE_PIECEWISE,
        source_space=SOURCE_SPACE, target_space=TARGET_SPACE, scope=REGISTRATION_SCOPE,
        raw_range_domain_m=(1.0, 5.0), knots_raw_m=(1.0, 2.0, 3.0, 4.0, 5.0),
        knots_registered_m=(1.0, 2.0, 3.0, 4.0, 5.0), calibration_fingerprint={},
        dav2_profile={}, mapanything_profile={}, fit_metrics={}, sample_counts={},
        generation_tool_version="fixture", provenance={},
    )


def _samples(entry: DepthRegistrationEntry) -> dict:
    rows = []
    for partition in ("fit", "holdout"):
        for capture in ("a", "b"):
            for index, raw in enumerate(np.linspace(*entry.raw_range_domain_m, 16)):
                rows.append({
                    "sample_id": f"{partition}-{capture}-{index}", "partition": partition,
                    "capture_id": f"{partition}-{capture}", "recording_id": f"{partition}-{capture}",
                    "source_frame_id": index, "raw_timestamp_us": index * 100_000,
                    "reference_timestamp_us": index * 100_000, "raw_depth_m": float(raw),
                    "target_range_m": float(raw), "anchor_support_count": 8,
                    "anchor_id": "observed-person-anchor", "reference_anchor_id": "observed-person-anchor",
                    "reference_kind": "independent_same_anchor_range",
                    "reference_evidence_ref": "fixtures/independent_ranges.json",
                    "reference_evidence_sha256": "a" * 64,
                })
    return {"contract": SAMPLE_CONTRACT, "camera_id": entry.camera_id,
            "registration_id": entry.registration_id, "source_space": SOURCE_SPACE,
            "target_space": TARGET_SPACE, "samples": rows}


def test_independent_heldout_samples_qualify_review_only() -> None:
    entry = _entry()
    evidence = _samples(entry)
    original = copy.deepcopy(evidence)
    result = review_registration(entry, evidence=evidence)
    assert result["qualification_ready"] is True
    assert result["readiness_scope"] == "offline_review_only_no_runtime_admission"
    heldout = result["qualification"]["partitions"]["holdout"]
    assert heldout["sample_count"] == 32
    assert heldout["capture_count"] == 2
    assert heldout["occupied_range_bins"] == [0, 1, 2, 3]
    assert heldout["usable_residuals"]["max_abs_error_m"] == 0
    assert evidence == original


def test_existing_builder_output_is_reviewed_without_refitting() -> None:
    raw = np.linspace(1, 5, 512)
    entry = build_registration_entry(camera_id="room", calibration_fingerprint={}, dav2_profile={},
                                     mapanything_profile={}, provenance={}, raw_depth_m=raw,
                                     registered_depth_m=raw, created_ts_us=1)
    before = entry.to_dict()
    result = review_registration(entry, evidence=_samples(entry))
    assert result["qualification_ready"] is True
    assert result["qualification"]["partitions"]["holdout"]["mapped_residuals"]["count"] == 32
    assert entry.to_dict() == before


def test_small_residuals_cannot_qualify_a_flat_mapping() -> None:
    entry = replace(_entry(), knots_registered_m=(1.0, 2.0, 2.0, 2.0, 2.0))
    evidence = _samples(entry)
    for row in evidence["samples"]:
        row["target_range_m"] = min(row["raw_depth_m"], 2.0)
    result = review_registration(entry, evidence=evidence)
    assert result["qualification_ready"] is False
    assert result["mapping"]["observable_domain_fraction"] == 0.25
    assert "mapping_observable_domain_fraction_insufficient" in result["reasons"]
    assert "holdout_usable_fraction_insufficient" in result["reasons"]
    assert result["qualification"]["partitions"]["holdout"]["mapped_residuals"]["max_abs_error_m"] == 0


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("reference_kind", "tracking_world", "independent_reference_label_unsupported"),
    ("reference_kind", "phone_camera_center", "independent_reference_label_unsupported"),
    ("reference_evidence_sha256", "", "independent_reference_provenance_missing"),
    ("reference_anchor_id", "another-person", "reference_anchor_mismatch"),
    ("reference_timestamp_us", 10_000_000, "anchor_timestamp_mismatch"),
    ("anchor_support_count", 1, "anchor_support_insufficient"),
    ("raw_depth_m", float("nan"), "raw_depth_m_invalid"),
    ("target_range_m", True, "target_range_m_invalid"),
])
def test_bad_labels_fail_closed(field: str, value: object, reason: str) -> None:
    entry = _entry()
    samples = _samples(entry)
    samples["samples"][0][field] = value
    result = review_registration(entry, evidence=samples)
    assert result["qualification_ready"] is False
    assert result["qualification"]["rejected_sample_count"] == 1
    assert result["qualification"]["rejection_counts"][reason] == 1


@pytest.mark.parametrize("field", ["camera_id", "registration_id", "source_space", "target_space", "contract"])
def test_sample_identity_mismatch_cannot_qualify(field: str) -> None:
    samples = _samples(_entry())
    samples[field] = "wrong"
    result = review_registration(_entry(), evidence=samples)
    assert result["qualification_ready"] is False
    assert field + "_mismatch" in result["reasons"]


def test_duplicate_source_anchor_cannot_inflate_sample_count() -> None:
    entry = _entry()
    samples = _samples(entry)
    row = dict(samples["samples"][0], sample_id="new-id", partition="holdout")
    samples["samples"].append(row)
    result = review_registration(entry, evidence=samples)
    assert result["qualification_ready"] is False
    assert result["qualification"]["rejection_counts"]["duplicate_sample_or_source_anchor"] == 1


def test_holdout_cannot_extend_beyond_declared_fit_samples() -> None:
    entry = _entry()
    samples = _samples(entry)
    for row in samples["samples"]:
        if row["partition"] == "fit":
            row["raw_depth_m"] = row["target_range_m"] = 2.5
    result = review_registration(entry, evidence=samples)
    assert result["qualification_ready"] is False
    assert "holdout_requires_extrapolation_from_fit" in result["reasons"]


def test_different_capture_names_do_not_conceal_recording_time_leakage() -> None:
    entry = _entry()
    samples = _samples(entry)
    for row in samples["samples"]:
        row["recording_id"] = "same-recording"
        row["source_frame_id"] += 1000 * (1 + ["fit-a", "fit-b", "holdout-a", "holdout-b"].index(row["capture_id"]))
    result = review_registration(entry, evidence=samples)
    assert result["qualification_ready"] is False
    assert "fit_holdout_recording_time_overlap" in result["reasons"]


@pytest.mark.parametrize("mutation,reason", [
    ("range", "holdout_range_coverage_insufficient"),
    ("time", "holdout_capture_time_coverage_insufficient"),
    ("residual", "holdout_max_abs_error_m_unqualified"),
    ("out_of_domain", "holdout_usable_fraction_insufficient"),
])
def test_heldout_quality_and_coverage_gates(mutation: str, reason: str) -> None:
    entry = _entry()
    samples = _samples(entry)
    for row in samples["samples"]:
        if row["partition"] != "holdout":
            continue
        if mutation == "range":
            row["raw_depth_m"] = row["target_range_m"] = 2.5
        elif mutation == "time":
            row["raw_timestamp_us"] //= 100
            row["reference_timestamp_us"] //= 100
        elif mutation == "residual":
            row["target_range_m"] += 3
        else:
            row["raw_depth_m"] += 10
    result = review_registration(entry, evidence=samples)
    assert result["qualification_ready"] is False
    assert reason in result["reasons"]


def test_telemetry_retains_domain_diagnostics_without_creating_labels() -> None:
    entry = replace(_entry(), knots_registered_m=(1.0, 2.0, 2.0, 2.0, 2.0))
    rows = [{"depth_anchor_m": value, "depth_registration_status": "ok"} for value in (0.5, 1.5, 2.5, None)]
    result = review_registration(entry, tracking_rows=rows)
    assert result["qualification_ready"] is False
    assert result["tracking_diagnostics"]["raw_anchor_status_counts"] == {
        "out_of_domain": 1, "usable": 1, "unobservable_plateau": 1, "missing_or_invalid_raw_depth": 1}
    assert result["tracking_diagnostics"]["qualification_label_status"] == "unsupported_independent_reference_labels"


def test_cli_reports_missing_samples_and_never_overwrites_input(tmp_path: Path) -> None:
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({"cameras": {"room": _entry().to_dict()}}))
    original = registration.read_bytes()
    output = tmp_path / "report.json"
    args = ["--registration", str(registration), "--camera", "room", "--output", str(output)]
    assert main(args) == 1
    report = json.loads(output.read_text())
    assert report["qualification_ready"] is False
    assert len(report["input_evidence"]["registration"]["sha256"]) == 64
    with pytest.raises(SystemExit) as exc:
        main(args[:-1] + [str(registration)])
    assert exc.value.code == 2
    assert registration.read_bytes() == original


def test_cli_qualified_sample_document_returns_success(tmp_path: Path) -> None:
    entry = _entry()
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({"cameras": {"room": entry.to_dict()}}))
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps(_samples(entry)))
    output = tmp_path / "report.json"
    assert main(["--registration", str(registration), "--camera", "room", "--samples", str(samples),
                 "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["qualification_ready"] is True
    assert report["qualification"]["reference_evidence"][0]["ref"] == "fixtures/independent_ranges.json"


def test_cli_reads_companion_tracking_and_filters_identity(tmp_path: Path) -> None:
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({"cameras": {"room": _entry().to_dict()}}))
    tracking = tmp_path / "tracking.ndjson"
    tracking.write_text(json.dumps({"message_type": "tracking", "message": {"tracks": [
        {"camera_id": "room", "tracker_id": 7, "depth_anchor_m": 2.0},
        {"camera_id": "room", "tracker_id": 8, "depth_anchor_m": 2.0},
    ]}}) + "\n")
    output = tmp_path / "report.json"
    assert main(["--registration", str(registration), "--camera", "room", "--tracking", str(tracking),
                 "--tracker-id", "7", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["tracking_diagnostics"]["row_count"] == 1


def test_cli_input_size_bound_is_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from noesis.calibration import depth_registration_review as module
    monkeypatch.setattr(module, "MAX_INPUT_BYTES", 8)
    registration = tmp_path / "registration.json"
    registration.write_text(" " * 9)
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit) as exc:
        main(["--registration", str(registration), "--camera", "room", "--output", str(output)])
    assert exc.value.code == 2
    assert not output.exists()


def test_cli_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    registration = tmp_path / "registration.json"
    registration.write_text('{"cameras": {}, "cameras": {}}')
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit) as exc:
        main(["--registration", str(registration), "--camera", "room", "--output", str(output)])
    assert exc.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("policy", [{"range_bins": 0}, {"min_usable_fraction": float("nan")}, {"range_bins": 2.5}])
def test_invalid_review_policy_rejected(policy: dict) -> None:
    with pytest.raises(ValueError):
        ReviewPolicy(**policy)


def test_nonfinite_domain_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        mapping_review(replace(_entry(), raw_range_domain_m=(float("nan"), 5.0)))

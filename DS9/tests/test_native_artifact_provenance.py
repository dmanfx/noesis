from __future__ import annotations

import hashlib
import os
import sysconfig
from pathlib import Path

import pytest
import yaml

from DS9.noesis import native_artifact_provenance as provenance


def _source_digest(sources: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for label, payload in sorted(sources.items()):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _fixture(
    tmp_path: Path,
    module_name: str,
) -> tuple[Path, Path, dict[str, Path], Path]:
    repo_root = tmp_path / "repo"
    ds9_root = repo_root / "DS9"
    native_dir = ds9_root / "native_extensions"
    native_dir.mkdir(parents=True, exist_ok=True)
    contract = provenance._NATIVE_CONTRACTS[module_name]
    source_payloads: dict[str, bytes] = {}
    source_paths: dict[str, Path] = {}
    for index, label in enumerate(contract.sources):
        payload = f"reviewed-source-{index}\n".encode()
        path = repo_root / label
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        source_payloads[label] = payload
        source_paths[label] = path
    output_payload = b"reviewed-native-extension"
    suffix = str(sysconfig.get_config_var("EXT_SUFFIX"))
    output = native_dir / f"{module_name}{suffix}"
    output.write_bytes(output_payload)
    artifact = {
        "id": contract.artifact_id,
        "kind": "native_extension",
        "role": "test-only reviewed native extension",
        "output": contract.output,
        "sources": list(contract.sources),
        "builder": contract.builder,
        "required_profiles": list(contract.required_profiles),
        "state": "staged_unverified",
        "compatibility": {
            "deepstream_major": 9,
            "cuda": "13.1",
            "tensorrt": None,
        },
        "provenance": {
            "source_sha256": _source_digest(source_payloads),
            "output_sha256": hashlib.sha256(output_payload).hexdigest(),
            "built_at_utc": "2026-07-11T00:00:00Z",
            "build_host": "test-ds9-build-image",
            "command": "test-only reviewed build command",
        },
    }
    supplemental = module_name in provenance._SUPPLEMENTAL_NATIVE_MODULES
    manifest = {
        "schema_version": 2,
        "manifest_id": "noesis-ds9-artifacts",
        "schema": "DS9/docs/asset_manifest.schema.json",
        "target": {
            "platform": "linux-x86_64-dgpu",
            "python": "3.12",
            "deepstream": {
                "major": 9,
                "version": "9.0",
                "home": "/opt/nvidia/deepstream/deepstream-9.0",
            },
            "cuda": "13.1",
            "tensorrt": "10.14.1.48",
        },
        "policy": {
            "ds8_binary_reuse": "forbidden",
            "root_engine_reuse": "forbidden",
            "root_native_extension_reuse": "forbidden",
            "require_ds9_owned_outputs": True,
            "require_provenance_for_validated": True,
            "allowed_states": ["missing", "staged_unverified", "validated"],
            "output_roots": ["DS9/native_extensions"],
            "source_roots": ["DS9/native"],
        },
        "artifacts": [] if supplemental else [artifact],
    }
    manifest_path = ds9_root / "asset_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    if supplemental:
        manifest_payload = manifest_path.read_bytes()
        native_manifest = {
            "schema_version": 1,
            "contract": "noesis.ds9.native_artifact_manifest",
            "base_manifest": {
                "path": "DS9/asset_manifest.yaml",
                "sha256": hashlib.sha256(manifest_payload).hexdigest(),
            },
            "artifacts": [artifact],
        }
        (ds9_root / provenance._SUPPLEMENTAL_NATIVE_MANIFEST).write_text(
            yaml.safe_dump(native_manifest, sort_keys=False),
            encoding="utf-8",
        )
    return ds9_root, native_dir, source_paths, output


def _attest(ds9_root: Path, native_dir: Path, module_name: str) -> dict[str, Path]:
    return provenance.attest_ds9_native_artifacts(
        ds9_root=ds9_root,
        native_dir=native_dir,
        module_names=(module_name,),
    )


def test_exact_content_passes_when_binary_mtime_is_older_than_source(
    tmp_path: Path,
) -> None:
    module_name = "noesis_depth_meta_ext"
    ds9_root, native_dir, sources, output = _fixture(tmp_path, module_name)
    os.utime(output, (1, 1))
    for source in sources.values():
        os.utime(source, (2, 2))

    assert _attest(ds9_root, native_dir, module_name) == {module_name: output}


def test_analytics_extension_uses_manifest_bound_supplement(
    tmp_path: Path,
) -> None:
    module_name = "noesis_analytics_meta_ext"
    ds9_root, native_dir, _sources, output = _fixture(tmp_path, module_name)

    assert _attest(ds9_root, native_dir, module_name) == {module_name: output}

    manifest_path = ds9_root / "asset_manifest.yaml"
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    with pytest.raises(
        provenance.DS9NativeArtifactProvenanceError,
        match="authority does not match the base manifest",
    ):
        _attest(ds9_root, native_dir, module_name)


def test_missing_supplemental_artifact_fails_before_binary_lookup(
    tmp_path: Path,
) -> None:
    module_name = "noesis_analytics_meta_ext"
    ds9_root, native_dir, _sources, output = _fixture(tmp_path, module_name)
    output.unlink()
    native_manifest_path = ds9_root / provenance._SUPPLEMENTAL_NATIVE_MANIFEST
    native_manifest = yaml.safe_load(native_manifest_path.read_text(encoding="utf-8"))
    artifact = native_manifest["artifacts"][0]
    artifact["state"] = "missing"
    artifact["provenance"] = {
        "source_sha256": artifact["provenance"]["source_sha256"],
        "output_sha256": "pending_ds9_1_rebuild",
        "built_at_utc": None,
        "build_host": None,
        "command": "bash DS9/scripts/build_noesis_analytics_meta_ext.sh",
    }
    native_manifest_path.write_text(
        yaml.safe_dump(native_manifest, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(
        provenance.DS9NativeArtifactProvenanceError,
        match="native.analytics_meta is pending rebuild for the active DS9 SDK",
    ):
        _attest(ds9_root, native_dir, module_name)


@pytest.mark.parametrize("target", ["source", "output"])
def test_newer_mtime_cannot_hide_source_or_binary_drift(
    tmp_path: Path,
    target: str,
) -> None:
    module_name = "noesis_depth_meta_ext"
    ds9_root, native_dir, sources, output = _fixture(tmp_path, module_name)
    path = next(iter(sources.values())) if target == "source" else output
    path.write_bytes(path.read_bytes() + b"tamper")
    os.utime(path, None)

    with pytest.raises(
        provenance.DS9NativeArtifactProvenanceError,
        match=f"{target} SHA-256 mismatch",
    ):
        _attest(ds9_root, native_dir, module_name)


def test_depth_tensor_cuda_kernel_is_part_of_source_attestation(tmp_path: Path) -> None:
    module_name = "noesis_depth_tracking_tensor_ext"
    ds9_root, native_dir, sources, _output = _fixture(tmp_path, module_name)
    kernel = sources[
        "DS9/native/noesis_depth_tracking_tensor_kernels.cu"
    ]
    kernel.write_bytes(kernel.read_bytes() + b"tamper")

    with pytest.raises(
        provenance.DS9NativeArtifactProvenanceError,
        match="source SHA-256 mismatch",
    ):
        _attest(ds9_root, native_dir, module_name)


def test_multiple_or_wrong_abi_outputs_fail_closed(tmp_path: Path) -> None:
    module_name = "noesis_depth_meta_ext"
    ds9_root, native_dir, _sources, output = _fixture(tmp_path, module_name)
    duplicate = native_dir / f"{module_name}.duplicate.so"
    duplicate.write_bytes(output.read_bytes())

    with pytest.raises(
        provenance.DS9NativeArtifactProvenanceError,
        match="exactly one active-ABI output",
    ):
        _attest(ds9_root, native_dir, module_name)

    duplicate.unlink()
    wrong_abi = native_dir / f"{module_name}.cpython-311-x86_64-linux-gnu.so"
    output.rename(wrong_abi)
    with pytest.raises(
        provenance.DS9NativeArtifactProvenanceError,
        match="exactly one active-ABI output",
    ):
        _attest(ds9_root, native_dir, module_name)


def test_duplicate_artifact_or_yaml_key_fails_closed(tmp_path: Path) -> None:
    module_name = "noesis_depth_meta_ext"
    ds9_root, native_dir, _sources, _output = _fixture(tmp_path, module_name)
    manifest_path = ds9_root / "asset_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"].append(dict(manifest["artifacts"][0]))
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(
        provenance.DS9NativeArtifactProvenanceError,
        match="duplicate artifact id",
    ):
        _attest(ds9_root, native_dir, module_name)

    _fixture(tmp_path, module_name)
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + "\nartifacts: []\n",
        encoding="utf-8",
    )
    with pytest.raises(
        provenance.DS9NativeArtifactProvenanceError,
        match="duplicate key 'artifacts'",
    ):
        _attest(ds9_root, native_dir, module_name)


def test_incompatible_manifest_authority_or_incomplete_provenance_fails(
    tmp_path: Path,
) -> None:
    module_name = "noesis_depth_meta_ext"
    ds9_root, native_dir, _sources, _output = _fixture(tmp_path, module_name)
    manifest_path = ds9_root / "asset_manifest.yaml"

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 3
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(
        provenance.DS9NativeArtifactProvenanceError,
        match="schema authority",
    ):
        _attest(ds9_root, native_dir, module_name)

    _fixture(tmp_path, module_name)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["policy"]["root_native_extension_reuse"] = "allowed"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(
        provenance.DS9NativeArtifactProvenanceError,
        match="policy authority",
    ):
        _attest(ds9_root, native_dir, module_name)

    _fixture(tmp_path, module_name)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    del manifest["artifacts"][0]["provenance"]["command"]
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(
        provenance.DS9NativeArtifactProvenanceError,
        match="provenance is incomplete",
    ):
        _attest(ds9_root, native_dir, module_name)


@pytest.mark.parametrize("target", ["source", "output"])
def test_symlinked_source_or_output_fails_closed(tmp_path: Path, target: str) -> None:
    module_name = "noesis_depth_meta_ext"
    ds9_root, native_dir, sources, output = _fixture(tmp_path, module_name)
    path = next(iter(sources.values())) if target == "source" else output
    outside = tmp_path / f"outside-{target}"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)

    with pytest.raises(
        provenance.DS9NativeArtifactProvenanceError,
        match="symlink",
    ):
        _attest(ds9_root, native_dir, module_name)


def test_identity_replacement_after_hashing_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "noesis_depth_meta_ext"
    ds9_root, native_dir, sources, _output = _fixture(tmp_path, module_name)
    source = next(iter(sources.values()))
    original = provenance._assert_identity
    replaced = False

    def replace_before_check(
        path: Path,
        expected: provenance._FileIdentity,
        *,
        label: str,
    ) -> None:
        nonlocal replaced
        if path == source and not replaced:
            replaced = True
            path.write_bytes(path.read_bytes())
        original(path, expected, label=label)

    monkeypatch.setattr(provenance, "_assert_identity", replace_before_check)
    with pytest.raises(
        provenance.DS9NativeArtifactProvenanceError,
        match="changed during attestation",
    ):
        _attest(ds9_root, native_dir, module_name)


def test_path_disappearance_after_descriptor_read_uses_provenance_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"reviewed")
    original = Path.lstat
    calls = 0

    def disappear_on_final_lstat(self: Path):
        nonlocal calls
        if self == path:
            calls += 1
            if calls == 2:
                raise FileNotFoundError(path)
        return original(self)

    monkeypatch.setattr(Path, "lstat", disappear_on_final_lstat)
    with pytest.raises(
        provenance.DS9NativeArtifactProvenanceError,
        match="path disappeared while it was read",
    ):
        provenance._read_stable_regular(
            path,
            root=tmp_path,
            label="fixture artifact",
            maximum_bytes=1024,
        )

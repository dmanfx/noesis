from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = REPO_ROOT / "DS9" / "noesis" / "runtime_paths.py"
PREFLIGHT_PATH = REPO_ROOT / "DS9" / "scripts" / "ds9_preflight.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runtime_paths = _load(HELPER_PATH, "ds9_runtime_paths_origin_test")


def _module_file(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    path.write_bytes(b"fixture")
    return path


def _layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = tmp_path / "repo"
    ds9_root = repo_root / "DS9"
    native_dir = ds9_root / "native_extensions"
    native_dir.mkdir(parents=True)
    monkeypatch.setenv("NOESIS_NATIVE_EXT_DIR", str(native_dir))
    return repo_root, ds9_root, native_dir


def test_duplicate_root_and_native_modules_select_ds9_after_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root, ds9_root, native_dir = _layout(tmp_path, monkeypatch)
    name = "noesis_pose_meta_ext"
    root_binary = _module_file(repo_root, name)
    native_binary = _module_file(native_dir, name)
    paths = [str(repo_root), str(native_dir), str(ds9_root)]

    with pytest.raises(
        runtime_paths.DS9NativeExtensionOriginError, match="selected module.*outside"
    ):
        runtime_paths.require_ds9_native_extension_origins(
            native_dir,
            module_names=(name,),
            search_path=paths,
            loaded_modules={},
        )

    configured = runtime_paths.configure_ds9_runtime_import_paths(
        ds9_root=ds9_root,
        repo_root=repo_root,
        path_entries=paths,
    )
    origins = runtime_paths.require_ds9_native_extension_origins(
        configured,
        module_names=(name,),
        search_path=paths,
        loaded_modules={},
    )

    assert paths[:3] == [str(native_dir), str(ds9_root), str(repo_root)]
    assert origins[name] == native_binary.resolve()
    assert origins[name] != root_binary.resolve()


def test_cached_root_module_is_rejected_even_when_search_selects_ds9(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root, _ds9_root, native_dir = _layout(tmp_path, monkeypatch)
    name = "noesis_reid_meta_ext"
    root_binary = _module_file(repo_root, name)
    _module_file(native_dir, name)

    with pytest.raises(
        runtime_paths.DS9NativeExtensionOriginError, match="cached module.*outside"
    ):
        runtime_paths.require_ds9_native_extension_origins(
            native_dir,
            module_names=(name,),
            search_path=[str(native_dir), str(repo_root)],
            loaded_modules={name: SimpleNamespace(__file__=str(root_binary))},
        )


def test_loader_rejects_cached_root_module_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root, _ds9_root, native_dir = _layout(tmp_path, monkeypatch)
    name = "noesis_reid_meta_ext"
    root_binary = _module_file(repo_root, name)
    _module_file(native_dir, name)
    monkeypatch.setitem(
        sys.modules, name, SimpleNamespace(__file__=str(root_binary))
    )

    with (
        mock.patch.object(runtime_paths.importlib, "import_module") as import_module,
        pytest.raises(
            runtime_paths.DS9NativeExtensionOriginError,
            match="cached module.*outside",
        ),
    ):
        runtime_paths.load_ds9_native_extensions(
            native_dir,
            module_names=(name,),
            search_path=[str(native_dir), str(repo_root)],
        )

    import_module.assert_not_called()


def test_loader_fails_closed_when_owned_binary_cannot_be_imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root, _ds9_root, native_dir = _layout(tmp_path, monkeypatch)
    name = "noesis_pose_meta_ext"
    _module_file(native_dir, name)
    monkeypatch.delitem(sys.modules, name, raising=False)

    with (
        mock.patch.object(
            runtime_paths.importlib,
            "import_module",
            side_effect=ImportError("undefined symbol: ds9_fixture"),
        ),
        pytest.raises(
            runtime_paths.DS9NativeExtensionOriginError,
            match="failed to import.*undefined symbol: ds9_fixture",
        ),
    ):
        runtime_paths.load_ds9_native_extensions(
            native_dir,
            module_names=(name,),
            search_path=[str(native_dir), str(repo_root)],
        )


def test_loader_revalidates_the_module_returned_by_importer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root, _ds9_root, native_dir = _layout(tmp_path, monkeypatch)
    name = "noesis_depth_meta_ext"
    native_binary = _module_file(native_dir, name)
    monkeypatch.delitem(sys.modules, name, raising=False)

    with mock.patch.object(
        runtime_paths.importlib,
        "import_module",
        return_value=SimpleNamespace(__file__=str(native_binary)),
    ) as import_module:
        origins = runtime_paths.load_ds9_native_extensions(
            native_dir,
            module_names=(name,),
            search_path=[str(native_dir), str(repo_root)],
        )

    import_module.assert_called_once_with(name)
    assert origins == {name: native_binary.resolve()}


def test_symlink_escaping_native_directory_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo_root, _ds9_root, native_dir = _layout(tmp_path, monkeypatch)
    name = "noesis_depth_meta_ext"
    outside = _module_file(tmp_path / "outside", name)
    selected = native_dir / outside.name
    selected.symlink_to(outside)

    with pytest.raises(
        runtime_paths.DS9NativeExtensionOriginError, match="selected module.*outside"
    ):
        runtime_paths.require_ds9_native_extension_origins(
            native_dir,
            module_names=(name,),
            search_path=[str(native_dir)],
            loaded_modules={},
        )


def test_nested_binary_is_not_directly_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo_root, _ds9_root, native_dir = _layout(tmp_path, monkeypatch)
    name = "noesis_latency_ext"
    nested = native_dir / "nested"
    _module_file(nested, name)

    with pytest.raises(
        runtime_paths.DS9NativeExtensionOriginError, match="selected module.*outside"
    ):
        runtime_paths.require_ds9_native_extension_origins(
            native_dir,
            module_names=(name,),
            search_path=[str(nested)],
            loaded_modules={},
        )


def test_missing_selected_origin_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo_root, _ds9_root, native_dir = _layout(tmp_path, monkeypatch)
    with mock.patch.object(
        runtime_paths.PathFinder,
        "find_spec",
        return_value=SimpleNamespace(origin=None),
    ):
        with pytest.raises(
            runtime_paths.DS9NativeExtensionOriginError,
            match="selected module has no concrete binary origin",
        ):
            runtime_paths.require_ds9_native_extension_origins(
                native_dir,
                module_names=("noesis_v3dt_meta_ext",),
                search_path=[str(native_dir)],
                loaded_modules={},
            )


def test_exact_seven_module_set_is_shared_with_preflight() -> None:
    expected = (
        "noesis_pose_meta_ext",
        "noesis_analytics_meta_ext",
        "noesis_v3dt_meta_ext",
        "noesis_reid_meta_ext",
        "noesis_latency_ext",
        "noesis_depth_meta_ext",
        "noesis_depth_tracking_tensor_ext",
    )
    preflight = _load(PREFLIGHT_PATH, "ds9_preflight_native_origin_set_test")

    assert runtime_paths.DS9_NATIVE_EXTENSION_MODULES == expected
    assert preflight.DS9_NATIVE_EXTENSION_MODULES == expected


def test_preflight_reports_native_import_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    preflight = _load(PREFLIGHT_PATH, "ds9_preflight_native_load_failure_test")
    native_dir = tmp_path / "native_extensions"
    for module_name in runtime_paths.DS9_NATIVE_EXTENSION_MODULES:
        _module_file(native_dir, module_name)
    failure = preflight.DS9NativeExtensionOriginError(
        "noesis_pose_meta_ext failed to import: undefined symbol"
    )
    monkeypatch.setattr(
        preflight,
        "configure_ds9_runtime_import_paths",
        lambda **_kwargs: native_dir,
    )
    attest = mock.Mock()
    monkeypatch.setattr(preflight, "attest_ds9_native_artifacts", attest)
    load = mock.Mock(side_effect=failure)
    monkeypatch.setattr(preflight, "load_ds9_native_extensions", load)

    assert not preflight._native_extensions_ok()

    attest.assert_called_once_with(ds9_root=preflight.DS9_ROOT, native_dir=native_dir)
    load.assert_called_once_with(native_dir)
    stderr = capsys.readouterr().err
    assert "load/origin validation failed" in stderr
    assert "undefined symbol" in stderr

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from noesis.server import world_read_api
from noesis.server.internal_auth import configure_internal_rest_app, load_internal_token
from noesis_core.contracts.base import ArtifactFingerprint, ProducerRef
from noesis_core.world_service import CanonicalWorldService, WorldArtifacts

ROOT = Path(__file__).resolve().parents[2]


def _tree():
    return ast.parse((ROOT / "DS9/noesis/ds9_runtime_core.py").read_text())


def _execute(nodes, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "actual-ds91-world-read-binding", "exec"), namespace)


def test_actual_rest_builder_includes_private_router_under_existing_bearer_middleware(tmp_path, monkeypatch):
    tree = _tree(); builder = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_build_rest_app")
    imports = [node for node in ast.walk(builder) if isinstance(node, ast.ImportFrom) and node.module == "noesis.server"]
    assert any(any(alias.name == "world_read_api" for alias in node.names) for node in imports)
    include = next(node for node in builder.body if isinstance(node, ast.Expr)
        and ast.unparse(node) == "app.include_router(world_read_api.router)")
    auth = next(node for node in builder.body if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "configure_internal_rest_app")
    assert builder.body.index(include) < builder.body.index(auth)
    token_path = tmp_path / "private-world-read" / "gateway-token"
    monkeypatch.setenv("NOESIS_INTERNAL_AUTH_MODE", "required"); monkeypatch.setenv("NOESIS_INTERNAL_AUTH_TOKEN_FILE", str(token_path))
    app = FastAPI(); namespace = {"app": app, "world_read_api": world_read_api, "configure_internal_rest_app": configure_internal_rest_app}
    _execute([include, auth], namespace)
    token = load_internal_token(token_path)
    with TestClient(app) as client:
        assert client.get(world_read_api.PATH).status_code == 401
        result = client.get(world_read_api.PATH, headers={"Authorization": "Bearer " + token})
        assert result.status_code == 200 and result.json()["reason"] == "world_unbound"
        assert result.headers["cache-control"] == "private, no-store"


def test_actual_startup_branch_binds_exact_world_owner_once_and_closed_owner_becomes_unknown():
    tree = _tree()
    bindings = [node for node in ast.walk(tree) if isinstance(node, ast.If)
        and ast.unparse(node.test) == "prebuilt_rest_app is not None"
        and len(node.body) == 1 and isinstance(node.body[0], ast.Assign)
        and ast.unparse(node.body[0]) == "prebuilt_rest_app.state.canonical_world_service = world_service"]
    assert len(bindings) == 1
    creations = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "world_service" for target in node.targets)
        and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "startup_transaction.acquire"]
    assert len(creations) == 1 and bindings[0].lineno > creations[0].lineno
    assert "create_runtime_world_service" in ast.unparse(creations[0])
    producer = ProducerRef(runtime="test", instance_id="binding-fixture", run_id="binding-run", software_revision="fixture")
    artifact = lambda role, char: ArtifactFingerprint(role=role, sha256=char * 64)
    service = CanonicalWorldService(producer=producer, clock_us=lambda: 2_000_000,
        artifacts=WorldArtifacts(calibration=artifact("calibration", "a"), model=artifact("model", "b"), config=artifact("config", "c")))
    app = FastAPI(); app.include_router(world_read_api.router)
    try:
        with TestClient(app) as client:
            assert client.get(world_read_api.PATH).json()["reason"] == "world_unbound"
            _execute(bindings, {"prebuilt_rest_app": app, "world_service": service})
            assert app.state.canonical_world_service is service
            assert client.get(world_read_api.PATH).json()["reason"] == "snapshot_unavailable"
            committed = service.publish(0, [], metadata={"camera_id": "fixture-camera"})
            result = client.get(world_read_api.PATH).json()
            assert result["status"] == "available" and result["snapshot"] == committed.snapshot.model_dump(mode="json")
            service.close(); assert client.get(world_read_api.PATH).json()["reason"] == "world_closed"
        _execute(bindings, {"prebuilt_rest_app": None, "world_service": service})
    finally: service.close()

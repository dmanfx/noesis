"""Actual SDK-neutral startup helper with real bounded floorplan registration."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pytest
import numpy as np
from noesis_core.active_floorplan import ActiveFloorplanRegistry

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'DS9/noesis/ds9_runtime_core.py'


def actual_prime():
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == '_prime_bound_bev_floorplans')
    namespace = {'Mapping': Mapping}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace[node.name]


def registered_provider(registry, calls):
    spec = importlib.util.spec_from_file_location('floorplan_fixture', ROOT/'tests/test_active_floorplan_registry.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)

    def provider(*, camera, cache_only, scene_prior_only):
        assert cache_only is True and scene_prior_only is True
        calls.append(camera)
        payload = fixture._pcf_payload()
        prior = 'sceneprior_' + camera + '_exact'
        payload.update(camera_id=camera, snapshot_id=prior, world_frame_revision=prior)
        payload['scene_prior_meta'].update(prior_id=prior, source_frame_revision=prior)
        registry.record_scene_prior(camera, payload, calibration_fingerprint=fixture._FINGERPRINT)
        return payload
    return provider


def test_bound_maps_exist_before_first_frame_without_a_client_request():
    cameras = {2: 'family-room', 0: 'living-room', 1: 'kitchen'}
    registry = ActiveFloorplanRegistry({c: c for c in cameras.values()})
    assert all(registry.bounds_for(c) is None for c in cameras.values())
    calls = []
    priors = SimpleNamespace(binding=lambda _: SimpleNamespace(include_floorplan_layers=True))
    result = actual_prime()(cameras, priors, registered_provider(registry, calls), registry.bounds_for)
    assert calls == list(result) == ['family-room', 'kitchen', 'living-room']
    assert all(registry.bounds_for(c) is not None for c in cameras.values())


@pytest.mark.parametrize('payload', [None, [], {'error': 'calibration_unavailable'}])
def test_invalid_bound_provider_is_not_a_config_bounds_fallback(payload):
    priors = SimpleNamespace(binding=lambda _: SimpleNamespace(include_floorplan_layers=True))
    with pytest.raises(RuntimeError, match='startup failed'):
        actual_prime()({2: 'family-room'}, priors, lambda **_: payload, lambda _: None)


def test_payload_without_registered_authority_is_not_accepted():
    priors = SimpleNamespace(binding=lambda _: SimpleNamespace(include_floorplan_layers=True))
    with pytest.raises(RuntimeError, match='registration unavailable'):
        actual_prime()({2: 'family-room'}, priors, lambda **_: {}, lambda _: None)


def test_actual_empty_frame_zero_is_paired_after_startup_priming():
    from noesis.telemetry.bev import BevRenderer
    spec = importlib.util.spec_from_file_location('bev_fixture', ROOT/'DS9/tests/test_bev_parity.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    registry = ActiveFloorplanRegistry({'family-room': 'family-room'})
    def renderer(ws):
        value = BevRenderer(ws, trails_cfg={'enabled': False}, smoothing_cfg={'enabled': False},
            frame='camera_local_ground_m', floorplan_bounds_provider=registry.bounds_for)
        value.h_cache.get = lambda *a, **k: np.eye(3)
        return value
    args = dict(camera_id='family-room', calib=fixture._calibration(), footpoints=[],
        timestamp_us=1_000_000, source_id=2, frame_id=0, observed_at_us=1_000_000,
        tracking_publication_sequence=0, tracking_outbound_submission_id=1)
    old_ws = fixture._Ws()
    assert renderer(old_ws).render_and_publish(**args).status == 'startup_pending'
    assert old_ws.messages == []  # Reproduces the real unpaired first cohort.
    priors = SimpleNamespace(binding=lambda _: SimpleNamespace(include_floorplan_layers=True))
    actual_prime()({2: 'family-room'}, priors, registered_provider(registry, []), registry.bounds_for)
    ws = fixture._Ws()
    ws.next_submission_id = 2
    receipt = renderer(ws).render_and_publish(**args)
    assert receipt.status == 'admitted'
    assert len(ws.messages) == 1 and ws.messages[0]['type'] == 'bev-frame'
    assert ws.messages[0]['cohort'] == dict(source_id=2, source_epoch=None, frame_id=0,
        observed_at_us=1_000_000, tracking_publication_sequence=0, tracking_outbound_submission_id=1)
    assert ws.messages[0]['footpoints'] == []


def test_no_prior_or_unbound_camera_does_not_acquire_new_authority():
    def forbidden(*args, **kwargs):
        raise AssertionError('unbound camera must not be composed')
    prime = actual_prime()
    assert prime({2: 'family-room'}, None, forbidden, forbidden) == ()
    for binding in (None, SimpleNamespace(include_floorplan_layers=False)):
        priors = SimpleNamespace(binding=lambda _: binding)
        assert prime({2: 'family-room'}, priors, forbidden, forbidden) == ()


def test_actual_startup_primes_before_websocket_and_aborts_on_failure():
    tree = ast.parse(SOURCE.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_run_main')
    call = next(n for n in ast.walk(main) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == '_prime_bound_bev_floorplans')
    ws = next(n for n in ast.walk(main) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name) and n.func.id == '_start_websocket_server')
    prepare = next(n for n in ast.walk(main) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == 'prepare'
                   and isinstance(n.func.value, ast.Name) and n.func.value.id == 'deepstream_pipeline')
    assert call.lineno < prepare.lineno < ws.lineno
    guarded = next(n for n in ast.walk(main) if isinstance(n, ast.Try)
                   and any(c is call for c in ast.walk(ast.Module(body=n.body, type_ignores=[]))))
    assert '_abort_startup' in ast.unparse(guarded.handlers[0])
    assert 'bound_bev_floorplan_startup_failed' in ast.unparse(guarded.handlers[0])

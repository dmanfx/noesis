"""Execute the actual scalar helper/person builder and original single loop."""
from __future__ import annotations

import ast
import copy
import logging
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

from noesis_core.object_observations import ObjectObservationService
from noesis.telemetry.world_contract_adapter import frame_temporal_contract

ROOT = Path(__file__).resolve().parents[2]


def actual_hook_loop():
    tree = ast.parse((ROOT / "DS9/noesis/pipelines/hooks.py").read_text())
    processor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_AnalyticsTelemetryProcessor")
    method = next(node for node in processor.body if isinstance(node, ast.FunctionDef) and node.name == "_handle_servicemaker_frame_admitted")
    loop = next(node for node in ast.walk(method) if isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and node.iter.id == "object_items")
    assert sum(isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and node.iter.id == "object_items" for node in ast.walk(method)) == 1
    stop = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name) and node.test.left.id == "raw")
    part = copy.deepcopy(loop); part.body = part.body[:stop + 1] + ast.parse("tracks.append(raw)").body
    wrapper = ast.parse("def exercise(self, frame_meta, camera_id, object_capture):\n tracks=[]\n object_items=frame_meta.object_items\n object_ordinal=0\n return tracks\n").body[0]
    wrapper.body.insert(-1, part)
    helpers = [copy.deepcopy(node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"_rect_to_bbox", "_bbox_center"}]
    builder = copy.deepcopy(next(node for node in processor.body if isinstance(node, ast.FunctionDef) and node.name == "_build_track_dict_servicemaker"))
    room_helper = copy.deepcopy(next(node for node in processor.body if isinstance(node, ast.FunctionDef) and node.name == "_object_room_from_servicemaker"))
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + helpers + [builder, room_helper, wrapper], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {"math": math, "Mapping": Mapping, "logger": logging.getLogger("object-hook-test"), "noesis_analytics_meta_ext": None,
        "time": time, "_increment_core_counter": lambda *args: None, "_record_core_stage_timing": lambda *args: None}
    exec(compile(module, str(ROOT / "DS9/noesis/pipelines/hooks.py"), "exec"), scope)
    return scope


def test_original_single_iterator_nonperson_handoff_preserves_exact_person_builder_and_original_temporal_contract():
    scope = actual_hook_loop()
    owner = ObjectObservationService("hook-native-run", clock_us=lambda: 1_250_000)
    try:
        processor = SimpleNamespace(_tracking_mode_is_v3dt=lambda: False, _analytics_unique_id=None)
        processor._build_track_dict_servicemaker = lambda obj, camera: scope["_build_track_dict_servicemaker"](processor, obj, camera)
        processor._object_room_from_servicemaker = lambda obj: scope["_object_room_from_servicemaker"](processor, obj)
        rect = SimpleNamespace(left=10., top=20., width=30., height=40.)
        person = SimpleNamespace(class_id=0, object_id=7, label="person", confidence=.91, rect_params=rect)
        detected = SimpleNamespace(class_id=56, object_id=9, label="chair", confidence=.86, rect_params=rect)
        tracked_only = SimpleNamespace(class_id=56, object_id=9, label="chair", confidence=-.1, rect_params=rect)
        class Once:
            def __init__(self): self.used = False
            def __iter__(self):
                assert not self.used, "SDK iterator was consumed twice"
                self.used = True
                yield from (person, detected, tracked_only)
        frame = SimpleNamespace(frame_number=44, buffer_pts=12345, pipeline_width=640, pipeline_height=480,
            source_width=1920, source_height=1080, object_items=Once())
        temporal = frame_temporal_contract(frame, observed_at_s=1.25)
        capture = owner.begin_frame(frame, camera_id="native-camera", source_id=2, source_epoch=3, frame_id=44, temporal=temporal)
        actual = scope["exercise"](processor, frame, "native-camera", capture)
        expected = scope["_build_track_dict_servicemaker"](processor, person, "native-camera")
        assert actual == [expected]
        assert expected["track_id"] == 7 and expected["class_id"] == 0 and expected["confidence"] == .91
        assert capture.submit()
        deadline = time.monotonic() + 2
        while owner._queue.unfinished_tasks and time.monotonic() < deadline: time.sleep(.002)
        assert not owner._queue.unfinished_tasks
        camera = owner.snapshot()["cameras"][0]
        assert camera["observed_at_us"] == temporal["observed_at_us"] == 1_250_000
        assert camera["media_pts_ns"] == 12345 and camera["source_epoch"] == 3
        assert camera["invalid_count"] == 1 and len(camera["observations"]) == 1
        assert camera["observations"][0]["ordinal"] == 1 and camera["observations"][0]["class_id"] == 56
    finally:
        owner.close()
        assert not owner._worker.is_alive()


def test_runtime_source_binds_same_transaction_owner_and_private_router_without_new_pipeline_elements():
    tree = ast.parse((ROOT / "DS9/noesis/ds9_runtime_core.py").read_text())
    source = ast.unparse(tree)
    assert "app.include_router(object_observations.router)" in source
    assert "install_object_observation_owner(startup_transaction, pipeline, prebuilt_rest_app, world_service.producer.run_id, track_service=object_track_service)" in source
    assert "close_object_observation_owner(pipeline, prebuilt_rest_app, object_observation_service)" in source

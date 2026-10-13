"""Focused safety/geometry/edit tests for the real offline producer-consumer boundary."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
from functools import partial
from http.server import ThreadingHTTPServer
import urllib.request
import urllib.error
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from noesis_core.scene_semantics import advisory_context, corners, diagnostics, export_accepted, seal_map, validate_map
from tools.spatial_scene_review.serve_review import ReviewStore, Handler


def payload():
    wall = {"element_id": "wall.0", "kind": "wall", "category": "wall", "label": "wall", "geometry": {
        "center_m": [0, 1.2, 0], "size_m": [4, 0, 2.4], "basis_row_major": [1,0,0,0,0,1,0,-1,0]}}
    obj = {"element_id": "object.0", "kind": "object", "category": "sofa", "label": "sofa", "geometry": {
        "center_m": [1,.5,2], "size_m": [2,1,1], "basis_row_major": [1,0,0,0,0,1,0,-1,0]},
        "review_state": "accepted", "review_actor": "test", "review_reason": "synthetic fixture"}
    return {"map_id": "test", "proposal_revision_sha256": "a"*64, "world_frame_revision": "revision.1",
            "world_transform_sha256": "b"*64, "source_artifacts": [{"role": "test", "path": "fixture", "sha256": "c"*64}],
            "prepared_to_backend_rotation": [1,0,0,0,0,1,0,-1,0], "conversion_note": "synthetic test only", "elements": [wall,obj]}


def entity():
    return {"entity_id": "recorded.test", "world_frame": "backend_world_m", "world_frame_revision": "revision.1",
            "world_transform_sha256": "b"*64, "position_state": "measured", "lifecycle": "present", "room_id": None,
            "observed_at_us": 100, "stale_after_us": 200, "position": {"x": 2.5, "y": 0, "z": 2}, "conflict": False,
            "position_quantity": "ground_footprint",
            "subject": {"generation": 4}, "continuity_segment_id": 22,
            "sources": [{"accepted": True, "world_frame": "backend_world_m", "position_state": "measured", "observed_at_us": 100,
                         "world_frame_revision": "revision.1", "world_transform_sha256": "b"*64}]}


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.scene = seal_map(payload())

    def test_sealed_roundtrip_and_extra_version_rejection(self):
        raw = self.scene.model_dump(mode="json")
        self.assertEqual(validate_map(raw), self.scene)
        for field, value in (("contract_version", 2), ("units", "feet"), ("coordinate_frame", "image_px"), ("authority", "live"), ("unexpected", 1)):
            changed = {**raw, field:value}
            with self.assertRaises(ValueError): validate_map(changed)

    def test_digest_tampering_rejected(self):
        changed = self.scene.model_dump(mode="json");changed["elements"][1]["geometry"]["center_m"][0] += 1
        with self.assertRaisesRegex(ValueError, "digest"): validate_map(changed)

    def test_nan_reflection_tilt_and_zero_object_width_rejected(self):
        for change in ("nan","reflection","tilt","zero"):
            raw=payload();g=raw["elements"][1]["geometry"]
            if change=="nan":g["center_m"][0]=float('nan')
            if change=="reflection":g["basis_row_major"][0]=-1
            if change=="tilt":g["basis_row_major"]=[1,0,0,0,1,0,0,0,1]
            if change=="zero":g["size_m"][1]=0
            with self.assertRaises(ValueError):seal_map(raw)

    def test_duplicate_ids_missing_wall_and_review_reason_rejected(self):
        raw=payload();raw["elements"].append(copy.deepcopy(raw["elements"][0]))
        with self.assertRaises(ValueError):seal_map(raw)
        raw=payload();raw["elements"][0].update(kind="opening",category="door",wall_id="missing")
        with self.assertRaises(ValueError):seal_map(raw)
        raw=payload();raw["elements"][1]["review_reason"]=""
        with self.assertRaises(ValueError):seal_map(raw)

    def test_all_corners_upright_and_ground_distance_preserves_entity(self):
        points=corners(self.scene.elements[1].geometry)
        self.assertEqual(len(points),8);self.assertEqual({p[1] for p in points},{0,1})
        e=entity();before=copy.deepcopy(e);result=advisory_context(self.scene,e,at_us=150)
        self.assertEqual(e,before);self.assertEqual(result["nearby"][0]["ground_footprint_distance_m"],.5)
        self.assertEqual(result["continuity_segment_id"],22);self.assertEqual(result["lifecycle_generation"],4)
        self.assertIsNone(result["room_id"])

    def test_synthetic_temporal_and_binding_cases_fail_closed(self):
        for field,value in (("position_state","predicted"),("position_state","held"),("lifecycle","held"),
                            ("lifecycle","lost"),("world_frame_revision","other"),("world_transform_sha256","d"*64),
                            ("conflict",True),("sources",[])):
            e=entity();e[field]=value
            self.assertEqual(advisory_context(self.scene,e,at_us=150)["status"],"unknown")
        for at in (99,200,300):self.assertEqual(advisory_context(self.scene,entity(),at_us=at)["status"],"unknown")

    def test_opening_overrun_is_detected_and_cannot_export_as_accepted(self):
        raw=payload();raw["elements"][0].update(review_state="accepted",review_actor="test",review_reason="test")
        opening={"element_id":"door.0","kind":"opening","category":"door","label":"door","wall_id":"wall.0",
                 "geometry":{"center_m":[2,1,0],"size_m":[.8,0,2],"basis_row_major":[1,0,0,0,0,1,0,-1,0]},
                 "review_state":"accepted","review_actor":"test","review_reason":"test"}
        raw["elements"].append(opening);scene=seal_map(raw)
        self.assertEqual(diagnostics(scene)[-1]["issues"][0]["code"],"opening_wall_mismatch")
        with self.assertRaisesRegex(ValueError,"unresolved"):export_accepted(scene)

    def test_export_excludes_pending_and_rejected(self):
        exported=export_accepted(self.scene);self.assertEqual(len(exported.elements),1)
        self.assertEqual(exported.elements[0].element_id,"object.0");validate_map(exported.model_dump(mode="json"))

    def test_actual_store_persistence_audit_conflict_import_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"drafts").mkdir();(root/"exports").mkdir()
            raw=self.scene.model_dump(mode="json")
            for name in ("proposals.json","reviewed.json"):(root/name).write_text(json.dumps(raw))
            store=ReviewStore(root);elements=copy.deepcopy(raw["elements"]);elements[1]["label"]="Renamed sofa"
            request={"base_revision":self.scene.revision_sha256,"elements":elements,"actor":"test reviewer","reason":"test rename"}
            saved=store.save(request);self.assertTrue(saved["saved"])
            self.assertEqual(ReviewStore(root).current().elements[1].label,"Renamed sofa")
            audit=json.loads(next((root/"drafts").glob('*.json')).read_text())["audit"]
            self.assertEqual(audit["changes"][0]["before"]["label"],"sofa")
            with self.assertRaises(RuntimeError):store.save(request)
            request["base_revision"]=saved["map"]["revision_sha256"];request["elements"]=raw["elements"]
            store.save(request);self.assertEqual(len(list((root/"drafts").glob('*.json'))),2)
            with self.assertRaises(ValueError):store.save({**request,"path":"/etc/test"})
            request["base_revision"]=store.current().revision_sha256;request["elements"]=raw["elements"][:1]
            with self.assertRaises(ValueError):store.save(request)
            self.assertTrue(store.export()["map"]["elements"])

    def test_actual_http_json_boundary_and_private_file_denial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"drafts").mkdir();(root/"exports").mkdir()
            for name in ("proposals.json","reviewed.json"):(root/name).write_text(json.dumps(self.scene.model_dump(mode="json")))
            (root/"recorded-context.json").write_text('[]')
            server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,store=ReviewStore(root),three_root=root))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            url=f'http://127.0.0.1:{server.server_port}'
            try:
                self.assertEqual(json.load(urllib.request.urlopen(url+'/api/state'))['map']['map_id'],'test')
                with self.assertRaises(urllib.error.HTTPError) as exc:urllib.request.urlopen(url+'/reviewed.json')
                self.assertEqual(exc.exception.code,404)
                elements=self.scene.model_dump(mode="json")["elements"];elements[1]["label"]='HTTP edited sofa'
                raw=json.dumps({'base_revision':self.scene.revision_sha256,'elements':elements,'actor':'test','reason':'HTTP roundtrip'}).encode()
                bad=urllib.request.Request(url+'/api/save',data=raw,headers={'Content-Type':'application/json'})
                with self.assertRaises(urllib.error.HTTPError) as exc:urllib.request.urlopen(bad)
                self.assertEqual(exc.exception.code,403)
                request=urllib.request.Request(url+'/api/save',data=raw,headers={'Content-Type':'application/json','Origin':url})
                self.assertTrue(json.load(urllib.request.urlopen(request))['saved'])
                for raw in (b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":1e309}'):
                    request=urllib.request.Request(url+'/api/save',data=raw,headers={'Content-Type':'application/json','Origin':url})
                    with self.assertRaises(urllib.error.HTTPError) as exc:urllib.request.urlopen(request)
                    self.assertEqual(exc.exception.code,400)
            finally:
                server.shutdown();server.server_close();thread.join(timeout=2)


if __name__ == "__main__":unittest.main()

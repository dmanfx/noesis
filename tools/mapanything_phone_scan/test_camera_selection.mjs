import assert from "node:assert/strict";
import test from "node:test";
import { cameraSelection, changeCameraSelection, cameraSelectionMarkup, nativeCameraSelection, nativeCameraIndices } from "./static/camera_selection.js";
import { NativeCaptureUI, NATIVE_UI_SCHEMA } from "./static/native_capture.js";

const cameras = ["family", "kitchen", "living", "offline"].map((camera_id) => ({ camera_id, label: camera_id, available: camera_id !== "offline" }));
const state = (overrides = {}) => cameraSelection({ cameras, maxCameras: 2, ...overrides });
const markup = (s) => cameraSelectionMarkup(s, { fieldsetId: "recording-cameras", primarySelectId: "primary-camera" });

test("checkbox set and primary reference are independent, including phone-only", () => {
  let s = changeCameraSelection(state(), { cameraId: "family", checked: true });
  assert.deepEqual(s.ids, ["family"]); assert.equal(s.primaryId, "family");
  s = changeCameraSelection(s, { cameraId: "kitchen", checked: true });
  assert.equal(s.primaryId, "family");
  s = changeCameraSelection(s, { primaryId: "kitchen" });
  assert.deepEqual(s.ids, ["family", "kitchen"]);
  assert.throws(() => changeCameraSelection(s, { primaryId: "living" }), /checked/);
  s = changeCameraSelection(s, { cameraId: "kitchen", checked: false });
  assert.deepEqual(s.ids, ["family"]); assert.equal(s.primaryId, "family");
  s = changeCameraSelection(s, { cameraId: "family", checked: false });
  assert.deepEqual(s.ids, []); assert.equal(s.primaryId, "");
  assert.match(markup(s), /No static camera · phone-only reconstruction/);
});

test("all inventory cameras are listed; max and offline block additions, not removals", () => {
  const s = state({ selectedIds: ["family", "kitchen"], primaryId: "family" });
  const html = markup(s);
  for (const c of cameras) assert.ok(html.includes(`data-camera-id="${c.camera_id}"`));
  assert.match(html, /server limit 2/); assert.match(html, /Recording limit reached/); assert.match(html, /Unavailable/);
  assert.throws(() => changeCameraSelection(s, { cameraId: "living", checked: true }), /At most 2/);
  assert.throws(() => changeCameraSelection(state(), { cameraId: "offline", checked: true }), /unavailable/);
  const offline = state({ cameras: cameras.map((c) => ({ ...c, available: false })), selectedIds: s.ids, primaryId: "family" });
  assert.doesNotMatch(markup(offline).match(/<input[^>]+data-camera-id="family"[^>]*>/)[0], /disabled/);
  assert.deepEqual(changeCameraSelection(offline, { cameraId: "family", checked: false }).ids, ["kitchen"]);
  const disappeared = state({ cameras: [], selectedIds: ["family"], primaryId: "family" });
  assert.match(markup(disappeared), /Unavailable/);
  assert.deepEqual(changeCameraSelection(disappeared, { cameraId: "family", checked: false }).ids, []);
});

test("recording locks every edit and advertised limits cannot imply third-camera support", () => {
  const s = state({ selectedIds: ["family"], primaryId: "family", locked: true });
  for (const action of [{ cameraId: "family", checked: false }, { cameraId: "kitchen", checked: true }, { primaryId: "family" }])
    assert.throws(() => changeCameraSelection(s, action), /locked/);
  assert.match(markup(s), /data-camera-primary[^>]+disabled/);
  const single = state({ maxCameras: 1, selectedIds: ["family"], primaryId: "family" });
  assert.throws(() => changeCameraSelection(single, { cameraId: "kitchen", checked: true }), /At most 1/);
  const future = state({ maxCameras: 3 });
  assert.equal(future.maximum, 2); assert.match(markup(future), /server limit 3 · this recorder supports 2/);
});

test("labels and IDs are escaped and primary options include checked cameras only", () => {
  const html = markup(state({ cameras: [{ camera_id: 'x"', label: "<script>", available: true }, ...cameras], selectedIds: ['x"'], primaryId: 'x"' }));
  assert.ok(!html.includes("<script>")); assert.match(html, /&quot;/);
  const options = html.slice(html.indexOf("<select"));
  assert.match(options, /value="x&quot;"/); assert.doesNotMatch(options, /value="kitchen"/);
});

const snapshot = (fields = {}) => ({ schema: NATIVE_UI_SCHEMA, cameras: [], room_cameras: cameras, saved: [], enabled: {},
  room_camera_index: 0, extra_room_camera_index: 1, supports_camera_ids: true, multi_camera_capture: true,
  server: "https://roomwalk.test", camera_index: -1, imu_minutes: 1, ...fields });

test("native checked set maps exact existing indices and phone-only without changing Java contract", () => {
  let s = nativeCameraSelection(snapshot());
  assert.deepEqual(s.ids, ["family", "kitchen"]);
  assert.deepEqual(nativeCameraIndices(s, cameras), { room_camera_index: 0, extra_room_camera_index: 1 });
  s = changeCameraSelection(s, { primaryId: "kitchen" });
  assert.deepEqual(nativeCameraIndices(s, cameras), { room_camera_index: 1, extra_room_camera_index: 0 });
  s = changeCameraSelection(s, { cameraId: "family", checked: false });
  s = changeCameraSelection(s, { cameraId: "kitchen", checked: false });
  assert.deepEqual(nativeCameraIndices(s, cameras), { room_camera_index: -1, extra_room_camera_index: -1 });
  assert.throws(() => nativeCameraIndices(nativeCameraSelection(snapshot({ room_camera_index: 3, extra_room_camera_index: -1 })), cameras), /unavailable/);
});

test("native saved selection reload and exact Apply ACK preserve dirty checkbox edits", () => {
  const ui = new NativeCaptureUI({ platform: { __ROOMWALK_ANDROID__: true, location: { href: "https://roomwalk.test/" } } });
  ui.receive(snapshot());
  assert.deepEqual(ui.roomSelection.ids, ["family", "kitchen"]);
  ui.roomSelection = changeCameraSelection(ui.roomSelection, { primaryId: "kitchen" });
  ui.configDirty = true;
  ui.pendingConfiguration = { server: "https://roomwalk.test", camera_index: -1, room_camera_index: 1, extra_room_camera_index: 0, imu_minutes: 1 };
  ui.receive(snapshot());
  assert.equal(ui.configDirty, true); assert.equal(ui.roomSelection.primaryId, "kitchen");
  ui.receive(snapshot({ room_camera_index: 1, extra_room_camera_index: 0 }));
  assert.equal(ui.configDirty, false); assert.equal(ui.pendingConfiguration, null);
  assert.equal(ui.roomSelection.primaryId, "kitchen"); assert.deepEqual(new Set(ui.roomSelection.ids), new Set(["family", "kitchen"]));
  ui.receive(snapshot({ active: true }));
  assert.throws(() => changeCameraSelection(ui.roomSelection, { cameraId: "family", checked: false }), /locked/);
  ui.receive(snapshot({ room_camera_index: -1, extra_room_camera_index: -1 }));
  assert.deepEqual(ui.roomSelection.ids, []);
});

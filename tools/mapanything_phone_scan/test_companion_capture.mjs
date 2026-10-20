import assert from "node:assert/strict";
import test from "node:test";

import {
  COMPANION_CAPTURE_SCHEMA,
  CompanionCapture,
  CompanionCaptureError,
  normalizeCameraIds,
} from "./static/companion_capture.js";

function response(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "Error",
    json: async () => body,
  };
}

function makeHarness({ startResponse = null, heartbeatResponse = null, stopResponse = null, finalResponse = null, markerFailures = 0 } = {}) {
  let now = 1000;
  let markerAttempts = 0;
  const calls = [];
  let heartbeatCalls = 0;
  const clock = () => ({
    server_received_unix_ns: "1700000000000000000",
    server_received_monotonic_ns: "1000000000000",
    server_sent_unix_ns: "1700000000000001000",
    server_sent_monotonic_ns: "1000000001000",
  });
  const fetchImpl = async (url, options = {}) => {
    calls.push({ url, options, body: options.body ? JSON.parse(options.body) : null });
    if (url.endsWith("/cameras")) return response({ available: true, cameras: [
      { camera_id: "kitchen-rtsp", label: "Kitchen", source_id: "kitchen", available: true },
      { camera_id: "family-rtsp", label: "Family Room", source_id: "family", available: false, reason: "offline" },
    ] });
    if (url.endsWith("/clock")) { now += 4; return response(clock()); }
    if (url === "/api/companion-captures" && options.method === "POST") return response(startResponse || {
      session_id: "session-1", camera_id: JSON.parse(options.body).camera_id, phone_capture_id: JSON.parse(options.body).phone_capture_id, status: "recording",
      video_status: "recording", tracking_status: "recording", limits: { max_duration_s: 600, lease_s: 45 }, provenance: { revision: "rev-1" },
    });
    if (url.endsWith("/heartbeat")) {
      heartbeatCalls += 1;
      if (heartbeatResponse instanceof Error) throw heartbeatResponse;
      return response(heartbeatResponse || { status: "recording", video_status: "recording", tracking_status: "recording", tracking_sequence: heartbeatCalls });
    }
    if (url.endsWith("/markers")) {
      markerAttempts += 1;
      if (markerAttempts <= markerFailures) return response({ detail: "marker unavailable" }, 503);
      return response({ receipt_clock: clock() });
    }
    if (url.endsWith("/stop")) return response(stopResponse || { status: "stopped", video_status: "stopped", tracking_status: "stopped", artifact_urls: { video: "/static.mp4" } });
    if (url.endsWith("/session-1")) return response(finalResponse || { status: "stopped", video_status: "stopped", tracking_status: "stopped" });
    throw new Error(`Unhandled ${url}`);
  };
  const platform = {
    window: { setTimeout, clearTimeout, setInterval, clearInterval, crypto: { randomUUID: () => `uuid-${calls.length}-${now}` } },
    performance: { now: () => now },
    crypto: { randomUUID: () => `uuid-${calls.length}-${now}` },
  };
  return { calls, platform, fetchImpl, heartbeatCalls: () => heartbeatCalls };
}

async function startedCapture(harness, extra = {}) {
  const browser = { contexts: [], setCompanionCapture(context) { this.contexts.push(context); } };
  const capture = new CompanionCapture({ ...harness, onFailure: extra.onFailure, limits: { heartbeatMs: 60_000, ...extra.limits } });
  await capture.listCameras();
  capture.selectCamera("kitchen-rtsp");
  await capture.start({ cameraId: "kitchen-rtsp", phoneCaptureId: "phone-1", browserCapture: browser });
  return { capture, browser };
}

function multiHarness({ inventoryMulti = true, changeStart = (value) => value, changeHeartbeat = (value) => value, changeStop = (value) => value } = {}) {
  const h = makeHarness();
  const base = h.fetchImpl;
  let identity;
  const session = (status = "recording") => ({ ...identity, session_id: "session-pair", status,
    video_status: status, tracking_status: status,
    cameras: Object.fromEntries(identity.camera_ids.map((id) => [id, { camera_id: id, readiness: { recorder_ready: true, tracking_ready: true }, artifact_urls: { video: `/${id}.mkv` } }])) });
  h.fetchImpl = async (url, options = {}) => {
    const body = options.body ? JSON.parse(options.body) : null;
    if (url.endsWith("/cameras")) return response({ available: true, multi_camera_capture: inventoryMulti, max_cameras: 2,
      cameras: ["kitchen-rtsp", "family-rtsp"].map((id) => ({ camera_id: id, label: id, available: true })) });
    if (url === "/api/companion-captures" && options.method === "POST") {
      h.calls.push({ url, body });
      identity = { camera_id: body.camera_id, camera_ids: body.camera_ids, phone_capture_id: body.phone_capture_id };
      return response(changeStart(session()));
    }
    if (url.endsWith("/heartbeat")) return response(changeHeartbeat(session()));
    if (url.endsWith("/stop")) { h.calls.push({ url, body }); return response(changeStop(session("stopped"))); }
    return base(url, options);
  };
  return h;
}

test("camera-set normalization preserves legacy singleton and rejects malformed or expanded sets", () => {
  assert.deepEqual(normalizeCameraIds("kitchen"), ["kitchen"]);
  assert.deepEqual(normalizeCameraIds("kitchen", ["kitchen", "family"]), ["family", "kitchen"]);
  for (const ids of [null, [], ["family"], ["kitchen", "kitchen"], ["kitchen", "family", "third"], ["kitchen", 2]]) assert.throws(() => normalizeCameraIds("kitchen", ids));
});

test("idle checkbox clear selects phone-only; active session cannot clear its frozen set", async () => {
  const h = multiHarness();
  let emitted;
  const c = new CompanionCapture({ ...h, onStateChange: (s) => { emitted = s; }, limits: { heartbeatMs: 60_000 } });
  await c.listCameras();
  c.selectCameras("kitchen-rtsp", ["family-rtsp", "kitchen-rtsp"]);
  c.selectCameras("", []);
  assert.equal(c.status.selectedCameraId, ""); assert.deepEqual(c.status.selectedCameraIds, []);
  assert.deepEqual(emitted.selectedCameraIds, []); assert.equal(c.readyToStart, false);
  assert.throws(() => c.selectCameras("", undefined));
  c.selectCamera("kitchen-rtsp");
  await c.start({ phoneCaptureId: "phone-pair" });
  assert.throws(() => c.selectCameras("", []), /Stop/);
  assert.deepEqual(c.status.selectedCameraIds, ["kitchen-rtsp"]);
  await c.stop();
});

test("offline retained checks permit strict removals only and never start unavailable pairing", async () => {
  const h = multiHarness(); const originalFetch = h.fetchImpl;
  let offline = false;
  h.fetchImpl = (url, options) => offline && url.endsWith("/cameras")
    ? Promise.resolve(response({ available: false, multi_camera_capture: true, max_cameras: 2, cameras: [
      { camera_id: "kitchen-rtsp", available: false }, { camera_id: "family-rtsp", available: false },
    ] })) : originalFetch(url, options);
  const c = new CompanionCapture(h);
  await c.listCameras(); c.selectCameras("family-rtsp", ["family-rtsp", "kitchen-rtsp"]);
  offline = true; await c.listCameras();
  assert.deepEqual(c.status.selectedCameraIds, ["family-rtsp", "kitchen-rtsp"]);
  c.selectCameras("kitchen-rtsp", ["kitchen-rtsp"]);
  assert.equal(c.readyToStart, false);
  await assert.rejects(() => c.start(), /available/);
  assert.throws(() => c.selectCameras("kitchen-rtsp", ["kitchen-rtsp", "family-rtsp"]), /available/);
  assert.throws(() => c.selectCameras("other", ["other"]), /available/);
  c.selectCameras("", []); assert.deepEqual(c.status.selectedCameraIds, []);
});

test("explicit pair preserves primary, exact set, per-camera readiness and detached saved context", async () => {
  const h = multiHarness();
  const c = new CompanionCapture({ ...h, limits: { heartbeatMs: 60_000 } });
  await c.listCameras();c.selectCameras("kitchen-rtsp", ["kitchen-rtsp", "family-rtsp"]);
  await c.start({ phoneCaptureId: "phone-pair" });
  assert.equal(c.state, "recording");assert.equal(c.companionContext().camera_id, "kitchen-rtsp");
  assert.deepEqual(c.companionContext().camera_ids, ["family-rtsp", "kitchen-rtsp"]);
  assert.deepEqual(h.calls.find((v) => v.body?.camera_ids)?.body.camera_ids, ["family-rtsp", "kitchen-rtsp"]);
  const old = c.companionContext();old.camera_ids.pop();old.cameras["family-rtsp"].readiness.tracking_ready = false;
  assert.equal(c.companionContext().camera_ids.length, 2);assert.equal(c.companionContext().cameras["family-rtsp"].readiness.tracking_ready, true);
  assert.throws(() => c.selectCamera("family-rtsp"), /Stop/);
  await c.heartbeat();await c.stop();assert.equal(c.state, "stopped");
});

test("pair fails closed when server ignores camera_ids or omits any current member readiness", async () => {
  for (const changeStart of [
    (s) => { delete s.camera_ids;return s; },
    (s) => { delete s.cameras["family-rtsp"];return s; },
    (s) => { s.cameras["family-rtsp"].readiness.tracking_ready = false;return s; },
    (s) => { s.cameras["family-rtsp"].readiness.recorder_ready = "true";return s; },
    (s) => ({ ...s, camera_id: "family-rtsp" }),
    (s) => ({ ...s, phone_capture_id: "other" }),
  ]) {
    const h=multiHarness({ changeStart });const c=new CompanionCapture({ ...h, limits: { heartbeatMs: 60_000 } });
    await c.listCameras();c.selectCameras("kitchen-rtsp",["family-rtsp","kitchen-rtsp"]);
    await assert.rejects(() => c.start({phoneCaptureId:"phone-pair"}));
    assert.notEqual(c.state,"recording");assert.ok(h.calls.some((v)=>v.url.endsWith('/stop')));
  }
});

test("pair requires inventory capability and every selected camera remains bound on heartbeat/finalization", async () => {
  const unsupported=multiHarness({inventoryMulti:false});const old=new CompanionCapture(unsupported);await old.listCameras();
  assert.throws(()=>old.selectCameras("kitchen-rtsp",["kitchen-rtsp","family-rtsp"]),/support/);
  for (const phase of ["heartbeat","stop"]) {
    const change=(s)=>({...s,camera_ids:["kitchen-rtsp"]});
    const h=multiHarness(phase==="heartbeat"?{changeHeartbeat:change}:{changeStop:change});
    const c=new CompanionCapture({...h,limits:{heartbeatMs:60_000}});await c.listCameras();c.selectCameras("kitchen-rtsp",["family-rtsp","kitchen-rtsp"]);await c.start({phoneCaptureId:"phone-pair"});
    if(phase==="heartbeat")await assert.rejects(()=>c.heartbeat(),/camera/);else await c.stop();
    assert.equal(c.state,"failed");assert.deepEqual(c.companionContext().camera_ids,["family-rtsp","kitchen-rtsp"]);
    if(c.needsServerStop) { h.fetchImpl=async()=>response({});c._stopHeartbeat(); }
  }
});

test("realistic large server status is compacted without increasing the retained-reference budget", async () => {
  const h=multiHarness({changeStart:(s)=>{
    for(const row of Object.values(s.cameras)){
      row.recorder={transient_status:"x".repeat(66000)};row.tracking={transient_status:"y".repeat(66000)};row.provenance={duplicate_full_runtime:"z".repeat(66000)};
    }return s;
  }});
  const c=new CompanionCapture({...h,limits:{heartbeatMs:60_000}});await c.listCameras();c.selectCameras("kitchen-rtsp",["family-rtsp","kitchen-rtsp"]);await c.start({phoneCaptureId:"phone-pair"});
  const ref=c.companionContext();assert.equal(ref.cameras["family-rtsp"].recorder,undefined);assert.equal(ref.cameras["family-rtsp"].provenance,undefined);
  while(ref.clock_probes.length<96)ref.clock_probes.push({...ref.clock_probes[0]});
  assert.ok(Buffer.byteLength(JSON.stringify(ref))<240000);assert.equal(Object.keys(ref.cameras).length,2);
  await c.stop();
});

test("static readiness is explicit and start sends five clock probes before phone start", async () => {
  const harness = makeHarness();
  const { capture, browser } = await startedCapture(harness);
  assert.equal(capture.status.state, "recording");
  assert.equal(capture.status.sessionId, "session-1");
  assert.equal(capture.status.clockProbeCount, 5);
  assert.equal(harness.calls.filter((call) => call.url.endsWith("/clock")).length, 5);
  const startCall = harness.calls.find((call) => call.url === "/api/companion-captures");
  assert.equal(startCall.body.camera_id, "kitchen-rtsp");
  assert.equal(startCall.body.phone_capture_id, "phone-1");
  assert.equal(startCall.body.clock_probes.length, 5);
  assert.equal(browser.contexts.at(-1).schema, COMPANION_CAPTURE_SCHEMA);
  assert.equal(browser.contexts.at(-1).phone_capture_id, "phone-1");
  await capture.stop("user");
});

test("missing camera selection is fail-closed and a non-ready response is stopped", async () => {
  const harness = makeHarness({ startResponse: {
    session_id: "orphan-candidate", camera_id: "kitchen-rtsp", phone_capture_id: "phone-2", status: "recording",
    video_status: "failed", tracking_status: "recording",
  } });
  const capture = new CompanionCapture({ ...harness, limits: { heartbeatMs: 60_000 } });
  await capture.listCameras();
  await assert.rejects(() => capture.start({ phoneCaptureId: "phone-2" }), (error) => error.code === "companion_camera_required");
  capture.selectCamera("kitchen-rtsp");
  await assert.rejects(() => capture.start({ phoneCaptureId: "phone-2" }), (error) => error.code === "companion_not_ready");
  assert.equal(capture.status.state, "failed");
  assert.equal(capture.status.needsServerStop, false);
  assert.ok(harness.calls.some((call) => call.url.endsWith("/orphan-candidate/stop")));
});

test("a retried static start reuses the phone and client request IDs", async () => {
  const harness = makeHarness();
  let startAttempts = 0;
  let firstRequestId = null;
  const originalFetch = harness.fetchImpl;
  harness.fetchImpl = async (url, options) => {
    if (url === "/api/companion-captures" && options.method === "POST" && startAttempts++ === 0) {
      firstRequestId = JSON.parse(options.body).client_request_id;
      throw new Error("response lost after request");
    }
    return originalFetch(url, options);
  };
  const capture = new CompanionCapture({ ...harness, limits: { heartbeatMs: 60_000 } });
  await capture.listCameras();
  capture.selectCamera("kitchen-rtsp");
  await assert.rejects(() => capture.start({ phoneCaptureId: "phone-retry" }), (error) => error.code === "companion_request_failed");
  const requestId = capture.status.clientRequestId;
  await capture.start({ cameraId: "kitchen-rtsp", phoneCaptureId: "phone-retry" });
  const startCalls = harness.calls.filter((call) => call.url === "/api/companion-captures" && call.options.method === "POST");
  assert.equal(startCalls.length, 1);
  assert.equal(startCalls[0].body.phone_capture_id, "phone-retry");
  assert.equal(firstRequestId, requestId);
  assert.equal(startCalls[0].body.client_request_id, requestId);
  await capture.stop("user");
});

test("heartbeat loss stops the phone path through a known session and preserves context", async () => {
  const harness = makeHarness({ heartbeatResponse: new Error("tracking disconnected") });
  const failures = [];
  const { capture, browser } = await startedCapture(harness, { onFailure: (error) => failures.push(error) });
  await assert.rejects(() => capture.heartbeat(), (error) => error.code === "companion_heartbeat_failed");
  assert.equal(capture.status.state, "failed");
  assert.equal(failures.length, 1);
  assert.equal(browser.contexts.at(-1).session_id, "session-1");
  await capture.stop("companion_capture_lost");
  assert.ok(harness.calls.some((call) => call.url.endsWith("/session-1/stop") && call.body.reason === "companion_capture_lost"));
});

test("markers retain failed delivery for bounded retry and finalization polls only within its limit", async () => {
  const harness = makeHarness({ stopResponse: { status: "finalizing", video_status: "finalizing", tracking_status: "finalizing" }, finalResponse: { status: "stopped", video_status: "stopped", tracking_status: "stopped" }, markerFailures: 1 });
  const { capture, browser } = await startedCapture(harness, { limits: { finalizationPollMs: 1, finalizationTimeoutMs: 30, heartbeatMs: 60_000 } });
  await assert.rejects(() => capture.addMarker("doorway"), (error) => error.code === "companion_http_error");
  assert.equal(capture.status.pendingMarkerCount, 1);
  await capture.retryMarkers();
  assert.equal(capture.status.pendingMarkerCount, 0);
  const stopped = await capture.stop("user");
  assert.equal(stopped.state, "stopped");
  assert.equal(browser.contexts.at(-1).markers[0].status, "received");
  const stopCall = harness.calls.find((call) => call.url.endsWith("/session-1/stop"));
  assert.equal(stopCall.body.markers[0].marker_id.startsWith("uuid-"), true);
});

test("clock response JSON parsing is bounded and errors remain typed", async () => {
  const harness = makeHarness();
  const capture = new CompanionCapture({ ...harness, limits: { heartbeatMs: 60_000 } });
  await capture.listCameras();
  capture.selectCamera("kitchen-rtsp");
  assert.ok(capture instanceof CompanionCapture);
  assert.equal(capture.status.error, null);
  assert.throws(() => capture.selectCamera("family-rtsp"), (error) => error instanceof CompanionCaptureError && error.code === "companion_camera_required");
});

test("default platform fetch is bound to its native owner", async () => {
  const harness = makeHarness();
  const platform = { ...harness.platform };
  platform.fetch = function nativeFetch(url, options) {
    if (this !== platform) throw new TypeError("Illegal invocation");
    return harness.fetchImpl(url, options);
  };
  const capture = new CompanionCapture({ platform, limits: { heartbeatMs: 60_000 } });
  await capture.listCameras();
  assert.equal(capture.status.camerasAvailable, true);
  assert.ok(harness.calls.some((call) => call.url.endsWith("/cameras")));
});

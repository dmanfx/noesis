import assert from "node:assert/strict";
import test from "node:test";

import {
  BrowserCapture,
  BrowserCaptureError,
  buildTarArchive,
  chooseVideoMimeType,
} from "./static/browser_capture.js";

class FakeTrack extends EventTarget {
  kind = "video";
  stopped = false;

  getSettings() {
    return { width: 7680, height: 4320, frameRate: 30, facingMode: "environment", resizeMode: "none" };
  }

  getCapabilities() {
    return { width: { min: 320, max: 3840 }, height: { min: 240, max: 2160 }, frameRate: { min: 1, max: 60 } };
  }

  stop() {
    this.stopped = true;
  }
}

class FakeStream {
  constructor(track = new FakeTrack()) {
    this.track = track;
  }

  getVideoTracks() {
    return [this.track];
  }

  getTracks() {
    return [this.track];
  }
}

class FakeAccelerometer extends EventTarget {
  static latest;

  constructor(options) {
    super();
    this.options = options;
    this.timestamp = null;
    this.x = null;
    this.y = null;
    this.z = null;
    this.constructor.latest = this;
  }

  start() {}
  stop() {}

  emit(timestamp, x, y, z) {
    this.timestamp = timestamp;
    this.x = x;
    this.y = y;
    this.z = z;
    this.dispatchEvent(new Event("reading"));
  }
}

class FakeGyroscope extends FakeAccelerometer {}

class FakeRecorder {
  static latest;
  static isTypeSupported(type) {
    return type === "video/webm;codecs=vp8" || type === "video/webm";
  }

  constructor(_stream, options) {
    this.mimeType = options.mimeType;
    this.state = "inactive";
    this.onstart = null;
    this.ondataavailable = null;
    this.onstop = null;
    this.onerror = null;
    FakeRecorder.latest = this;
  }

  start() {
    this.state = "recording";
    this.onstart?.(new Event("start"));
  }

  stop() {
    if (this.state === "inactive") throw new Error("already stopped");
    this.state = "inactive";
    this.ondataavailable?.({ data: new Blob(["encoded camera bytes"], { type: this.mimeType }) });
    this.onstop?.(new Event("stop"));
  }
}

class FakeVideo {
  srcObject = null;
  callbacks = new Map();
  nextCallbackId = 1;

  play() {
    return Promise.resolve();
  }

  requestVideoFrameCallback(callback) {
    const id = this.nextCallbackId++;
    this.callbacks.set(id, callback);
    return id;
  }

  cancelVideoFrameCallback(id) {
    this.callbacks.delete(id);
  }

  emitFrame(callbackTime, metadata) {
    const [id, callback] = this.callbacks.entries().next().value || [];
    if (!callback) return;
    this.callbacks.delete(id);
    callback(callbackTime, metadata);
  }
}

class FakeDocument extends EventTarget {
  visibilityState = "visible";
}

class FakeScreenOrientation extends EventTarget {
  type = "portrait-primary";
  angle = 0;
}

function makePlatform({ generic = true, devicemotion = null } = {}) {
  const window = new EventTarget();
  window.isSecureContext = true;
  window.setTimeout = setTimeout;
  window.clearTimeout = clearTimeout;
  window.setInterval = setInterval;
  window.clearInterval = clearInterval;
  window.crypto = { randomUUID: () => `capture-${Math.random().toString(16).slice(2)}` };
  const document = new FakeDocument();
  const screen = { orientation: new FakeScreenOrientation() };
  const track = new FakeTrack();
  const platform = {
    window,
    document,
    screen,
    location: { protocol: "https:" },
    performance: { timeOrigin: 1_700_000_000_000, now: () => 100 },
    navigator: {
      userAgent: "Mozilla/5.0 (Linux; Android 14; Pixel 8 Build/UP1A) Chrome/130 Mobile Safari/537.36",
      platform: "Linux armv8l",
      mediaDevices: { getSupportedConstraints: () => ({ resizeMode: true }), getUserMedia: async () => new FakeStream(track) },
    },
    MediaRecorder: FakeRecorder,
    Blob,
  };
  if (generic) {
    window.Accelerometer = FakeAccelerometer;
    window.Gyroscope = FakeGyroscope;
  } else if (devicemotion) {
    window.DeviceMotionEvent = devicemotion;
  }
  return { platform, window, document, screen, track };
}

function tarEntries(buffer) {
  const bytes = new Uint8Array(buffer);
  const entries = [];
  let offset = 0;
  while (offset + 1024 <= bytes.length) {
    const header = bytes.slice(offset, offset + 512);
    if (header.every((byte) => byte === 0)) break;
    const name = new TextDecoder().decode(header.slice(0, 100)).replace(/\0.*$/, "");
    const sizeText = new TextDecoder().decode(header.slice(124, 136)).replace(/\0.*$/, "").trim();
    const size = parseInt(sizeText || "0", 8);
    entries.push({ name, size, offset: offset + 512 });
    offset += 512 + size + ((512 - (size % 512)) % 512);
  }
  return entries;
}

function emitGenericReadings() {
  FakeAccelerometer.latest.emit(10, 0.1, 0.2, 9.8);
  FakeGyroscope.latest.emit(10, 0.01, 0.02, 0.03);
  FakeAccelerometer.latest.emit(20, 0.1, 0.2, 9.8);
  FakeGyroscope.latest.emit(20, 0.01, 0.02, 0.03);
}

test("buildTarArchive produces an uncompressed TAR from Blob parts", async () => {
  const archive = buildTarArchive([
    { name: "capture_manifest.json", data: new Blob(["{}"]) },
    { name: "phone_walk.webm", data: new Blob(["video"]) },
  ]);
  assert.equal(archive.type, "application/x-tar");
  assert.deepEqual(tarEntries(await archive.arrayBuffer()).map(({ name, size }) => ({ name, size })), [
    { name: "capture_manifest.json", size: 2 },
    { name: "phone_walk.webm", size: 5 },
  ]);
});

test("generic sensor capture waits for real readings and preserves metadata", async () => {
  const { platform } = makePlatform();
  const video = new FakeVideo();
  const capture = new BrowserCapture({ platform });
  assert.equal(chooseVideoMimeType(FakeRecorder), "video/webm;codecs=vp8");
  await capture.open(video);
  assert.equal(capture.status.state, "waiting_sensors");
  assert.equal(capture.status.ready, false);
  assert.throws(() => capture.start(), (error) => error instanceof BrowserCaptureError && error.code === "capture_not_ready");
  FakeAccelerometer.latest.emit(10, null, 0, 9.8);
  FakeGyroscope.latest.emit(10, 0.01, 0.02, null);
  assert.deepEqual(capture.status.counts, { accelerometer: 0, gyroscope: 0 });
  capture.video.emitFrame(50, { mediaTime: 0.01 });
  capture.video.emitFrame(60, { mediaTime: 0.02 });
  emitGenericReadings();
  assert.equal(capture.status.state, "ready");
  capture.start({
    captureId: "phone-capture-1",
    companionCapture: {
      schema: "noesis.phone_capture.companion_ref.v1",
      session_id: "static-session-1",
      camera_id: "kitchen-rtsp",
      phone_capture_id: "phone-capture-1",
      clock_probes: [{ name: "start_1", server_received_unix_ns: "1" }],
      markers: [],
    },
  });
  video.emitFrame(21, { mediaTime: 0.5, expectedDisplayTime: 22, width: 1920, height: 1080 });
  FakeAccelerometer.latest.emit(30, 0.2, 0.3, 9.7);
  FakeGyroscope.latest.emit(30, 0.04, 0.05, 0.06);
  const bundle = await capture.stop("user");
  assert.equal(capture.status.state, "complete");
  assert.equal(bundle.uploadable, true);
  assert.equal(capture.manifest.schema, "noesis.phone_capture.browser.v1");
  assert.equal(capture.manifest.capture_id, "phone-capture-1");
  assert.equal(capture.manifest.companion_capture.session_id, "static-session-1");
  assert.match(capture.manifest.device.id, /^browser-session:/);
  assert.equal(capture.manifest.camera.settings.width, 7680);
  assert.equal(capture.manifest.timing.media_recorder_start_event_ms !== null, true);
  assert.equal(capture.manifest.timing.media_recorder_stop_event_ms !== null, true);
  assert.equal(capture.manifest.video_frames[0].capture_time_ms, null);
  assert.equal(capture.manifest.video_frames[2].media_time_s, 0.5);
  assert.equal(capture.manifest.sensors.accelerometer.samples.length, 3);
  assert.equal(capture.manifest.sensors.gyroscope.samples.length, 3);
  const gyroSample = capture.manifest.sensors.gyroscope.samples[0];
  assert.equal(gyroSample.timestamp_domain, "sensor_timestamp_domain_unverified");
  assert.equal(gyroSample.received_timestamp_domain, "performance_time_origin_ms");
  assert.deepEqual(tarEntries(await bundle.blob.arrayBuffer()).map((entry) => entry.name), ["capture_manifest.json", "phone_walk.webm"]);
});

test("device motion fallback requires both fields and records axis/unit provenance", async () => {
  class DeviceMotionEventFallback {}
  DeviceMotionEventFallback.requestPermission = async () => "granted";
  const { platform, window } = makePlatform({ generic: false, devicemotion: DeviceMotionEventFallback });
  const capture = new BrowserCapture({ platform });
  await capture.open(new FakeVideo());
  window.dispatchEvent(new Event("devicemotion"));
  assert.equal(capture.status.ready, false);
  const event = new Event("devicemotion");
  Object.defineProperty(event, "timeStamp", { value: 42 });
  Object.assign(event, {
    accelerationIncludingGravity: { x: 1, y: 2, z: 3 },
    rotationRate: { alpha: 180, beta: 90, gamma: 45 },
  });
  window.dispatchEvent(event);
  window.dispatchEvent(event);
  capture.video.emitFrame(50, { mediaTime: 0.01 });
  capture.video.emitFrame(60, { mediaTime: 0.02 });
  assert.equal(capture.status.ready, true);
  capture.start();
  const bundle = await capture.stop("user");
  const gyro = capture.manifest.sensors.gyroscope.samples[0];
  assert.ok(Math.abs(gyro.x - Math.PI / 2) < 1e-12);
  assert.ok(Math.abs(gyro.y - Math.PI / 4) < 1e-12);
  assert.ok(Math.abs(gyro.z - Math.PI) < 1e-12);
  assert.equal(capture.manifest.sensors.gyroscope.samples[0].source_units, "deg/s");
  assert.match(capture.manifest.sensors.gyroscope.timestamp_provenance, /not hardware acquisition/);
  assert.equal(bundle.uploadable, true);
});

test("successive captures reset IDs and rows, while hidden page stops and preserves a partial", async () => {
  const { platform, document } = makePlatform();
  const capture = new BrowserCapture({ platform });
  await capture.open(new FakeVideo());
  capture.video.emitFrame(50, { mediaTime: 0.01 });
  capture.video.emitFrame(60, { mediaTime: 0.02 });
  emitGenericReadings();
  capture.start();
  const first = await capture.stop("user");
  const firstId = capture.manifest.capture_id;
  capture.markBundleUploaded();
  await capture.close();
  await capture.open(new FakeVideo());
  capture.video.emitFrame(50, { mediaTime: 0.01 });
  capture.video.emitFrame(60, { mediaTime: 0.02 });
  emitGenericReadings();
  capture.start();
  document.visibilityState = "hidden";
  document.dispatchEvent(new Event("visibilitychange"));
  const second = await new Promise((resolve, reject) => {
    capture.stop("page_hidden").then(resolve, reject);
  });
  assert.notEqual(capture.manifest.capture_id, firstId);
  assert.equal(capture.manifest.sensors.accelerometer.samples.length, 2);
  assert.equal(capture.manifest.stop_reason, "page_hidden");
  assert.equal(second.partial, true);
  assert.equal(second.uploadable, true);
  assert.equal(first.fileName.endsWith(".tar"), true);
});

test("HTTP setup fails closed with an actionable secure-context error", async () => {
  const { platform } = makePlatform();
  platform.location.protocol = "http:";
  const capture = new BrowserCapture({ platform });
  await assert.rejects(() => capture.open(new FakeVideo()), (error) => error.code === "insecure_context" && /trust or install the appliance CA/.test(error.message));
  assert.equal(capture.status.state, "error");
});

test("8K negotiation retries portrait only and never falls back to HD", async () => {
  const { platform, track } = makePlatform();
  const requests = [];
  platform.navigator.mediaDevices.getUserMedia = async (request) => {
    requests.push(request);
    const error = new Error("unsupported camera mode");
    error.name = "OverconstrainedError";
    throw error;
  };
  const capture = new BrowserCapture({ platform });
  await assert.rejects(() => capture.open(new FakeVideo()), { code: "camera_8k_unavailable" });
  assert.deepEqual(requests.map((r) => [r.video.width.exact, r.video.height.exact]), [[7680, 4320], [4320, 7680]]);
  assert.ok(requests.every((r) => r.video.resizeMode.exact === "none" && r.audio === false));
  platform.navigator.mediaDevices.getUserMedia = async () => new FakeStream(track);
  track.getSettings = () => ({ width: 1920, height: 1080, resizeMode: "none", facingMode: "environment" });
  await assert.rejects(() => capture.open(new FakeVideo()), { code: "camera_8k_mismatch" });
  assert.equal(track.stopped, true);
});

test("missing unscaled support and failed preview cannot start capture", async () => {
  const { platform, track } = makePlatform();
  platform.navigator.mediaDevices.getSupportedConstraints = () => ({});
  const capture = new BrowserCapture({ platform });
  await assert.rejects(() => capture.open(new FakeVideo()), { code: "unscaled_capture_unavailable" });
  platform.navigator.mediaDevices.getSupportedConstraints = () => ({ resizeMode: true });
  const video = new FakeVideo();
  video.play = async () => { throw new Error("preview failed"); };
  await assert.rejects(() => capture.open(video), /preview failed/);
  assert.equal(capture.status.ready, false);
  assert.equal(track.stopped, true);
});

test("future sensor clock fails setup and preserves the original timestamp", async () => {
  const { platform } = makePlatform();
  const capture = new BrowserCapture({ platform });
  await capture.open(new FakeVideo());
  FakeAccelerometer.latest.emit(1_416_896_228, 0, 0, 9.8);
  assert.equal(capture.status.state, "error");
  assert.equal(capture.status.error.code, "sensor_clock_invalid");
  assert.equal(capture._samples.accelerometer[0].timestamp_ms, 1_416_896_228);
  assert.equal(capture.status.ready, false);
  await capture.close();
});

test("mode changes before start and sensor reversals during recording fail closed", async () => {
  const { platform, track } = makePlatform();
  const capture = new BrowserCapture({ platform });
  await capture.open(new FakeVideo());
  capture.video.emitFrame(50, { mediaTime: 0.01 });
  capture.video.emitFrame(60, { mediaTime: 0.02 });
  emitGenericReadings();
  const settings = track.getSettings;
  track.getSettings = () => ({ width: 3840, height: 2160 });
  assert.throws(() => capture.start(), { code: "camera_8k_mismatch" });
  track.getSettings = settings;
  capture.start();
  FakeAccelerometer.latest.emit(15, 0, 0, 9.8);
  const bundle = await capture.stop();
  assert.equal(bundle.partial, true);
  assert.equal(bundle.uploadable, true);
  assert.equal(capture.manifest.stop_reason, "sensor_clock_invalid");
  assert.equal(capture.manifest.sensors.accelerometer.samples.at(-1).timestamp_ms, 15);
  assert.equal(capture.manifest.admission.metric_vio_allowed, false);
  assert.equal(capture.manifest.video_frames[0].recording_phase, "preview");
});

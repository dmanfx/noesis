/**
 * Browser camera + motion capture for the phone-walk review tool.
 *
 * This is deliberately a provenance recorder. Browser timestamps and sensor
 * APIs are not accepted as metric VIO evidence by themselves. The output is a
 * bounded, uncompressed TAR containing the original MediaRecorder Blob and a
 * JSON manifest which makes that limitation explicit.
 */

export const BROWSER_CAPTURE_SCHEMA = "noesis.phone_capture.browser.v1";
export const BROWSER_8K_MODE = Object.freeze({ width: 7680, height: 4320, videoBitsPerSecond: 128_000_000 });

export const DEFAULT_CAPTURE_LIMITS = Object.freeze({
  maxDurationMs: 10 * 60 * 1000,
  maxBytes: 512 * 1024 * 1024,
  maxSensorRows: 120_000,
  maxVideoFrames: 100_000,
});

const MIME_TYPES = [
  "video/webm;codecs=vp9",
  "video/webm;codecs=vp8",
  "video/webm",
  "video/mp4;codecs=avc1.42E01E",
  "video/mp4",
];

const ZERO_BLOCK = new Uint8Array(512);
const TEXT_ENCODER = typeof TextEncoder === "function" ? new TextEncoder() : null;

export class BrowserCaptureError extends Error {
  constructor(message, { code = "capture_error", helpUrl = null, cause = null } = {}) {
    super(message);
    this.name = "BrowserCaptureError";
    this.code = code;
    this.helpUrl = helpUrl;
    this.cause = cause;
  }
}

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function safeString(value, fallback = "") {
  return typeof value === "string" ? value : fallback;
}

function encodeText(value) {
  if (TEXT_ENCODER) return TEXT_ENCODER.encode(String(value));
  // This branch only supports old browsers; all current target browsers have
  // TextEncoder. Keep it small so it cannot accidentally become a hot path.
  const encoded = unescape(encodeURIComponent(String(value)));
  const bytes = new Uint8Array(encoded.length);
  for (let index = 0; index < encoded.length; index += 1) bytes[index] = encoded.charCodeAt(index);
  return bytes;
}

function copyJsonValue(value, depth = 0) {
  if (depth > 4 || value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (Array.isArray(value)) return value.slice(0, 256).map((entry) => copyJsonValue(entry, depth + 1));
  if (typeof value !== "object") return null;
  const output = {};
  for (const key of Object.keys(value).slice(0, 256)) {
    const item = copyJsonValue(value[key], depth + 1);
    if (item !== undefined) output[key] = item;
  }
  return output;
}

function parseOctal(value, length) {
  const text = Math.max(0, Number(value) || 0).toString(8);
  const padded = text.padStart(length - 1, "0").slice(-(length - 1));
  return `${padded}\0`;
}

function writeAscii(target, offset, value, length) {
  const bytes = encodeText(value);
  target.set(bytes.slice(0, length), offset);
}

function tarHeader(path, size) {
  if (!path || path.startsWith("/") || path.includes("..") || path.includes("\\") || path.length > 100) {
    throw new BrowserCaptureError(`Unsafe TAR member path: ${path}`, { code: "archive_path" });
  }
  const header = new Uint8Array(512);
  writeAscii(header, 0, path, 100);
  writeAscii(header, 100, parseOctal(0o600, 8), 8);
  writeAscii(header, 108, parseOctal(0, 8), 8);
  writeAscii(header, 116, parseOctal(0, 8), 8);
  writeAscii(header, 124, parseOctal(size, 12), 12);
  writeAscii(header, 136, parseOctal(Math.floor(Date.now() / 1000), 12), 12);
  header.fill(0x20, 148, 156);
  header[156] = 0x30;
  writeAscii(header, 257, "ustar\0", 6);
  writeAscii(header, 263, "00", 2);
  writeAscii(header, 265, "browser", 32);
  writeAscii(header, 297, "browser", 32);
  let checksum = 0;
  for (const byte of header) checksum += byte;
  writeAscii(header, 148, `${checksum.toString(8).padStart(6, "0")}\0 `, 8);
  return header;
}

function tarPadding(size) {
  const remainder = Number(size) % 512;
  return remainder ? ZERO_BLOCK.slice(0, 512 - remainder) : ZERO_BLOCK.slice(0, 0);
}

/**
 * Build an uncompressed TAR without concatenating the recorded video into a
 * second giant ArrayBuffer. Each file can be a Blob, Uint8Array, or string.
 */
export function buildTarArchive(files, { BlobCtor = globalThis.Blob } = {}) {
  if (typeof BlobCtor !== "function") throw new BrowserCaptureError("This browser cannot create a capture archive", { code: "blob_unavailable" });
  const parts = [];
  for (const file of files) {
    const name = safeString(file?.name);
    const data = file?.data instanceof BlobCtor ? file.data : file?.data instanceof Uint8Array ? file.data : encodeText(file?.data ?? "");
    const size = Number(data.size ?? data.byteLength ?? 0);
    parts.push(tarHeader(name, size), data, tarPadding(size));
  }
  parts.push(ZERO_BLOCK, ZERO_BLOCK);
  return new BlobCtor(parts, { type: "application/x-tar" });
}

export function chooseVideoMimeType(MediaRecorderCtor = globalThis.MediaRecorder) {
  if (typeof MediaRecorderCtor !== "function") return null;
  const isSupported = typeof MediaRecorderCtor.isTypeSupported === "function"
    ? (value) => MediaRecorderCtor.isTypeSupported(value)
    : () => true;
  return MIME_TYPES.find((value) => isSupported(value)) || null;
}

export function browserModel(userAgent = "", platform = "") {
  const ua = safeString(userAgent);
  if (/Android/i.test(ua)) {
    // Android user agents generally expose the handset between the first two
    // semicolons. Keep the full UA in device.user_agent for exact provenance.
    const match = ua.match(/Android[^;)]*;\s*([^;)]+?)(?:\s+Build[ /][^;)]+)?[;)]/i);
    const model = match?.[1]?.trim();
    return model && !/^wv$/i.test(model) ? model : "Android browser";
  }
  if (/iPhone/i.test(ua)) return "iPhone browser";
  if (/iPad/i.test(ua)) return "iPad browser";
  return safeString(platform, "Browser") || "Browser";
}

function orientationSnapshot(screenObject) {
  const orientation = screenObject?.orientation;
  return {
    type: safeString(orientation?.type, "unknown"),
    angle: finiteNumber(orientation?.angle),
  };
}

function nowFrom(performanceObject) {
  const value = performanceObject?.now?.();
  return finiteNumber(value) ?? 0;
}

function timeOriginFrom(performanceObject) {
  const value = finiteNumber(performanceObject?.timeOrigin);
  return value ?? Date.now() - nowFrom(performanceObject);
}

function streamTrack(stream) {
  return stream?.getVideoTracks?.()[0] || stream?.getTracks?.()?.find((track) => track.kind === "video") || null;
}

function sensorValue(sensor, key) {
  return finiteNumber(sensor?.[key]);
}

function allFinite(values) {
  return values.every((value) => finiteNumber(value) !== null);
}

function eventTimestampDomain(value) {
  return Number(value) > 1e11 ? "event.timeStamp_epoch_ms" : "event.timeStamp_relative_ms";
}

export class BrowserCapture {
  constructor({
    platform = globalThis,
    limits = DEFAULT_CAPTURE_LIMITS,
    onStateChange = () => {},
    onBundleReady = () => {},
    onBeforeFinalize = async () => {},
  } = {}) {
    this.platform = platform;
    this.window = platform.window || platform;
    this.navigator = platform.navigator || this.window.navigator || {};
    this.document = platform.document || this.window.document || null;
    this.performance = platform.performance || globalThis.performance;
    this.screen = platform.screen || this.window.screen || null;
    this.location = platform.location || this.window.location || {};
    this.MediaRecorderCtor = platform.MediaRecorder || this.window.MediaRecorder || globalThis.MediaRecorder;
    this.BlobCtor = platform.Blob || this.window.Blob || globalThis.Blob;
    this.limits = { ...DEFAULT_CAPTURE_LIMITS, ...limits };
    this.onStateChange = onStateChange;
    this.onBundleReady = onBundleReady;
    this.onBeforeFinalize = onBeforeFinalize;
    this.state = "idle";
    this.error = null;
    this.stream = null;
    this.track = null;
    this.recorder = null;
    this.video = null;
    this.sensorMode = null;
    this.sensors = {};
    this._sensorListeners = [];
    this._chunks = [];
    this._bytes = 0;
    this._sampleCounts = { accelerometer: 0, gyroscope: 0 };
    this._samples = { accelerometer: [], gyroscope: [] };
    this._lastSampleReceived = { accelerometer: null, gyroscope: null };
    this._videoFrames = [];
    this._events = [];
    this._startedMs = null;
    this._stoppedMs = null;
    this._recorderStartCallMs = null;
    this._recorderStartEventMs = null;
    this._recorderStopCallMs = null;
    this._recorderStopEventMs = null;
    this._stopReason = null;
    this._metadataFrameHandle = null;
    this._durationTimer = null;
    this._sensorReadyTimer = null;
    this._sensorFreshnessTimer = null;
    this._finalizePromise = null;
    this._completionResolve = null;
    this._completionReject = null;
    this.bundleUploaded = false;
    this._lastStateEmitMs = -Infinity;
    this._deviceMotionPermissionRequested = false;
    this.bundle = null;
    this.manifest = null;
    this._companionCapture = null;
    this._boundVisibility = () => {
      if (this.document?.visibilityState !== "hidden") return;
      if (this.isRecording || this.state === "stopping") void this.stop("page_hidden").catch(() => {});
      else if (["starting", "waiting_sensors", "ready"].includes(this.state)) void this.close();
    };
    this._boundOrientation = () => {
      if (this.state !== "idle" && this.state !== "closed") this._event("screen.orientation.change", { orientation: orientationSnapshot(this.screen) });
      else this._emitState(true);
    };
    this.document?.addEventListener?.("visibilitychange", this._boundVisibility);
    this.screen?.orientation?.addEventListener?.("change", this._boundOrientation);
  }

  get isReady() {
    const now = nowFrom(this.performance);
    return !this.error && this.track?.readyState !== "ended"
      && this._videoFrames.length >= 2
      && now - this._videoFrames.at(-1).callback_time_ms <= 2000
      && this._sampleCounts.accelerometer >= 2
      && this._sampleCounts.gyroscope >= 2
      && this._lastSampleReceived.accelerometer !== null
      && this._lastSampleReceived.gyroscope !== null
      && now - this._lastSampleReceived.accelerometer <= 2_000
      && now - this._lastSampleReceived.gyroscope <= 2_000;
  }

  get isRecording() {
    return this.state === "recording";
  }

  hasUnsavedWork() {
    return this.isRecording || this.state === "stopping" || Boolean(this.bundle && !this.bundleUploaded);
  }

  markBundleUploaded() {
    this.bundleUploaded = true;
    this._emitState(true);
  }

  /**
   * Attach a bounded, JSON-safe snapshot of the companion session. The app
   * updates this while the static session is alive so the final TAR contains
   * the latest probes, markers, and server status.
   */
  setCompanionCapture(context) {
    this._companionCapture = context ? copyJsonValue(context) : null;
    this._emitState(true);
    return this._companionCapture;
  }

  discardBundle() {
    if (this.isRecording || this.state === "stopping") throw new BrowserCaptureError("Stop the recording before discarding its local bundle.", { code: "capture_active" });
    this._stopSensors();
    this._stopTracks();
    this._stopVideoFrameWatch();
    this._chunks = [];
    this.bundle = null;
    this.manifest = null;
    this.state = "closed";
    this._emitState(true);
  }

  get status() {
    return {
      state: this.state,
      error: this.error,
      secure: this._isSecureContext(),
      sensorMode: this.sensorMode,
      ready: this.isReady,
      camera: this.camera,
      synchronizationVerified: false,
      estimatedMaxVideoSeconds: Math.floor(Math.max(0, this.limits.maxBytes - 16 * 1024 * 1024) * 8 / BROWSER_8K_MODE.videoBitsPerSecond),
      counts: { ...this._sampleCounts },
      videoFrames: this._videoFrames.length,
      bytes: this._bytes,
      durationMs: this._startedMs === null ? 0 : Math.max(0, (this._stoppedMs ?? nowFrom(this.performance)) - this._startedMs),
      stopReason: this._stopReason,
      bundle: this.bundle,
      manifest: this.manifest,
      unsaved: this.hasUnsavedWork(),
    };
  }

  _emitState(force = false) {
    const now = nowFrom(this.performance);
    if (!force && now - (this._lastStateEmitMs ?? -Infinity) < 250) return;
    this._lastStateEmitMs = now;
    this.onStateChange(this.status);
  }

  _event(type, detail = {}, atMs = nowFrom(this.performance)) {
    this._events.push({
      type,
      at_ms: finiteNumber(atMs),
      received_ms: nowFrom(this.performance),
      ...copyJsonValue(detail),
    });
    this._emitState(true);
  }

  _isSecureContext() {
    // Camera and motion permissions are intentionally HTTPS-only for this
    // tool. Do not rely on localhost's browser exception for a phone URL.
    return this.location?.protocol === "https:" && this.window?.isSecureContext !== false;
  }

  _ensureSecureContext() {
    if (this._isSecureContext()) return;
    throw new BrowserCaptureError(
      "Browser camera and motion capture requires an HTTPS secure context. Open the phone URL with https:// and trust or install the appliance CA before trying again.",
      {
        code: "insecure_context",
        helpUrl: "https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts",
      },
    );
  }

  _resetCaptureData() {
    if (this._durationTimer !== null) this.window.clearInterval?.(this._durationTimer);
    if (this._sensorReadyTimer !== null) this.window.clearTimeout?.(this._sensorReadyTimer);
    if (this._sensorFreshnessTimer !== null) this.window.clearInterval?.(this._sensorFreshnessTimer);
    this._chunks = [];
    this._bytes = 0;
    this._videoFrames = [];
    this._events = [];
    this._sampleCounts = { accelerometer: 0, gyroscope: 0 };
    this._samples = { accelerometer: [], gyroscope: [] };
    this._lastSampleReceived = { accelerometer: null, gyroscope: null };
    this._startedMs = null;
    this._stoppedMs = null;
    this._recorderStartCallMs = null;
    this._recorderStartEventMs = null;
    this._recorderStopCallMs = null;
    this._recorderStopEventMs = null;
    this._stopReason = null;
    this._captureId = null;
    this.bundle = null;
    this.bundleUploaded = false;
    this.manifest = null;
    this._companionCapture = null;
    this._durationTimer = null;
    this._sensorReadyTimer = null;
    this._sensorFreshnessTimer = null;
    this._finalizePromise = null;
    this._completionResolve = null;
    this._completionReject = null;
    this._lastStateEmitMs = -Infinity;
    this.camera = null;
    this._deviceMotionPermissionRequested = false;
  }

  async open(videoElement) {
    if (this.state === "recording" || this.state === "starting") return this.status;
    this.state = "starting";
    this.error = null;
    this._resetCaptureData();
    this.video = videoElement || null;
    this._emitState(true);
    try {
      this._ensureSecureContext();
      if (!this.navigator.mediaDevices?.getUserMedia) {
        throw new BrowserCaptureError("This browser does not expose getUserMedia. Use a current Chrome or Safari over HTTPS.", { code: "camera_api_unavailable" });
      }
      if (!chooseVideoMimeType(this.MediaRecorderCtor)) {
        throw new BrowserCaptureError("This browser cannot record a supported WebM or MP4 camera stream.", { code: "recorder_unavailable" });
      }
      // iOS requires DeviceMotionEvent.requestPermission() to run directly
      // from the user gesture. Ask before awaiting getUserMedia so the camera
      // prompt cannot consume that activation.
      await this._prepareSensorPermission();
      if (!this.navigator.mediaDevices.getSupportedConstraints?.().resizeMode) {
        throw new BrowserCaptureError("This browser cannot request unscaled camera output. 8K capture is unavailable here; use the native camera + IMU recorder and import its bundle.", { code: "unscaled_capture_unavailable" });
      }
      // Try both orientations of the same 8K mode. Never fall back to 4K/HD.
      for (const [width, height] of [[7680, 4320], [4320, 7680]]) {
        try {
          this.stream = await this.navigator.mediaDevices.getUserMedia({
            audio: false,
            video: { facingMode: { exact: "environment" }, width: { exact: width }, height: { exact: height }, resizeMode: { exact: "none" }, frameRate: { ideal: 30, max: 30 } },
          });
          break;
        } catch (error) {
          if (error?.name !== "OverconstrainedError") throw error;
          if (width === 4320) throw new BrowserCaptureError("The browser camera does not expose unscaled 8K (7680 × 4320 or portrait). No lower-resolution recording was started. Use the native camera + IMU recorder and import its bundle.", { code: "camera_8k_unavailable", cause: error });
        }
      }
      this.track = streamTrack(this.stream);
      if (!this.track) throw new BrowserCaptureError("The camera permission succeeded but no video track was returned.", { code: "camera_track_missing" });
      this._recordCameraDetails();
      this._verifyCameraMode();
      this.track.addEventListener?.("ended", () => {
        if (["starting", "waiting_sensors", "ready", "recording"].includes(this.state)) this._handleFatal("camera_track_ended", "The camera track ended; the partial capture was preserved.");
      });
      if (this.video) {
        this.video.srcObject = this.stream;
        this.video.muted = true;
        this.video.playsInline = true;
        await this.video.play();
        this._watchVideoFrames();
      } else {
        throw new BrowserCaptureError("Camera preview is required to check frame progress.", { code: "preview_unavailable" });
      }
      this._event("camera.ready", {
        settings: this._cameraSettings(),
        capabilities: this._cameraCapabilities(),
      });
      await this._startSensors();
      if (this.error) throw this.error;
      this.state = "waiting_sensors";
      this._event("sensors.waiting", { message: "Keep the phone stationary while both sensor streams initialize." });
      if (this.isReady) {
        this.state = "ready";
        this._event("capture.ready", { message: "Both sensor streams are producing readings; keep the phone stationary until recording starts." });
      }
      this._sensorReadyTimer = this.window.setTimeout?.(() => {
        if (!this.isReady && this.state === "waiting_sensors") this._handleFatal("sensor_timeout", "The camera preview and both motion streams must keep producing readings. Check camera/motion permissions and keep this page in the foreground.");
      }, 10_000);
      this._emitState();
      return this.status;
    } catch (error) {
      this._handleOpenError(error);
      throw this.error;
    }
  }

  _handleOpenError(error) {
    this.error = error instanceof BrowserCaptureError
      ? error
      : new BrowserCaptureError(this._permissionMessage(error), { code: "capture_open_failed", cause: error });
    this.state = "error";
    this._event("capture.error", { code: this.error.code, message: this.error.message });
    this._stopSensors();
    this._stopTracks();
    this._stopVideoFrameWatch();
    this._emitState();
  }

  _permissionMessage(error) {
    const name = safeString(error?.name);
    if (name === "NotAllowedError" || name === "SecurityError") return "Camera or motion permission was denied. Allow camera and motion sensors for this HTTPS page, then try again.";
    if (name === "NotFoundError") return "No rear camera was found on this device.";
    return error?.message || "Browser camera and motion setup failed.";
  }

  _recordCameraDetails() {
    const details = {
      settings: this._cameraSettings(),
      capabilities: this._cameraCapabilities(),
      requested: { mode: "8k", width: 7680, height: 4320, allow_portrait: true, resize_mode: "none", facing_mode: "environment", audio: false },
    };
    this.camera = details;
  }

  _verifyCameraMode() {
    const settings = this._cameraSettings();
    const dimensions = [settings.width, settings.height].sort((a, b) => b - a);
    if (dimensions[0] !== 7680 || dimensions[1] !== 4320 || settings.resizeMode !== "none" || settings.facingMode !== "environment") {
      throw new BrowserCaptureError(`8K verification failed: browser returned ${settings.width || "?"} × ${settings.height || "?"}, resize mode ${settings.resizeMode || "unknown"}, camera ${settings.facingMode || "unknown"}. No lower-resolution recording was started.`, { code: "camera_8k_mismatch" });
    }
  }

  _cameraSettings() {
    return copyJsonValue(this.track?.getSettings?.() || {});
  }

  _cameraCapabilities() {
    return copyJsonValue(this.track?.getCapabilities?.() || {});
  }

  async _startSensors() {
    const AccelerometerCtor = this.window.Accelerometer || this.platform.Accelerometer;
    const GyroscopeCtor = this.window.Gyroscope || this.platform.Gyroscope;
    if (typeof AccelerometerCtor === "function" && typeof GyroscopeCtor === "function") {
      this.sensorMode = "generic_sensor";
      try {
        const accelerometer = new AccelerometerCtor({ frequency: 60, referenceFrame: "device" });
        const gyroscope = new GyroscopeCtor({ frequency: 60, referenceFrame: "device" });
        this.sensors = { accelerometer, gyroscope };
        this._listenGenericSensor(accelerometer, "accelerometer");
        this._listenGenericSensor(gyroscope, "gyroscope");
        accelerometer.start();
        gyroscope.start();
        this._event("sensors.started", { api: "Generic Sensor API", accelerometer: "Accelerometer (including gravity)", gyroscope: "Gyroscope", reference_frame: "device" });
        return;
      } catch (error) {
        this._stopSensors();
        throw new BrowserCaptureError(this._permissionMessage(error), { code: "sensor_permission_denied", cause: error });
      }
    }
    await this._startDeviceMotionFallback();
  }

  async _prepareSensorPermission() {
    const AccelerometerCtor = this.window.Accelerometer || this.platform.Accelerometer;
    const GyroscopeCtor = this.window.Gyroscope || this.platform.Gyroscope;
    if (typeof AccelerometerCtor === "function" && typeof GyroscopeCtor === "function") return;
    const DeviceMotionEventCtor = this.window.DeviceMotionEvent || this.platform.DeviceMotionEvent;
    if (typeof DeviceMotionEventCtor !== "function" || typeof DeviceMotionEventCtor.requestPermission !== "function") return;
    let permission;
    try {
      permission = await DeviceMotionEventCtor.requestPermission();
    } catch (error) {
      throw new BrowserCaptureError("Motion permission could not be requested. Start capture from a direct tap and allow motion access for this HTTPS page.", { code: "motion_permission_failed", cause: error });
    }
    if (permission !== "granted") throw new BrowserCaptureError("Motion permission was denied. Allow motion access for this HTTPS page, then try again.", { code: "motion_permission_denied" });
    this._deviceMotionPermissionRequested = true;
  }

  _listenGenericSensor(sensor, kind) {
    const handler = () => {
      const timestamp = finiteNumber(sensor.timestamp);
      const values = [sensor.x, sensor.y, sensor.z].map(finiteNumber);
      if (timestamp === null || values.some((value) => value === null)) return;
      this._appendSample(kind, {
        timestamp_ms: timestamp,
        received_ms: nowFrom(this.performance),
        x: values[0],
        y: values[1],
        z: values[2],
        timestamp_source: "Generic Sensor sensor.timestamp",
        timestamp_domain: "sensor_timestamp_domain_unverified",
        received_timestamp_domain: "performance_time_origin_ms",
      });
    };
    const errorHandler = (event) => this._handleFatal(`${kind}_sensor_error`, `${kind === "accelerometer" ? "Acceleration" : "Gyroscope"} sensor failed. The partial capture was preserved.`, event?.error?.message);
    sensor.addEventListener?.("reading", handler);
    sensor.addEventListener?.("error", errorHandler);
    this._sensorListeners.push(() => {
      sensor.removeEventListener?.("reading", handler);
      sensor.removeEventListener?.("error", errorHandler);
    });
  }

  async _startDeviceMotionFallback() {
    const DeviceMotionEventCtor = this.window.DeviceMotionEvent || this.platform.DeviceMotionEvent;
    if (typeof DeviceMotionEventCtor !== "function") {
      throw new BrowserCaptureError("This browser exposes neither Generic Sensor acceleration/gyro streams nor DeviceMotionEvent with both streams.", { code: "sensor_api_unavailable" });
    }
    if (typeof DeviceMotionEventCtor.requestPermission === "function" && !this._deviceMotionPermissionRequested) {
      let permission;
      try {
        permission = await DeviceMotionEventCtor.requestPermission();
      } catch (error) {
        throw new BrowserCaptureError("Motion permission could not be requested. Start capture from a direct tap and allow motion access for this HTTPS page.", { code: "motion_permission_failed", cause: error });
      }
      if (permission !== "granted") throw new BrowserCaptureError("Motion permission was denied. Allow motion access for this HTTPS page, then try again.", { code: "motion_permission_denied" });
    }
    this.sensorMode = "devicemotion_fallback";
    const handler = (event) => {
      // Some browsers dispatch DeviceMotionEvent while leaving either value
      // null. Such events are intentionally ignored; they cannot establish a
      // two-stream fallback contract.
      const acceleration = event?.accelerationIncludingGravity;
      const rotation = event?.rotationRate;
      const accelerationValues = [acceleration?.x, acceleration?.y, acceleration?.z].map(finiteNumber);
      // DeviceMotion rotationRate names are alpha(z), beta(x), gamma(y).
      // Store the device-frame vector as x=beta, y=gamma, z=alpha.
      const rotationValues = [rotation?.beta, rotation?.gamma, rotation?.alpha].map(finiteNumber);
      if (accelerationValues.some((value) => value === null) || rotationValues.some((value) => value === null)) return;
      const timestamp = finiteNumber(event.timeStamp);
      if (timestamp === null) return;
      const received = nowFrom(this.performance);
      const timestampDomain = eventTimestampDomain(timestamp);
      this._appendSample("accelerometer", {
        timestamp_ms: timestamp,
        received_ms: received,
        x: accelerationValues[0],
        y: accelerationValues[1],
        z: accelerationValues[2],
        timestamp_source: "DeviceMotionEvent.timeStamp",
        timestamp_domain: timestampDomain,
        fallback_units: "m/s^2",
      });
      // DeviceMotion rotationRate is degrees/second. The manifest units stay
      // SI and retain the conversion provenance alongside each row.
      const radians = rotationValues.map((value) => value * Math.PI / 180);
      this._appendSample("gyroscope", {
        timestamp_ms: timestamp,
        received_ms: received,
        x: radians[0],
        y: radians[1],
        z: radians[2],
        timestamp_source: "DeviceMotionEvent.timeStamp",
        timestamp_domain: timestampDomain,
        source_units: "deg/s",
        conversion: "deg/s * pi/180 -> rad/s",
      });
    };
    this._deviceMotionHandler = handler;
    this.window.addEventListener?.("devicemotion", handler, { passive: true });
    this._sensorListeners.push(() => this.window.removeEventListener?.("devicemotion", handler));
    this._event("sensors.started", {
      api: "DeviceMotionEvent fallback",
      accelerometer: "accelerationIncludingGravity",
      gyroscope: "rotationRate",
      reference_frame: "device (browser event contract)",
      timestamp_provenance: "event.timeStamp; browser event arrival is retained separately",
    });
  }

  _appendSample(kind, sample) {
    if (this._sampleCounts[kind] >= Number(this.limits.maxSensorRows)) {
      this._handleFatal("max_sensor_rows", `The ${kind} sample limit was reached; the partial capture was preserved.`);
      return;
    }
    const target = this._samples?.[kind] || (this._samples = { accelerometer: [], gyroscope: [] })[kind];
    target.push(sample);
    this._sampleCounts[kind] += 1;
    this._lastSampleReceived[kind] = Number(sample.received_ms);
    // Retain the offending raw sample; never repair clock origins from arrival.
    const previous = target.at(-2);
    if (this.sensorMode === "generic_sensor" && (sample.timestamp_ms < 0 || sample.timestamp_ms > sample.received_ms + 5 || sample.received_ms - sample.timestamp_ms > 2000 || (previous && sample.timestamp_ms <= previous.timestamp_ms))) {
      this._handleFatal("sensor_clock_invalid", `${kind} timestamps are stale, non-monotonic, or outside the browser clock. Reliable camera/IMU timing is unavailable; use native synchronized capture. Raw readings are preserved.`);
      return;
    }
    this._updateReadyState();
    this._emitState();
  }

  _updateReadyState() {
    if (this.state === "waiting_sensors" && this.isReady) {
      this.state = "ready";
      if (this._sensorReadyTimer !== null) this.window.clearTimeout?.(this._sensorReadyTimer);
      this._sensorReadyTimer = null;
      this._event("capture.ready", { message: "Both sensor streams are producing readings; keep the phone stationary until recording starts." });
    }
  }

  _watchVideoFrames() {
    if (!this.video || typeof this.video.requestVideoFrameCallback !== "function") throw new BrowserCaptureError("This browser cannot check camera frame progress.", { code: "preview_timing_unavailable" });
    const callback = (callbackTime, metadata = {}) => {
      this._metadataFrameHandle = null;
      if (this._videoFrames.length < Number(this.limits.maxVideoFrames)) {
        const copied = copyJsonValue(metadata);
        this._videoFrames.push({
          media_time_s: finiteNumber(metadata.mediaTime),
          callback_time_ms: finiteNumber(callbackTime) ?? nowFrom(this.performance),
          capture_time_ms: finiteNumber(metadata.captureTime),
          metadata: copied,
          frame_index_claimed: false,
          recording_phase: this.state === "recording" ? "recording" : "preview",
          timestamp_provenance: "requestVideoFrameCallback callback and metadata; no exact encoded frame index",
        });
      } else if (!this._events.some((event) => event.type === "video_frames.limit")) {
        this._event("video_frames.limit", { max_video_frames: Number(this.limits.maxVideoFrames) });
      }
      this._updateReadyState();
      this._emitState();
      if (this.state !== "closed" && this.state !== "error" && this.video) this._metadataFrameHandle = this.video.requestVideoFrameCallback(callback);
    };
    this._videoFrameCallback = callback;
    this._metadataFrameHandle = this.video.requestVideoFrameCallback(callback);
  }

  _stopVideoFrameWatch() {
    if (this.video && this._metadataFrameHandle !== null && typeof this.video.cancelVideoFrameCallback === "function") {
      this.video.cancelVideoFrameCallback(this._metadataFrameHandle);
    }
    this._metadataFrameHandle = null;
  }

  start({ captureId = null, companionCapture = null } = {}) {
    if (this.state !== "ready") throw new BrowserCaptureError("Camera and both sensor streams must be ready before recording starts.", { code: "capture_not_ready" });
    if (!this.isReady) throw new BrowserCaptureError("Wait for progressing camera frames and fresh acceleration and rotation readings.", { code: "sensor_readings_required" });
    this._verifyCameraMode();
    this.error = null;
    const mimeType = chooseVideoMimeType(this.MediaRecorderCtor);
    this._samples = this._samples || { accelerometer: [], gyroscope: [] };
    this._captureId = captureId ? safeString(captureId).replace(/[^A-Za-z0-9._-]/g, "-").slice(0, 120) : this._newCaptureId();
    if (!this._captureId) this._captureId = this._newCaptureId();
    this._companionCapture = companionCapture ? copyJsonValue(companionCapture) : this._companionCapture;
    this._startedMs = nowFrom(this.performance);
    this._recorderStartCallMs = this._startedMs;
    this.recorder = new this.MediaRecorderCtor(this.stream, { mimeType, videoBitsPerSecond: BROWSER_8K_MODE.videoBitsPerSecond });
    this._finalizePromise = new Promise((resolve, reject) => {
      this._completionResolve = resolve;
      this._completionReject = reject;
    });
    this.recorder.onstart = () => {
      this._recorderStartEventMs = nowFrom(this.performance);
      this._event("recording.started", {
        mime_type: mimeType,
        media_recorder_start_call_ms: this._recorderStartCallMs,
        media_recorder_start_event_ms: this._recorderStartEventMs,
      }, this._recorderStartEventMs);
    };
    this.recorder.ondataavailable = (event) => {
      const chunk = event?.data;
      if (!chunk || Number(chunk.size || 0) <= 0) return;
      const size = Number(chunk.size);
      if (this._bytes + size > Number(this.limits.maxBytes)) {
        this._event("recording.limit", { limit: "max_bytes", max_bytes: Number(this.limits.maxBytes), dropped_chunk_bytes: size });
        this._stopReason = "max_bytes";
        void this.stop("max_bytes").catch(() => {});
        return;
      }
      this._chunks.push(chunk);
      this._bytes += size;
      // Reserve room for bounded sensor/preview metadata and a final chunk.
      if (this._bytes >= Math.max(0, this.limits.maxBytes - 32 * 1024 * 1024) && this.isRecording) void this.stop("max_bytes").catch(() => {});
      this._emitState();
    };
    this.recorder.onerror = (event) => this._handleFatal("media_recorder_error", "The browser recorder failed; the partial capture was preserved.", event?.error?.message);
    this.recorder.onstop = () => {
      this._recorderStopEventMs = nowFrom(this.performance);
      void this._finalize().catch((error) => this._handleFinalizeFailure(error));
    };
    this.state = "recording";
    try {
      this.recorder.start(1000);
    } catch (error) {
      this.error = new BrowserCaptureError(error?.message || "The browser recorder could not start.", { code: "recorder_start_failed", cause: error });
      this._stopReason = "media_recorder_error";
      this.state = "error";
      if (this._durationTimer !== null) this.window.clearInterval?.(this._durationTimer);
      this._durationTimer = null;
      this._stopSensors();
      this._stopVideoFrameWatch();
      this._stopTracks();
      this._completionReject?.(this.error);
      this._completionReject = null;
      throw this.error;
    }
    this._durationTimer = this.window.setInterval?.(() => {
      if (!this.isRecording) return;
      const now = nowFrom(this.performance);
      try { this._verifyCameraMode(); } catch (error) {
        this._handleFatal("camera_mode_changed", error.message);
        return;
      }
      if (now - this._startedMs >= Number(this.limits.maxDurationMs)) {
        void this.stop("max_duration_ms").catch(() => {});
      } else if (now - (this._videoFrames.at(-1)?.callback_time_ms ?? -Infinity) > 2000) {
        this._handleFatal("preview_stalled", "The camera preview stopped progressing; the partial capture was preserved.");
      } else if (this._lastSampleReceived.accelerometer === null || this._lastSampleReceived.gyroscope === null || now - this._lastSampleReceived.accelerometer > 2_000 || now - this._lastSampleReceived.gyroscope > 2_000) {
        this._handleFatal("sensor_stalled", "A motion sensor stopped delivering readings; the partial capture was preserved.");
      } else {
        this._emitState();
      }
    }, 250);
    this._emitState(true);
    return this.status;
  }

  stop(reason = "user") {
    if (this.state === "stopping") return this._finalizePromise || Promise.resolve(null);
    if (this.state !== "recording") return this.bundle ? Promise.resolve(this.bundle) : Promise.resolve(null);
    this.state = "stopping";
    this._stopReason = reason;
    this._stoppedMs = nowFrom(this.performance);
    this._recorderStopCallMs = this._stoppedMs;
    this._event("recording.stop_requested", { reason }, this._stoppedMs);
    if (this._durationTimer !== null) this.window.clearInterval?.(this._durationTimer);
    this._durationTimer = null;
    this._stopSensors();
    this._stopVideoFrameWatch();
    try {
      if (this.recorder && this.recorder.state !== "inactive") this.recorder.stop();
      else void this._finalize().catch((error) => this._handleFinalizeFailure(error));
    } catch (error) {
      this.error = new BrowserCaptureError(error?.message || "The browser recorder could not stop.", { code: "recorder_stop_failed", cause: error });
      void this._finalize().catch((finalizeError) => this._handleFinalizeFailure(finalizeError));
    }
    this._emitState(true);
    return this._finalizePromise;
  }

  async _finalize() {
    this._stoppedMs ??= nowFrom(this.performance);
    try {
      await this.onBeforeFinalize({ captureId: this._captureId, reason: this._stopReason, stoppedMs: this._stoppedMs });
    } catch (error) {
      // A static companion stop is best effort. Preserve the phone capture
      // even when the server-side finalization path is unavailable.
      this._event("companion.finalize_hook_failed", { message: error?.message || String(error) });
    }
    // Record the actual onstop observation before serializing the manifest.
    // Recovery paths which never observed onstop retain a null event time.
    if (this._recorderStopEventMs !== null) {
      this._event("recording.stopped", {
        reason: this._stopReason,
        media_recorder_stop_call_ms: this._recorderStopCallMs,
        media_recorder_stop_event_ms: this._recorderStopEventMs,
      }, this._recorderStopEventMs);
    }
    this._event("recording.finalized", {
      reason: this._stopReason,
      media_recorder_stop_event_observed: this._recorderStopEventMs !== null,
      finalized_ms: nowFrom(this.performance),
    });
    const mimeType = safeString(this.recorder?.mimeType, "video/webm");
    const extension = mimeType.includes("mp4") ? "mp4" : "webm";
    const videoPath = `phone_walk.${extension}`;
    const videoBlob = new this.BlobCtor(this._chunks, { type: mimeType });
    const captureId = this._captureId || this._newCaptureId();
    let manifest = this._buildManifest({ captureId, videoPath, mimeType, videoBlob });
    let manifestBlob = new this.BlobCtor([JSON.stringify(manifest, null, 2)], { type: "application/json" });
    let archive = buildTarArchive([
      { name: "capture_manifest.json", data: manifestBlob },
      { name: videoPath, data: videoBlob },
    ], { BlobCtor: this.BlobCtor });
    if (archive.size > Number(this.limits.maxBytes)) {
      this.error = new BrowserCaptureError("The capture archive exceeded its byte limit and was not uploaded.", { code: "archive_too_large" });
      this._stopReason = "max_bytes";
      manifest = this._buildManifest({ captureId, videoPath, mimeType, videoBlob });
      manifestBlob = new this.BlobCtor([JSON.stringify(manifest, null, 2)], { type: "application/json" });
      archive = buildTarArchive([
        { name: "capture_manifest.json", data: manifestBlob },
        { name: videoPath, data: videoBlob },
      ], { BlobCtor: this.BlobCtor });
    }
    this.manifest = manifest;
    this.bundle = {
      blob: archive,
      fileName: `${captureId}.tar`,
      sizeBytes: archive.size,
      partial: this._stopReason !== "user",
      // A sensor stall or camera interruption is an uploadable partial
      // capture when the TAR contains video and both actual streams. Only
      // malformed/empty or over-limit archives are blocked locally.
      uploadable: videoBlob.size > 0
        && this._samples.accelerometer.length >= 2
        && this._samples.gyroscope.length >= 2
        && archive.size <= Number(this.limits.maxBytes),
    };
    this.state = this.error ? "error" : "complete";
    this._stopTracks();
    this.onBundleReady(this.bundle, this.status);
    this._emitState(true);
    this._completionResolve?.(this.bundle);
    this._completionResolve = null;
    this._completionReject = null;
    return this.bundle;
  }

  _handleFinalizeFailure(error) {
    this.error = error instanceof BrowserCaptureError
      ? error
      : new BrowserCaptureError(error?.message || "The browser capture could not be packaged.", { code: "archive_finalize_failed", cause: error });
    this.state = "error";
    this._stopSensors();
    this._stopTracks();
    this._emitState(true);
    this._completionReject?.(this.error);
    this._completionReject = null;
  }

  _buildManifest({ captureId, videoPath, mimeType, videoBlob }) {
    const timeOrigin = timeOriginFrom(this.performance);
    const device = this.navigator.userAgentData?.model || browserModel(this.navigator.userAgent, this.navigator.platform);
    const stopped = this._stoppedMs ?? nowFrom(this.performance);
    const started = this._startedMs ?? stopped;
    const samples = this._samples || { accelerometer: [], gyroscope: [] };
    return {
      schema: BROWSER_CAPTURE_SCHEMA,
      capture_id: captureId,
      video: {
        path: videoPath,
        mime_type: mimeType,
        size_bytes: Number(videoBlob.size || 0),
        requested_bits_per_second: BROWSER_8K_MODE.videoBitsPerSecond,
        recorder_bits_per_second: finiteNumber(this.recorder?.videoBitsPerSecond),
        timestamp_provenance: "MediaRecorder chunks; container/frame acquisition timestamps are not asserted",
      },
      device: {
        id: `browser-session:${captureId}`,
        model: safeString(device, "Android browser") || "Android browser",
        user_agent: safeString(this.navigator.userAgent, "unknown"),
      },
      camera: {
        settings: this.camera?.settings || {},
        capabilities: this.camera?.capabilities || {},
        requested: this.camera?.requested || { facing_mode: "environment", audio: false },
      },
      timing: {
        time_origin_ms: timeOrigin,
        started_ms: started,
        stopped_ms: stopped,
        started_epoch_ms: timeOrigin + started,
        stopped_epoch_ms: timeOrigin + stopped,
        media_recorder_start_ms: this._recorderStartCallMs,
        media_recorder_start_event_ms: this._recorderStartEventMs,
        media_recorder_start_provenance: "MediaRecorder start event; call time retained separately",
        media_recorder_stop_call_ms: this._recorderStopCallMs,
        media_recorder_stop_event_ms: this._recorderStopEventMs,
        screen_orientation: orientationSnapshot(this.screen),
        duration_ms: Math.max(0, stopped - started),
      },
      sensors: {
        accelerometer: {
          api: this.sensorMode === "generic_sensor" ? "Generic Sensor Accelerometer (including gravity)" : "DeviceMotionEvent.accelerationIncludingGravity fallback",
          units: "m/s^2",
          reference_frame: "device",
          timestamp_provenance: this.sensorMode === "generic_sensor" ? "sensor.timestamp; received_ms is performance.now()" : "event.timeStamp; received_ms is performance.now(); browser event timestamp is not hardware acquisition time",
          samples: samples.accelerometer,
        },
        gyroscope: {
          api: this.sensorMode === "generic_sensor" ? "Generic Sensor Gyroscope" : "DeviceMotionEvent.rotationRate fallback",
          units: "rad/s",
          reference_frame: "device",
          timestamp_provenance: this.sensorMode === "generic_sensor" ? "sensor.timestamp; received_ms is performance.now()" : "event.timeStamp; received_ms is performance.now(); browser event timestamp is not hardware acquisition time; rotationRate converted from deg/s",
          samples: samples.gyroscope,
        },
      },
      video_frames: this._videoFrames,
      events: this._events,
      companion_capture: this._companionCapture ? copyJsonValue(this._companionCapture) : null,
      stop_reason: this._stopReason || "unknown",
      admission: {
        metric_vio_allowed: false,
        reason: "browser camera and motion timestamps/calibration are unverified",
      },
    };
  }

  _newCaptureId() {
    const random = this.window.crypto?.randomUUID?.() || `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
    this._captureId = safeString(random).replace(/[^A-Za-z0-9._-]/g, "-").slice(0, 120);
    return this._captureId;
  }

  _handleFatal(reason, message, detail = null) {
    this._event("capture.fatal", { reason, message, detail });
    this.error = new BrowserCaptureError(message, { code: reason });
    if (this.state === "recording") {
      void this.stop(reason).catch(() => {});
    } else {
      this._stopSensors();
      this._stopTracks();
      this._stopVideoFrameWatch();
      this.state = "error";
      this._emitState();
    }
  }

  _stopSensors() {
    for (const remove of this._sensorListeners.splice(0)) remove();
    for (const sensor of Object.values(this.sensors)) sensor.stop?.();
    this.sensors = {};
    if (this._sensorReadyTimer !== null) this.window.clearTimeout?.(this._sensorReadyTimer);
    this._sensorReadyTimer = null;
  }

  _stopTracks() {
    for (const track of this.stream?.getTracks?.() || []) track.stop?.();
    this.stream = null;
    this.track = null;
    if (this.video) this.video.srcObject = null;
  }

  close() {
    if (this.isRecording) return this.stop("user");
    this._stopSensors();
    this._stopTracks();
    this._stopVideoFrameWatch();
    this.state = "closed";
    this._emitState();
    return Promise.resolve(null);
  }

  destroy() {
    this.document?.removeEventListener?.("visibilitychange", this._boundVisibility);
    this.screen?.orientation?.removeEventListener?.("change", this._boundOrientation);
    return this.close();
  }

  downloadBundle() {
    if (!this.bundle?.blob) return false;
    const URLCtor = this.window.URL || this.platform.URL || globalThis.URL;
    const anchor = this.document?.createElement?.("a");
    if (!URLCtor?.createObjectURL || !anchor) return false;
    const url = URLCtor.createObjectURL(this.bundle.blob);
    anchor.href = url;
    anchor.download = this.bundle.fileName;
    anchor.click();
    this.window.setTimeout?.(() => URLCtor.revokeObjectURL?.(url), 1000);
    return true;
  }
}

export default BrowserCapture;

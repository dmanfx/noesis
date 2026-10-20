/**
 * Browser-side coordinator for a static room-camera companion capture.
 *
 * The companion service owns the RTSP recording and tracking stream. This
 * module only carries explicit session IDs, bounded timing evidence, and
 * idempotent lifecycle requests; it never fabricates a world position or
 * joins a phone sample to the latest tracking state.
 */

export const COMPANION_CAPTURE_SCHEMA = "noesis.phone_capture.companion_ref.v1";

/** Missing camera_ids is the historical SINGLE camera, never an inferred pair. */
export function normalizeCameraIds(primary, cameraIds = undefined) {
  if (typeof primary !== "string" || !primary.trim()) throw new Error("A primary room camera is required.");
  const values = cameraIds === undefined ? [primary] : cameraIds;
  if (!Array.isArray(values) || values.length < 1 || values.length > 2
      || values.some((id) => typeof id !== "string" || !id.trim())) throw new Error("Choose one or two room cameras.");
  const normalized = values.map((id) => id.trim()).sort();
  if (new Set(normalized).size !== normalized.length || !normalized.includes(primary.trim())) throw new Error("Room cameras must be unique and include the primary camera.");
  return normalized;
}

function sameCameraIds(a, b) { return a.length === b.length && a.every((id, index) => id === b[index]); }

function compactCameraStates(values, ids) {
  const result = {};
  for (const id of ids) {
    const row = values?.[id];
    if (!row || row.camera_id !== id) continue;
    const urls = {};
    for (const [key, value] of Object.entries(row.artifact_urls || {}).slice(0, 16)) {
      if (key.length <= 160 && typeof value === "string" && value.length <= 2048) urls[key] = value;
    }
    result[id] = { camera_id: id, source_id: typeof row.source_id === "number" ? row.source_id : safeString(row.source_id).slice(0, 160),
      label: safeString(row.label, id).slice(0, 160), source_reference: safeString(row.source_reference).slice(0, 512),
      readiness: { recorder_ready: row.readiness?.recorder_ready === true, tracking_ready: row.readiness?.tracking_ready === true }, artifact_urls: urls };
  }
  return result;
}

export const COMPANION_DEFAULTS = Object.freeze({
  clockProbeTimeoutMs: 3_000,
  startTimeoutMs: 16_000,
  heartbeatTimeoutMs: 5_000,
  stopTimeoutMs: 8_000,
  finalizationTimeoutMs: 15_000,
  finalizationPollMs: 750,
  heartbeatMs: 10_000,
  maxClockProbes: 256,
  maxClockProbeFailures: 64,
  maxMarkers: 256,
  maxMarkerRetriesPerCall: 32,
});

const TERMINAL_STATUSES = new Set(["stopped", "failed", "error", "cancelled"]);
const HEALTHY_STATUSES = new Set(["ready", "recording", "started", "active", "ok", "healthy"]);

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function safeString(value, fallback = "") {
  return typeof value === "string" ? value : fallback;
}

function copyJsonValue(value, depth = 0) {
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (depth > 5) return null;
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

function newId(cryptoObject = globalThis.crypto) {
  const value = cryptoObject?.randomUUID?.();
  if (value) return value;
  const bytes = cryptoObject?.getRandomValues?.(new Uint8Array(16));
  if (bytes) {
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    return [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("").replace(/^(.{8})(.{4})(.{4})(.{4})(.{12})$/, "$1-$2-$3-$4-$5");
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (character) => {
    const random = Math.random() * 16 | 0;
    const valueForY = character === "x" ? random : (random & 0x3) | 0x8;
    return valueForY.toString(16);
  });
}

function statusValue(payload, key = "status") {
  return safeString(payload?.[key]).toLowerCase();
}

function statusHealthy(payload, key, { required = false } = {}) {
  const value = statusValue(payload, key);
  return required ? HEALTHY_STATUSES.has(value) : !value || HEALTHY_STATUSES.has(value);
}

function publicMarker(marker) {
  return {
    marker_id: marker.marker_id,
    label: marker.label,
    client_monotonic_ms: marker.client_monotonic_ms,
    client_epoch_ms: marker.client_epoch_ms,
    status: marker.status,
    receipt_clock: marker.receipt_clock || null,
    error: marker.error || null,
  };
}

function markerRequest(marker) {
  return {
    marker_id: marker.marker_id,
    label: marker.label,
    client_monotonic_ms: marker.client_monotonic_ms,
    client_epoch_ms: marker.client_epoch_ms,
  };
}

export class CompanionCaptureError extends Error {
  constructor(message, { code = "companion_capture_error", status = null, cause = null } = {}) {
    super(message);
    this.name = "CompanionCaptureError";
    this.code = code;
    this.status = status;
    this.cause = cause;
  }
}

export class CompanionCapture {
  constructor({
    platform = globalThis,
    fetchImpl,
    limits = COMPANION_DEFAULTS,
    onStateChange = () => {},
    onFailure = () => {},
  } = {}) {
    this.platform = platform;
    this.window = platform.window || platform;
    this.performance = platform.performance || globalThis.performance;
    const platformFetch = platform?.fetch;
    const defaultFetch = platformFetch || globalThis.fetch;
    // Native Window.fetch requires its owning Window as `this`. Keep injected
    // fetch implementations untouched so callers retain their existing
    // function and receiver semantics; only bind the default native path.
    this.fetchImpl = fetchImpl === undefined
      ? (typeof defaultFetch === "function"
        ? defaultFetch.bind(platformFetch ? platform : globalThis)
        : defaultFetch)
      : fetchImpl;
    this.crypto = platform.crypto || this.window.crypto || globalThis.crypto;
    this.limits = { ...COMPANION_DEFAULTS, ...limits };
    this.onStateChange = onStateChange;
    this.onFailure = onFailure;
    this.state = "idle";
    this.error = null;
    this.cameras = [];
    this.camerasAvailable = false;
    this.cameraReason = null;
    this.selectedCameraId = "";
    this.selectedCameraIds = [];
    this.multiCameraCapture = false;
    this.maxCameras = 1;
    this.phoneCaptureId = null;
    this.clientRequestId = null;
    this.session = null;
    this.clockProbes = [];
    this.clockProbeFailures = [];
    this.clockProbeLimitReached = false;
    this.clockProbeFailureDroppedCount = 0;
    this.markers = [];
    this.markerLimitReached = false;
    this.lastHeartbeat = null;
    this._heartbeatTimer = null;
    this._heartbeatInFlight = false;
    this._stoppingPromise = null;
    this._browserCapture = null;
    this._failureNotified = false;
  }

  get status() {
    const session = this.session || {};
    return {
      state: this.state,
      error: this.error,
      cameras: this.cameras,
      camerasAvailable: this.camerasAvailable,
      cameraReason: this.cameraReason,
      selectedCameraId: this.selectedCameraId,
      selectedCameraIds: this.selectedCameraIds.slice(),
      multiCameraCapture: this.multiCameraCapture,
      maxCameras: this.maxCameras,
      phoneCaptureId: this.phoneCaptureId,
      clientRequestId: this.clientRequestId,
      sessionId: session.session_id || null,
      needsServerStop: this.needsServerStop,
      cameraId: session.camera_id || this.selectedCameraId || null,
      cameraIds: session.camera_ids?.slice() || this.selectedCameraIds.slice(),
      cameraStates: copyJsonValue(session.cameras || {}),
      videoStatus: session.video_status || null,
      trackingStatus: session.tracking_status || null,
      staticStatus: session.status || null,
      lastHeartbeat: this.lastHeartbeat,
      markerCount: this.markers.length,
      pendingMarkerCount: this.markers.filter((marker) => marker.status !== "received").length,
      clockProbeCount: this.clockProbes.length,
      clockProbeFailureCount: this.clockProbeFailures.length,
      clockProbeLimitReached: this.clockProbeLimitReached,
      clockProbeFailureDroppedCount: this.clockProbeFailureDroppedCount,
      markerLimitReached: this.markerLimitReached,
      artifactUrls: copyJsonValue(session.artifact_urls || {}),
      provenance: copyJsonValue(session.provenance || {}),
    };
  }

  get hasLiveSession() {
    return ["starting", "recording", "stopping", "finalizing"].includes(this.state);
  }

  get needsServerStop() {
    const serverStatus = statusValue(this.session);
    return Boolean(this.session?.session_id)
      && !["stopped", "cancelled", "failed", "error"].includes(serverStatus)
      && !(["stopped", "cancelled"].includes(this.state));
  }

  get readyToStart() {
    const camera = this.cameras.find((item) => item.camera_id === this.selectedCameraId);
    return this.camerasAvailable && Boolean(camera?.available) && Boolean(this.selectedCameraId) && !this.hasLiveSession
      && (this.selectedCameraIds.length < 2 || this.multiCameraCapture && this.maxCameras >= 2)
      && this.selectedCameraIds.every((id) => this.cameras.some((item) => item.camera_id === id && item.available === true));
  }

  hasUnsavedWork() {
    return this.hasLiveSession || this.needsServerStop;
  }

  _nowMonotonicMs() {
    return finiteNumber(this.performance?.now?.()) ?? 0;
  }

  _nowEpochMs() {
    return Date.now();
  }

  _emitState(force = true) {
    this.onStateChange(this.status, force);
    this._syncBrowserCapture();
  }

  _setError(error, code = "companion_capture_error") {
    this.error = error instanceof CompanionCaptureError
      ? error
      : new CompanionCaptureError(error?.message || String(error), { code, cause: error });
    return this.error;
  }

  async _requestJson(url, { method = "GET", body = undefined, timeoutMs = 8_000 } = {}) {
    if (typeof this.fetchImpl !== "function") throw new CompanionCaptureError("This browser cannot contact the static capture service.", { code: "fetch_unavailable" });
    const controller = typeof AbortController === "function" ? new AbortController() : null;
    const setTimeoutImpl = this.window.setTimeout || globalThis.setTimeout;
    const clearTimeoutImpl = this.window.clearTimeout || globalThis.clearTimeout;
    let timeoutHandle = null;
    let timedOut = false;
    const request = this.fetchImpl(url, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller?.signal,
    });
    const timeout = new Promise((_, reject) => {
      timeoutHandle = setTimeoutImpl(() => {
        timedOut = true;
        controller?.abort?.();
        reject(new CompanionCaptureError(`Static capture service did not respond within ${Math.ceil(timeoutMs / 1000)} seconds.`, { code: "companion_request_timeout" }));
      }, timeoutMs);
    });
    try {
      const response = await Promise.race([request, timeout]);
      if (!response?.ok) {
        let detail = `${response?.status || 0} ${response?.statusText || "static capture request failed"}`.trim();
        try {
          const payload = await Promise.race([response.json(), timeout]);
          detail = payload?.detail || detail;
        } catch (_) {
          // Keep the HTTP status when the response is not JSON.
        }
        throw new CompanionCaptureError(detail, { code: "companion_http_error", status: response?.status || null });
      }
      if (response.status === 204) return null;
      // Bound JSON body consumption as well as network response arrival. A
      // proxy can send headers and then stall indefinitely otherwise.
      return await Promise.race([response.json(), timeout]);
    } catch (error) {
      if (timedOut) throw error;
      if (error instanceof CompanionCaptureError) throw error;
      throw new CompanionCaptureError(error?.message || "Static capture request failed.", { code: "companion_request_failed", cause: error });
    } finally {
      if (timeoutHandle !== null) clearTimeoutImpl(timeoutHandle);
    }
  }

  async listCameras() {
    try {
      const payload = await this._requestJson("/api/companion-captures/cameras", { timeoutMs: this.limits.clockProbeTimeoutMs });
      this.cameras = Array.isArray(payload?.cameras)
        ? payload.cameras.filter((camera) => camera?.camera_id).map((camera) => copyJsonValue(camera))
        : [];
      this.camerasAvailable = payload?.available === true && this.cameras.some((camera) => camera.available === true);
      this.multiCameraCapture = payload?.multi_camera_capture === true;
      this.maxCameras = payload?.max_cameras === 2 ? 2 : 1;
      this.cameraReason = safeString(payload?.reason) || null;
      this.error = null;
      // Inventory loss never silently changes the explicitly checked set.
      // Unavailable retained choices stay visible/removable; start stays strict.
      this._emitState();
      return this.status;
    } catch (error) {
      this.cameras = [];
      this.camerasAvailable = false;
      this.multiCameraCapture = false;
      this.maxCameras = 1;
      this.cameraReason = error.message;
      this._setError(error, "companion_cameras_unavailable");
      this._emitState();
      throw this.error;
    }
  }

  selectCamera(cameraId) {
    return this.selectCameras(cameraId, [cameraId]);
  }

  selectCameras(cameraId, cameraIds) {
    if (this.hasLiveSession) throw new CompanionCaptureError("Stop the current capture before changing cameras.");
    if (cameraId === "" && Array.isArray(cameraIds) && cameraIds.length === 0) {
      this.selectedCameraId = "";
      this.selectedCameraIds = [];
      this.error = null;
      this._emitState();
      return this.status;
    }
    const value = safeString(cameraId).trim();
    const ids = normalizeCameraIds(value, cameraIds);
    const removingOnly = ids.length < this.selectedCameraIds.length && ids.every((id) => this.selectedCameraIds.includes(id));
    const camera = this.cameras.find((item) => item.camera_id === value);
    if (!removingOnly && (!camera || camera.available !== true)) {
      throw new CompanionCaptureError("Choose an available room camera before starting the paired capture.", { code: "companion_camera_required" });
    }
    if (!removingOnly && ids.length > 1 && (!this.multiCameraCapture || this.maxCameras < 2)) throw new CompanionCaptureError("This server does not support two-camera capture.");
    if (!removingOnly && ids.some((id) => !this.cameras.some((item) => item.camera_id === id && item.available === true))) throw new CompanionCaptureError("Every selected room camera must be available.");
    this.selectedCameraId = value;
    this.selectedCameraIds = ids;
    this.error = null;
    this._emitState();
    return this.status;
  }

  reservePhoneCaptureId() {
    if (!this.phoneCaptureId) this.phoneCaptureId = newId(this.crypto);
    return this.phoneCaptureId;
  }

  _resetSessionState() {
    this._stopHeartbeat();
    this.error = null;
    this.phoneCaptureId = null;
    this.clientRequestId = null;
    this.session = null;
    this.clockProbes = [];
    this.clockProbeFailures = [];
    this.clockProbeLimitReached = false;
    this.clockProbeFailureDroppedCount = 0;
    this.markers = [];
    this.markerLimitReached = false;
    this.lastHeartbeat = null;
    this._stoppingPromise = null;
    this._failureNotified = false;
    this._browserCapture = null;
  }

  reset() {
    if (this.hasUnsavedWork()) throw new CompanionCaptureError("Finish or retry the static companion capture before resetting it.", { code: "companion_capture_active" });
    this._resetSessionState();
    this.state = "idle";
    this._emitState();
  }

  async _clockProbe(name, timeoutMs = this.limits.clockProbeTimeoutMs) {
    if (this.clockProbes.length >= Number(this.limits.maxClockProbes)) {
      this.clockProbeLimitReached = true;
      throw new CompanionCaptureError("The static capture clock-probe limit was reached; the missing probe is recorded as a bounded evidence gap.", { code: "companion_clock_limit" });
    }
    const index = this.clockProbes.length + this.clockProbeFailures.length;
    const sendMonotonic = this._nowMonotonicMs();
    const sendEpoch = this._nowEpochMs();
    try {
      const payload = await this._requestJson("/api/companion-captures/clock", { timeoutMs });
      const receiveMonotonic = this._nowMonotonicMs();
      const receiveEpoch = this._nowEpochMs();
      const required = ["server_received_unix_ns", "server_received_monotonic_ns", "server_sent_unix_ns", "server_sent_monotonic_ns"];
      if (required.some((key) => typeof payload?.[key] !== "string" || !payload[key])) {
        throw new CompanionCaptureError("The static capture clock response omitted required raw timestamps.", { code: "companion_clock_invalid" });
      }
      const roundTrip = Math.max(0, receiveMonotonic - sendMonotonic);
      const probe = {
        name,
        index,
        client_send_monotonic_ms: sendMonotonic,
        client_receive_monotonic_ms: receiveMonotonic,
        client_send_epoch_ms: sendEpoch,
        client_receive_epoch_ms: receiveEpoch,
        server_received_unix_ns: payload.server_received_unix_ns,
        server_received_monotonic_ns: payload.server_received_monotonic_ns,
        server_sent_unix_ns: payload.server_sent_unix_ns,
        server_sent_monotonic_ns: payload.server_sent_monotonic_ns,
        round_trip_ms: roundTrip,
        uncertainty_ms: roundTrip / 2,
        timestamp_provenance: "raw server clock response with client send/receive observations; no acquisition synchronization asserted",
      };
      this.clockProbes.push(probe);
      return probe;
    } catch (error) {
      const failure = {
        name,
        index,
        client_send_monotonic_ms: sendMonotonic,
        client_send_epoch_ms: sendEpoch,
        error: safeString(error?.message, "clock probe failed"),
      };
      if (this.clockProbeFailures.length < Number(this.limits.maxClockProbeFailures)) this.clockProbeFailures.push(failure);
      else this.clockProbeFailureDroppedCount += 1;
      throw error instanceof CompanionCaptureError ? error : new CompanionCaptureError(failure.error, { code: "companion_clock_failed", cause: error });
    }
  }

  async _collectClockProbes(name, count, { strict = false } = {}) {
    const probes = [];
    for (let index = 0; index < count; index += 1) {
      try {
        probes.push(await this._clockProbe(`${name}_${index + 1}`));
      } catch (error) {
        if (strict) throw error;
      }
    }
    return probes;
  }

  _context() {
    const session = this.session || {};
    return {
      schema: COMPANION_CAPTURE_SCHEMA,
      session_id: session.session_id || null,
      camera_id: session.camera_id || this.selectedCameraId || null,
      camera_ids: session.camera_ids?.slice() || this.selectedCameraIds.slice(),
      cameras: copyJsonValue(session.cameras || {}),
      phone_capture_id: this.phoneCaptureId,
      status: this.state,
      video_status: session.video_status || null,
      tracking_status: session.tracking_status || null,
      client_request_id: this.clientRequestId,
      clock_probes: this.clockProbes.slice(),
      clock_probe_failures: this.clockProbeFailures.slice(),
      clock_probe_limit_reached: this.clockProbeLimitReached,
      clock_probe_failure_dropped_count: this.clockProbeFailureDroppedCount,
      markers: this.markers.map(publicMarker),
      marker_limit_reached: this.markerLimitReached,
      provenance: copyJsonValue(session.provenance || {}),
      artifact_urls: copyJsonValue(session.artifact_urls || {}),
      timestamp_provenance: "companion service lifecycle and server clock responses; no exact phone/static acquisition alignment asserted",
    };
  }

  _syncBrowserCapture() {
    this._browserCapture?.setCompanionCapture?.(this._context());
  }

  _adopt(payload, { initial = false } = {}) {
    const primary = this.selectedCameraId;
    const required = initial || this.selectedCameraIds.length > 1;
    if (required || payload?.camera_id !== undefined || payload?.camera_ids !== undefined) {
      if (payload?.camera_id !== primary || !sameCameraIds(normalizeCameraIds(payload.camera_id, payload.camera_ids), this.selectedCameraIds)) throw new CompanionCaptureError("The server did not confirm the exact requested room-camera set.", { code: "companion_camera_set_mismatch" });
    }
    if ((required || payload?.phone_capture_id !== undefined) && payload?.phone_capture_id !== this.phoneCaptureId) throw new CompanionCaptureError("The companion phone-capture identity changed.", { code: "companion_identity_mismatch" });
    if (payload?.session_id !== undefined && this.session?.session_id && payload.session_id !== this.session.session_id) throw new CompanionCaptureError("The companion session identity changed.", { code: "companion_identity_mismatch" });
    const { cameras, ...scalars } = payload || {};
    this.session = { ...this.session, ...copyJsonValue(scalars), camera_ids: this.selectedCameraIds.slice(),
      cameras: cameras === undefined ? this.session?.cameras || {} : compactCameraStates(cameras, this.selectedCameraIds) };
  }

  _membersReady(payload) {
    return this.selectedCameraIds.length < 2 || this.selectedCameraIds.every((id) => {
      const member = payload?.cameras?.[id];
      return member?.camera_id === id && member.readiness?.recorder_ready === true && member.readiness?.tracking_ready === true;
    });
  }

  async start({ cameraId = this.selectedCameraId, cameraIds = this.selectedCameraIds.length ? this.selectedCameraIds : undefined, phoneCaptureId = null, clientRequestId = null, browserCapture = null } = {}) {
    if (this.hasLiveSession) throw new CompanionCaptureError("A static companion capture is already active.", { code: "companion_capture_active" });
    const camera = this.cameras.find((item) => item.camera_id === cameraId);
    if (!this.camerasAvailable || !camera || camera.available !== true) {
      throw new CompanionCaptureError("Choose an available room camera before starting the paired capture.", { code: "companion_camera_required" });
    }
    this.selectCameras(cameraId, normalizeCameraIds(cameraId, cameraIds));
    const requestedPhoneCaptureId = safeString(phoneCaptureId) || this.phoneCaptureId || this.reservePhoneCaptureId();
    if (!requestedPhoneCaptureId) throw new CompanionCaptureError("A phone capture ID is required before starting the static companion.", { code: "phone_capture_id_required" });
    const reusableRequestId = this.phoneCaptureId === requestedPhoneCaptureId ? this.clientRequestId : null;
    this._resetSessionState();
    this.selectedCameraId = cameraId;
    this.phoneCaptureId = requestedPhoneCaptureId;
    this.clientRequestId = safeString(clientRequestId) || reusableRequestId || newId(this.crypto);
    this._browserCapture = browserCapture;
    this.state = "starting";
    this._emitState();
    try {
      await this._collectClockProbes("start", 5, { strict: true });
      const payload = await this._requestJson("/api/companion-captures", {
        method: "POST",
        timeoutMs: this.limits.startTimeoutMs,
        body: {
          camera_id: cameraId,
          camera_ids: this.selectedCameraIds.slice(),
          phone_capture_id: requestedPhoneCaptureId,
          client_request_id: this.clientRequestId,
          clock_probes: this.clockProbes.slice(),
        },
      });
      // Retain only the returned session address for idempotent stop if identity validation fails.
      if (payload?.session_id) this.session = { session_id: payload.session_id };
      this._adopt(payload, { initial: true });
      const staticReady = statusValue(payload) === "recording"
        && statusHealthy(payload, "video_status", { required: true })
        && statusHealthy(payload, "tracking_status", { required: true }) && this._membersReady(payload);
      if (!staticReady || !payload?.session_id) {
        if (payload?.session_id) {
          const cleanup = await this._stopRequest("phone_start_failed", { bestEffort: true });
          if (cleanup) this._adopt(cleanup);
        }
        throw new CompanionCaptureError("The selected static camera did not reach recording and tracking ready state.", { code: "companion_not_ready", status: payload?.status || null });
      }
      this.state = "recording";
      this.error = null;
      this._emitState();
      this._startHeartbeat();
      return this.status;
    } catch (error) {
      this._stopHeartbeat();
      if (this.session?.session_id && !TERMINAL_STATUSES.has(statusValue(this.session))) {
        const cleanup = await this._stopRequest("phone_start_failed", { bestEffort: true });
        if (cleanup) { try { this._adopt(cleanup); } catch (_) { /* lease/retry remains explicit */ } }
      }
      this._setError(error, error?.code || "companion_start_failed");
      this.state = "failed";
      this._emitState();
      throw this.error;
    }
  }

  _startHeartbeat() {
    this._stopHeartbeat();
    const setIntervalImpl = this.window.setInterval || globalThis.setInterval;
    this._heartbeatTimer = setIntervalImpl(() => {
      if (this.state !== "recording" || this._heartbeatInFlight) return;
      void this.heartbeat().catch(() => {});
    }, this.limits.heartbeatMs);
  }

  _stopHeartbeat() {
    if (this._heartbeatTimer !== null) {
      (this.window.clearInterval || globalThis.clearInterval)(this._heartbeatTimer);
      this._heartbeatTimer = null;
    }
  }

  async heartbeat() {
    if (this.state !== "recording" || !this.session?.session_id || this._heartbeatInFlight) return this.status;
    this._heartbeatInFlight = true;
    try {
      await this._clockProbe("heartbeat", this.limits.heartbeatTimeoutMs);
      const payload = await this._requestJson(`/api/companion-captures/${encodeURIComponent(this.session.session_id)}/heartbeat`, {
        method: "POST",
        timeoutMs: this.limits.heartbeatTimeoutMs,
        body: { phone_capture_id: this.phoneCaptureId, clock_probes: this.clockProbes.slice(-1) },
      });
      this._adopt(payload);
      if (TERMINAL_STATUSES.has(statusValue(payload)) || !statusHealthy(payload, "video_status", { required: true }) || !statusHealthy(payload, "tracking_status", { required: true }) || !this._membersReady(payload)) {
        throw new CompanionCaptureError("The static companion camera or tracking stream became unavailable.", { code: "companion_stream_lost", status: payload?.status || null });
      }
      this.lastHeartbeat = {
        client_monotonic_ms: this._nowMonotonicMs(),
        client_epoch_ms: this._nowEpochMs(),
        status: payload?.status || null,
        video_status: payload?.video_status || null,
        tracking_status: payload?.tracking_status || null,
        tracking_sequence: payload?.tracking_sequence ?? null,
        static_timestamp: payload?.static_timestamp ?? null,
      };
      this._emitState();
      return this.status;
    } catch (error) {
      this._handleFailure(error, "companion_heartbeat_failed");
      throw this.error;
    } finally {
      this._heartbeatInFlight = false;
    }
  }

  _handleFailure(error, code = "companion_capture_failed") {
    this._stopHeartbeat();
    this.error = error instanceof CompanionCaptureError && error.code === code
      ? error
      : new CompanionCaptureError(error?.message || String(error), { code, status: error?.status || null, cause: error });
    this.state = "failed";
    this._emitState();
    if (!this._failureNotified) {
      this._failureNotified = true;
      try { this.onFailure(this.error, this.status); } catch (_) { /* UI failure handlers cannot break capture cleanup. */ }
    }
    return this.error;
  }

  async addMarker(label = "user_event") {
    if (this.state !== "recording" || !this.session?.session_id) throw new CompanionCaptureError("Markers are available only while paired recording is active.", { code: "marker_not_recording" });
    if (this.markers.length >= Number(this.limits.maxMarkers)) {
      this.markerLimitReached = true;
      this._emitState();
      throw new CompanionCaptureError("The marker limit was reached; start a new paired capture to record more markers.", { code: "marker_limit" });
    }
    const marker = {
      marker_id: newId(this.crypto),
      label: safeString(label, "user_event").slice(0, 80) || "user_event",
      client_monotonic_ms: this._nowMonotonicMs(),
      client_epoch_ms: this._nowEpochMs(),
      status: "pending",
      receipt_clock: null,
      error: null,
    };
    this.markers.push(marker);
    this._emitState();
    try {
      const payload = await this._requestJson(`/api/companion-captures/${encodeURIComponent(this.session.session_id)}/markers`, {
        method: "POST",
        timeoutMs: this.limits.heartbeatTimeoutMs,
        body: { phone_capture_id: this.phoneCaptureId, ...markerRequest(marker) },
      });
      marker.status = "received";
      marker.receipt_clock = copyJsonValue(payload?.receipt_clock || payload?.clock || payload || null);
      marker.error = null;
      this._emitState();
      return publicMarker(marker);
    } catch (error) {
      marker.status = "pending";
      marker.error = safeString(error?.message, "marker delivery failed");
      this._emitState();
      throw error;
    }
  }

  async retryMarkers() {
    const pending = this.markers.filter((marker) => marker.status !== "received").slice(0, Number(this.limits.maxMarkerRetriesPerCall));
    for (const marker of pending) {
      try {
        const payload = await this._requestJson(`/api/companion-captures/${encodeURIComponent(this.session?.session_id || "")}/markers`, {
          method: "POST",
          timeoutMs: this.limits.heartbeatTimeoutMs,
          body: { phone_capture_id: this.phoneCaptureId, ...markerRequest(marker) },
        });
        marker.status = "received";
        marker.receipt_clock = copyJsonValue(payload?.receipt_clock || payload?.clock || payload || null);
        marker.error = null;
      } catch (error) {
        marker.error = safeString(error?.message, "marker delivery failed");
      }
      this._emitState();
    }
    return this.status;
  }

  async _stopRequest(reason, { bestEffort = false } = {}) {
    if (!this.session?.session_id) return null;
    try {
      return await this._requestJson(`/api/companion-captures/${encodeURIComponent(this.session.session_id)}/stop`, {
        method: "POST",
        timeoutMs: this.limits.stopTimeoutMs,
        body: {
          phone_capture_id: this.phoneCaptureId,
          reason: safeString(reason, "user") || "user",
          clock_probes: this.clockProbes.slice(-5),
          markers: this.markers.map(markerRequest),
        },
      });
    } catch (error) {
      if (bestEffort) return null;
      throw error;
    }
  }

  async _pollFinalization() {
    if (!this.session?.session_id) return null;
    const deadline = this._nowMonotonicMs() + Number(this.limits.finalizationTimeoutMs);
    let payload = this.session;
    while (statusValue(payload) === "finalizing" && this._nowMonotonicMs() < deadline) {
      const remainingBeforeSleep = Math.max(0, deadline - this._nowMonotonicMs());
      await new Promise((resolve) => (this.window.setTimeout || globalThis.setTimeout)(resolve, Math.min(this.limits.finalizationPollMs, remainingBeforeSleep)));
      const remainingForRequest = Math.max(1, deadline - this._nowMonotonicMs());
      if (remainingForRequest <= 1) break;
      payload = await this._requestJson(`/api/companion-captures/${encodeURIComponent(this.session.session_id)}`, { timeoutMs: Math.min(this.limits.stopTimeoutMs, remainingForRequest) });
      this._adopt(payload);
      this._emitState();
    }
    return payload;
  }

  async stop(reason = "user") {
    if (this._stoppingPromise) return this._stoppingPromise;
    if (!this.session?.session_id || (!this.hasLiveSession && !this.needsServerStop)) return this.status;
    this._stoppingPromise = (async () => {
      this._stopHeartbeat();
      this.state = "stopping";
      this._emitState();
      // Final probes are best effort so a clock endpoint outage cannot lose
      // the phone TAR or prevent the idempotent static stop request.
      await this._collectClockProbes("stop", 5, { strict: false });
      try {
        const payload = await this._stopRequest(reason);
        this._adopt(payload);
        this.state = statusValue(payload) === "finalizing" ? "finalizing" : (statusValue(payload) === "failed" ? "failed" : "stopped");
        this._emitState();
        if (this.state === "finalizing") {
          try {
            const finalPayload = await this._pollFinalization();
            this._adopt(finalPayload);
            this.state = statusValue(finalPayload) === "stopped" ? "stopped" : statusValue(finalPayload) === "failed" ? "failed" : "finalizing";
            if (this.state === "finalizing") this.error = new CompanionCaptureError("Static capture finalization is still pending; the phone bundle remains available for retry or download.", { code: "companion_finalization_pending" });
          } catch (error) {
            this._setError(error, "companion_finalization_poll_failed");
            this.state = "finalizing";
          }
        }
      } catch (error) {
        this._setError(error, "companion_stop_failed");
        this.state = "failed";
        try { this.onFailure(this.error, this.status); } catch (_) { /* preserve phone finalization */ }
      }
      this._emitState();
      return this.status;
    })();
    try {
      return await this._stoppingPromise;
    } finally {
      this._stoppingPromise = null;
    }
  }

  companionContext() {
    return copyJsonValue(this._context());
  }
}

export default CompanionCapture;

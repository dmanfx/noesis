const TRANSIENT_ERRORS = new Set(['capture_event_busy', 'rate_limited']);

const uniqueCameraIds = (cameras) => [...new Set(
  (Array.isArray(cameras) ? cameras : [])
    .map((camera) => String(camera || '').trim())
    .filter(Boolean),
)];

const responseRequestId = (payload) => String(
  payload?.request_id ?? payload?.requestId ?? '',
).trim();

const responseError = (payload) => String(payload?.error || '').trim();

/**
 * Coordinates the dashboard's floorplan bootstrap without owning transport.
 * Bootstrap is deliberately cache-only. A cache miss is terminal for that
 * camera and never escalates into inference; only an explicit user refresh may
 * start a fresh depth/floorplan sequence. At most one cache request is active
 * across all cameras.
 */
export class FloorplanBootstrapCoordinator {
  constructor({
    requestIdPrefix = 'bev-bootstrap',
    maxTransientRetries = 2,
    retryDelayMs = 2_200,
  } = {}) {
    this.requestIdPrefix = String(requestIdPrefix || 'bev-bootstrap');
    this.maxTransientRetries = Math.max(0, Number(maxTransientRetries) || 0);
    this.retryDelayMs = Math.max(0, Number(retryDelayMs) || 0);
    this.run = 0;
    this.sequence = 0;
    this.queue = [];
    this.active = null;
    this.pendingRestart = null;
  }

  restart(cameras) {
    const normalized = uniqueCameraIds(cameras);
    if (this.active) {
      // Coalesce reconnect/calibration churn. The active backend request cannot
      // be cancelled, so finish it before starting the newest requested run.
      this.pendingRestart = normalized;
      return null;
    }
    return this.startRun(normalized);
  }

  cancel() {
    this.run += 1;
    this.queue = [];
    this.active = null;
    this.pendingRestart = null;
  }

  isCurrentRequest(requestId) {
    return Boolean(this.active && this.active.request.requestId === requestId);
  }

  handleResponse(payload, { renderable = false } = {}) {
    const requestId = responseRequestId(payload);
    if (!requestId || !this.active || requestId !== this.active.request.requestId) {
      return { handled: false, action: null };
    }

    const active = this.active;
    const error = responseError(payload);
    if (renderable && !error) {
      return this.finishCamera({ completedCamera: active.camera });
    }

    if (error === 'no_cached_floorplan') {
      return this.finishCamera({
        failedCamera: active.camera,
        error,
      });
    }

    if (TRANSIENT_ERRORS.has(error) && active.retryCount < this.maxTransientRetries) {
      const retryCount = active.retryCount + 1;
      return {
        handled: true,
        action: this.activate(active.camera, retryCount, this.retryDelayMs),
      };
    }

    return this.finishCamera({
      failedCamera: active.camera,
      error: error || 'invalid_floorplan_response',
    });
  }

  snapshot() {
    return {
      run: this.run,
      queue: [...this.queue],
      active: this.active ? {
        camera: this.active.camera,
        retryCount: this.active.retryCount,
        requestId: this.active.request.requestId,
      } : null,
      pendingRestart: this.pendingRestart ? [...this.pendingRestart] : null,
    };
  }

  startRun(cameras) {
    this.run += 1;
    this.queue = [...cameras];
    this.active = null;
    this.pendingRestart = null;
    return this.activateNextCamera();
  }

  activateNextCamera() {
    const camera = this.queue.shift();
    if (!camera) {
      this.active = null;
      return null;
    }
    return this.activate(camera, 0, 0);
  }

  activate(camera, retryCount, delayMs) {
    const requestId = [
      this.requestIdPrefix,
      this.run,
      ++this.sequence,
      'cache',
      camera,
    ].join('-');
    const request = {
      camera,
      requestId,
      maxAgeSec: 600,
      gridResM: 0.04,
      maxExtentM: 20,
      cacheOnly: true,
    };
    this.active = { camera, retryCount, request };
    return { request, delayMs };
  }

  finishCamera({ completedCamera = null, failedCamera = null, error = null } = {}) {
    this.active = null;
    let action = null;
    if (this.pendingRestart) {
      const cameras = this.pendingRestart;
      this.pendingRestart = null;
      action = this.startRun(cameras);
    } else {
      action = this.activateNextCamera();
    }
    return {
      handled: true,
      action,
      completedCamera,
      failedCamera,
      error,
    };
  }
}

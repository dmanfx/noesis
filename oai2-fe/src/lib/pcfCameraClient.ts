import type { PcfCameraProjection, PcfCameraRaster } from './pcfCameraProjection';

export type LoadedPcfCamera = PcfCameraRaster & { priorId: string; hasColor: boolean };
type Job = { projection: PcfCameraProjection; worker?: Worker; cancelled: boolean;
  resolve: (value: LoadedPcfCamera) => void; reject: (error: Error) => void; timer?: number };
const pending: Job[] = [], running = new Set<Job>();

function pump() {
  while (running.size < 2 && pending.length) {
    const job = pending.shift()!; if (job.cancelled) continue;
    running.add(job);
    const finish = (error?: Error, result?: LoadedPcfCamera) => {
      if (!running.delete(job)) return;
      if (job.timer !== undefined) window.clearTimeout(job.timer);
      job.worker?.terminate();
      if (!job.cancelled) { if (error) job.reject(error); else job.resolve(result!); }
      pump();
    };
    try {
      const worker = job.worker = new Worker(new URL('../workers/pcfCameraWorker.ts', import.meta.url), { type: 'module' });
      job.timer = window.setTimeout(() => finish(new Error('pcf_camera_load_timeout')), 35_000);
      worker.onmessage = ({ data }) => data.ok ? finish(undefined, data) : finish(new Error(data.error));
      worker.onerror = () => finish(new Error('pcf_camera_worker_failed'));
      worker.postMessage({ projection: job.projection, origin: window.location.origin });
    } catch (error) { finish(error instanceof Error ? error : new Error(String(error))); }
  }
}

/** At most two static geometry jobs; never blocks the live media callback. */
export function loadPcfCamera(projection: PcfCameraProjection) {
  let job: Job;
  const promise = new Promise<LoadedPcfCamera>((resolve, reject) => {
    job = { projection, resolve, reject, cancelled: false };
    if (pending.length + running.size >= 32) { reject(new Error('pcf_camera_queue_full')); return; }
    pending.push(job); pump();
  });
  return { promise, cancel() {
    if (!job || job.cancelled) return;
    job.cancelled = true;
    const index = pending.indexOf(job); if (index >= 0) pending.splice(index, 1);
    if (job.timer !== undefined) window.clearTimeout(job.timer);
    job.worker?.terminate(); running.delete(job); pump();
  } };
}

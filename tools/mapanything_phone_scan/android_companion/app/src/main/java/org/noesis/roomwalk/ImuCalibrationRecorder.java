package org.noesis.roomwalk;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.SystemClock;
import org.json.JSONObject;
import java.io.File;
import java.io.IOException;

/** Foreground-only long IMU acquisition. It never opens a camera or encoder. */
public final class ImuCalibrationRecorder {
    public static final long DEFAULT_DURATION_SECONDS = 60;
    public static final long MAX_DURATION_SECONDS = 3 * 60 * 60;
    private static final long STORAGE_RESERVE_BYTES = 64L * 1024 * 1024;
    public interface Listener {
        void onImuState(String state, JSONObject details);
        void onImuStopped(File directory, JSONObject manifest);
        void onImuError(String reason);
    }
    private final Context context;
    private final Listener listener;
    private final SensorManager sensors;
    private final Handler main = new Handler(Looper.getMainLooper());
    private final HandlerThread thread = new HandlerThread("RoomWalkImuCalibration");
    private final Handler control;
    private Session current;
    private boolean closing;

    public ImuCalibrationRecorder(Context context, Listener listener) {
        this.context = context.getApplicationContext(); this.listener = listener;
        sensors = (SensorManager) this.context.getSystemService(Context.SENSOR_SERVICE);
        thread.start(); control = new Handler(thread.getLooper());
    }
    public void start(File directory, long durationSeconds) {
        control.post(() -> {
            if (closing || current != null) { error("IMU recorder is unavailable or already active"); return; }
            if (durationSeconds < 60 || durationSeconds > MAX_DURATION_SECONDS) {
                error("Choose 1–180 minutes: short diagnostics or a three-hour stationary noise recording"); return;
            }
            Session session = new Session(directory, durationSeconds); current = session;
            try {
                if (!directory.isDirectory() && !directory.mkdirs()) throw new IOException("Cannot create IMU capture directory");
                if (new File(directory, "imu_capture_manifest.json").exists()) throw new IOException("IMU capture already exists");
                if (directory.getUsableSpace() < STORAGE_RESERVE_BYTES) throw new IOException("At least 64 MiB free storage is required");
                session.accel = NativeSensors.accelerometer(sensors); session.gyro = NativeSensors.gyroscope(sensors);
                if (session.accel == null || session.gyro == null) throw new IOException("Both accelerometer and gyroscope are required");
                session.writer = new ImuCalibrationWriter(directory, new ImuCalibrationWriter.Listener() {
                    public void onFailure(String reason) { control.post(() -> stopSession(session, reason, true)); }
                    public void onFinished() { control.post(() -> finishSession(session)); }
                });
                session.startNs = SystemClock.elapsedRealtimeNanos();
                session.initialManifest = manifest(session, "recording");
                BundleTools.writeJson(new File(directory, "imu_capture_manifest.json"), session.initialManifest);
                session.writer.start(); session.writerStarted = true;
                session.events = new SensorEventListener() {
                    public void onSensorChanged(SensorEvent event) {
                        if (current != session || session.stopping) return;
                        boolean accel = event.sensor.getType() == session.accel.getType();
                        session.writer.offer(new ImuCalibrationWriter.Sample(accel, event.timestamp, event.values, event.accuracy, SystemClock.elapsedRealtimeNanos()));
                    }
                    public void onAccuracyChanged(Sensor sensor, int accuracy) {}
                };
                boolean accel = sensors.registerListener(session.events, session.accel, 1_000_000 / CaptureEngine.SENSOR_RATE_HZ, 0, control);
                boolean gyro = sensors.registerListener(session.events, session.gyro, 1_000_000 / CaptureEngine.SENSOR_RATE_HZ, 0, control);
                if (!accel || !gyro) throw new IOException("Could not register both IMU streams");
                state("recording", progress(session));
                control.postDelayed(session.tick, 1000);
            } catch (Exception failure) {
                stopSession(session, "start_failed: " + failure.getMessage(), true);
                if (!session.writerStarted) finishSession(session);
            }
        });
    }
    public void stop(String reason) { control.post(() -> { if (current != null) stopSession(current, reason, false); }); }
    public void close() {
        control.post(() -> { closing = true; if (current != null) stopSession(current, "app_closed", false); else thread.quitSafely(); });
    }
    private void tick(Session session) {
        if (current != session || session.stopping) return;
        long now = SystemClock.elapsedRealtimeNanos();
        if (now - session.startNs >= session.durationSeconds * 1_000_000_000L) { stopSession(session, "duration_reached", false); return; }
        if (session.directory.getUsableSpace() < STORAGE_RESERVE_BYTES) { stopSession(session, "low_storage", true); return; }
        if (now - session.startNs > 5_000_000_000L) {
            if (session.writer.accelerometer.count == 0 || session.writer.gyroscope.count == 0) { stopSession(session, "missing_imu_samples", true); return; }
            if (now - session.writer.accelerometer.lastReceived > 2_000_000_000L || now - session.writer.gyroscope.lastReceived > 2_000_000_000L) {
                stopSession(session, "imu_stream_stalled", true); return;
            }
        }
        state("recording", progress(session));
        control.postDelayed(session.tick, 1000);
    }
    private void stopSession(Session session, String reason, boolean failure) {
        if (current != session) return;
        if (failure) { session.failed = true; session.reason = bounded(reason); }
        if (session.stopping) return;
        session.stopping = true; session.endNs = SystemClock.elapsedRealtimeNanos();
        if (!failure) session.reason = bounded(reason);
        control.removeCallbacks(session.tick);
        if (session.events != null) sensors.unregisterListener(session.events);
        state("saving", progress(session));
        if (session.writer != null) session.writer.finish();
    }
    private void finishSession(Session session) {
        if (current != session) return;
        if (!session.stopping) stopSession(session, "writer_stopped", true);
        try {
            boolean clean = session.writer != null && session.writer.isFinalized() && !session.failed
                    && session.writer.droppedRecords.get() == 0 && session.writer.accelerometer.count >= 2 && session.writer.gyroscope.count >= 2;
            String status = clean && "duration_reached".equals(session.reason) ? "complete" : session.writer != null && session.writer.isFinalized() ? "partial" : "failed";
            JSONObject result = manifest(session, status);
            BundleTools.writeJson(new File(session.directory, "imu_capture_manifest.json"), result);
            main.post(() -> listener.onImuStopped(session.directory, result));
        } catch (Exception failure) { error("IMU files retained; could not finalize manifest: " + failure.getMessage()); }
        finally { current = null; if (closing) thread.quitSafely(); }
    }
    private JSONObject progress(Session session) {
        try {
            return new JSONObject().put("elapsed_seconds", Math.max(0, ((session.endNs == 0 ? SystemClock.elapsedRealtimeNanos() : session.endNs) - session.startNs) / 1_000_000_000L))
                    .put("expected_duration_s", session.durationSeconds).put("accel_samples", session.writer == null ? 0 : session.writer.accelerometer.count)
                    .put("gyro_samples", session.writer == null ? 0 : session.writer.gyroscope.count)
                    .put("bytes", session.writer == null ? 0 : session.writer.bytes());
        } catch (Exception failure) { return new JSONObject(); }
    }
    private JSONObject manifest(Session session, String status) throws Exception {
        JSONObject streams = new JSONObject();
        if (session.accel != null) streams.put("accelerometer", stream(session, session.accel, true));
        if (session.gyro != null) streams.put("gyroscope", stream(session, session.gyro, false));
        return new JSONObject().put("schema", "noesis.phone_imu_calibration.v1").put("capture_id", session.directory.getName())
                .put("device", NativeSensors.device(context)).put("clock", "android.elapsedRealtimeNanos").put("axes", NativeSensors.AXES)
                .put("timestamp_unit", "ns").put("expected_duration_s", session.durationSeconds)
                .put("actual_duration_s", session.startNs == 0 ? 0 : Math.max(0, ((session.endNs == 0 ? SystemClock.elapsedRealtimeNanos() : session.endNs) - session.startNs) * 1e-9))
                .put("start_elapsed_realtime_ns", session.startNs).put("end_elapsed_realtime_ns", session.endNs == 0 ? JSONObject.NULL : session.endNs)
                .put("status", status).put("stop_reason", session.reason).put("dropped_records", session.writer == null ? 0 : session.writer.droppedRecords.get())
                .put("streams", streams).put("stationarity_verified", false).put("noise_calibration_complete", false)
                .put("limits", new JSONObject().put("maximum_rows_per_stream", ImuCalibrationWriter.MAX_ROWS_PER_STREAM)
                        .put("maximum_storage_bytes", ImuCalibrationWriter.MAX_BYTES).put("writer_queue_capacity", ImuCalibrationWriter.QUEUE_CAPACITY))
                .put("recording_kind", "imu_only").put("camera_opened", false).put("video_recorded", false);
    }
    private JSONObject stream(Session session, Sensor sensor, boolean accel) throws Exception {
        JSONObject result = NativeSensors.describe(context, sensor, accel ? "accel.csv" : "gyro.csv", accel ? "m/s^2" : "rad/s");
        ImuCalibrationWriter.Stats stats = session.writer == null ? new ImuCalibrationWriter.Stats() : accel ? session.writer.accelerometer : session.writer.gyroscope;
        return result.put("sample_count", stats.count).put("first_timestamp_ns", stats.firstTimestamp).put("last_timestamp_ns", stats.lastTimestamp)
                .put("nonmonotonic_timestamp_count", stats.nonmonotonic).put("minimum_interval_ns", stats.minimumInterval == Long.MAX_VALUE ? 0 : stats.minimumInterval)
                .put("maximum_interval_ns", stats.maximumInterval).put("unreliable_accuracy_sample_count", stats.unreliable)
                .put("observed_rate_hz", stats.count > 1 && stats.lastTimestamp > stats.firstTimestamp ? (stats.count - 1) * 1e9 / (stats.lastTimestamp - stats.firstTimestamp) : 0)
                .put("bytes", stats.bytes).put("sha256", stats.sha256 == null ? JSONObject.NULL : stats.sha256);
    }
    private void state(String state, JSONObject details) { main.post(() -> listener.onImuState(state, details)); }
    private void error(String reason) { main.post(() -> listener.onImuError(bounded(reason))); }
    private static String bounded(String reason) { return reason == null ? "unknown" : reason.substring(0, Math.min(reason.length(), 256)); }
    private final class Session {
        final File directory; final long durationSeconds; final Runnable tick;
        Sensor accel, gyro; SensorEventListener events; ImuCalibrationWriter writer;
        JSONObject initialManifest; long startNs, endNs; boolean stopping, failed, writerStarted;
        String reason = "recording";
        Session(File directory, long durationSeconds) { this.directory = directory; this.durationSeconds = durationSeconds; tick = () -> tick(this); }
    }
}

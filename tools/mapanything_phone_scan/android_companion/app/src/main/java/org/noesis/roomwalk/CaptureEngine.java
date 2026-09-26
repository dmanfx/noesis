package org.noesis.roomwalk;

import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureFailure;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.OutputConfiguration;
import android.hardware.camera2.params.SessionConfiguration;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.MediaCodec;
import android.media.CamcorderProfile;
import android.media.EncoderProfiles;
import android.media.MediaCodecInfo;
import android.media.MediaCodecList;
import android.media.MediaFormat;
import android.media.MediaMuxer;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.StatFs;
import android.os.SystemClock;
import android.util.Range;
import android.util.Size;
import android.view.Surface;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/** Camera2 surface recording with acquisition-clock evidence and no resolution fallback. */
public final class CaptureEngine implements AutoCloseable {
    public interface Listener {
        void onState(String state, JSONObject details);
        void onError(String message, JSONObject details);
        void onStopped(JSONObject result);
    }

    public static final int WIDTH = 7680;
    public static final int HEIGHT = 4320;
    public static final int FPS = 30;
    public static final int SENSOR_RATE_HZ = 200;
    public static final long MAX_DURATION_MS = 10 * 60 * 1000L;
    public static final long SHORT_TEST_DURATION_MS = 10_000;
    public static final long MAX_VIDEO_BYTES = 6L * 1024 * 1024 * 1024;
    public static final long MIN_FREE_BYTES = 512L * 1024 * 1024;
    private static final int MAX_FRAMES = 20000;
    private static final int MAX_SENSOR_SAMPLES = 160000;
    private static final int METADATA_QUEUE_CAPACITY = 8192;
    private static final int REQUESTED_BITRATE = 128_000_000;
    private static final long SENSOR_START_TIMEOUT_MS = 5000;
    private static final long SENSOR_STALL_TIMEOUT_NS = TimeUnit.SECONDS.toNanos(2);
    private static final int PROBE_MAX_CAMERAS = 16;
    private static final int PROBE_MAX_PHYSICAL_PER_CAMERA = 8;
    private static final int PROBE_MAX_PHYSICAL_TOTAL = 16;
    private static final int PROBE_MAX_LIST_VALUES = 64;
    private static final int PROBE_MAX_SIZE_ROWS = 1024;
    private static final int PROBE_MAX_BYTES = 448 * 1024; // Reserve space for direct-session diagnostics and nested JSON indentation.
    private static final String AXES = "android_device_x_right_y_up_z_out_of_screen";
    private static final String SENSOR_HEADER =
            "timestamp_ns,x,y,z,bias_x,bias_y,bias_z,accuracy,received_elapsed_realtime_ns\n";

    private final Context context;
    private final Listener listener;
    private final CameraManager cameras;
    private final SensorManager sensors;
    private final Handler main = new Handler(Looper.getMainLooper());
    private final HandlerThread controlThread = new HandlerThread("RoomWalkCamera");
    private final HandlerThread sensorThread = new HandlerThread("RoomWalkImu");
    private final Handler control;
    private final Handler sensorHandler;
    private volatile Session current;
    private volatile boolean closing;

    public CaptureEngine(Context context, Listener listener) {
        this.context = context.getApplicationContext();
        this.listener = listener;
        cameras = (CameraManager) this.context.getSystemService(Context.CAMERA_SERVICE);
        sensors = (SensorManager) this.context.getSystemService(Context.SENSOR_SERVICE);
        controlThread.start();
        sensorThread.start();
        control = new Handler(controlThread.getLooper());
        sensorHandler = new Handler(sensorThread.getLooper());
    }

    /** Read-only advertised capability probe; session configuration still has to succeed. */
    public JSONObject probe() {
        JSONObject result = object("schema", "noesis.phone_capture.android_capabilities.v1",
                "android_api_level", Build.VERSION.SDK_INT,
                "device", device(), "required_width", WIDTH, "required_height", HEIGHT,
                "required_fps", FPS, "minimum_recording_api_level", 33,
                "max_duration_ms", MAX_DURATION_MS, "max_video_bytes", MAX_VIDEO_BYTES,
                "minimum_free_bytes", MIN_FREE_BYTES,
                "advertised_capabilities_only", true,
                "diagnostic_revision", 3,
                "recording_path", "default_sensor_pixel_mode_top_level_camera",
                "additional_modes_are_diagnostic_only", true);
        JSONArray rows = new JSONArray();
        ProbeBudget budget = new ProbeBudget();
        try {
            String[] ids = cameras.getCameraIdList();
            List<CameraCharacteristics> topLevelCharacteristics = new ArrayList<>();
            put(result, "camera_count", ids.length);
            put(result, "cameras_truncated", ids.length > PROBE_MAX_CAMERAS);
            for (int i = 0; i < Math.min(ids.length, PROBE_MAX_CAMERAS); i++) {
                String id = ids[i];
                JSONObject row;
                CameraCharacteristics characteristics = null;
                try {
                    Candidate candidate = candidate(id);
                    characteristics = candidate.characteristics;
                    row = candidate.json;
                } catch (Exception error) {
                    row = object("id", id, "label", "Camera " + id,
                            "supported8k", false, "reason", describe(error));
                }
                try {
                    if (characteristics == null) characteristics = cameras.getCameraCharacteristics(id);
                    put(row, "diagnostics", cameraDiagnostics(id, characteristics, budget, true));
                } catch (Exception error) {
                    put(row, "diagnostics_error", probeError(error));
                }
                topLevelCharacteristics.add(characteristics);
                rows.put(row);
            }
            // Give top-level camera modes first access to the shared detail budget.
            for (int i = 0; i < topLevelCharacteristics.size(); i++) {
                CameraCharacteristics characteristics = topLevelCharacteristics.get(i);
                if (characteristics != null) put(rows.optJSONObject(i), "physical_cameras", physicalDiagnostics(characteristics, ids, budget));
            }
        } catch (Exception error) {
            put(result, "error", describe(error));
        }
        put(result, "cameras", rows);
        put(result, "sensors", object("accelerometer", sensorInfo(selectAccelerometer(), "accel.csv", "m/s^2"),
                "gyroscope", sensorInfo(selectGyroscope(), "gyro.csv", "rad/s")));
        put(result, "diagnostic_limits", object("top_level_cameras", PROBE_MAX_CAMERAS,
                "physical_cameras_per_parent", PROBE_MAX_PHYSICAL_PER_CAMERA,
                "physical_cameras_total", PROBE_MAX_PHYSICAL_TOTAL,
                "values_per_list", PROBE_MAX_LIST_VALUES, "stream_size_rows_total", PROBE_MAX_SIZE_ROWS,
                "vendor_key_scan_per_camera", 512, "vendor_keys_per_camera", 32, "vendor_numeric_values_total", 2048,
                "maximum_report_bytes", PROBE_MAX_BYTES));
        put(result, "diagnostic_size_rows", PROBE_MAX_SIZE_ROWS - budget.sizeRowsRemaining);
        put(result, "diagnostic_physical_camera_rows", PROBE_MAX_PHYSICAL_TOTAL - budget.physicalRowsRemaining);
        put(result, "report_truncated", false);
        // Vendor strings/errors are outside our control. Retain readiness rows first if
        // the bounded detailed inventories still exceed the upload contract's byte cap.
        for (int i = rows.length() - 1; i >= 0 && jsonBytes(result) > PROBE_MAX_BYTES; i--) {
            JSONObject row = rows.optJSONObject(i);
            row.remove("physical_cameras");
            row.remove("diagnostics");
            put(row, "diagnostics_truncated", true);
            put(result, "report_truncated", true);
        }
        if (jsonBytes(result) > PROBE_MAX_BYTES) {
            return object("schema", "noesis.phone_capture.android_capabilities.v1", "diagnostic_revision", 3,
                    "android_api_level", Build.VERSION.SDK_INT, "device", new JSONObject(),
                    "error", "Phone capability report exceeded its byte limit; vendor fields were omitted",
                    "report_truncated", true, "cameras", new JSONArray());
        }
        return result;
    }

    public void start(File sessionDir, Surface previewSurface, String cameraId, boolean shortTest, long standardUseCase) {
        start(sessionDir, previewSurface, cameraId, shortTest, standardUseCase, true);
    }

    /** Start a capture, optionally binding the saved measured focus used by calibration takes. */
    public void start(File sessionDir, Surface previewSurface, String cameraId, boolean shortTest, long standardUseCase, boolean useCalibrationFocus) {
        control.post(() -> {
            if (closing || current != null) {
                error("Recorder is already active or closed", object("state", "busy"));
                return;
            }
            Session session = new Session(sessionDir, previewSurface, shortTest, standardUseCase, useCalibrationFocus);
            current = session;
            state("starting", object("session_dir", sessionDir.getAbsolutePath()));
            try {
                prepare(session, cameraId);
            } catch (Exception failure) {
                fail(session, "start_failed", failure);
            }
        });
    }

    public void stop() {
        control.post(() -> {
            if (current != null) requestStop(current, "user_stop", false);
        });
    }

    @Override public void close() {
        closing = true;
        control.post(() -> {
            if (current != null) requestStop(current, "app_closed", true);
            else shutdownThreads();
        });
    }

    private Candidate candidate(String id) throws Exception {
        return candidate(id, true);
    }

    private Candidate candidate(String id, boolean requireCalibrationFocus) throws Exception {
        CameraCharacteristics characteristics = cameras.getCameraCharacteristics(id);
        Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
        Integer source = characteristics.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE);
        StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        Size wanted = new Size(WIDTH, HEIGHT);
        Size[] videoSizes = map == null ? null : map.getOutputSizes(MediaCodec.class);
        boolean sizeAvailable = contains(videoSizes, wanted);
        Size maximumVideoSize = largestSize(videoSizes);
        long minDurationNs = sizeAvailable ? map.getOutputMinFrameDuration(MediaCodec.class, wanted) : 0;
        Range<Integer>[] fpsRanges = characteristics.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
        boolean fixedFps = fpsRanges != null && Arrays.asList(fpsRanges).contains(new Range<>(FPS, FPS));
        EncoderChoice encoder = chooseEncoder();
        Size preview = choosePreview(map);
        JSONArray reasons = new JSONArray();
        if (Build.VERSION.SDK_INT < 33) reasons.put("Android 13/API 33 is required to set the camera output timestamp base explicitly");
        if (facing == null || facing != CameraCharacteristics.LENS_FACING_BACK) reasons.put("Only a rear camera is supported");
        if (!sizeAvailable) reasons.put("The companion's standard Camera2 video path does not advertise 7680x4320"
                + (maximumVideoSize == null ? "; no MediaCodec output sizes are listed" : "; largest listed output is " + maximumVideoSize)
                + ". Stock-camera and other sensor modes are checked separately in the diagnostic report");
        if (sizeAvailable && (minDurationNs <= 0 || minDurationNs > 33_333_334L)) reasons.put("Camera2 does not advertise an 8K frame duration supporting 30 FPS");
        if (!fixedFps) reasons.put("Camera2 does not advertise a fixed 30 FPS AE range");
        if (source == null || source != CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME) reasons.put("Camera sensor clock is not REALTIME");
        if (encoder == null) reasons.put("No hardware surface encoder advertises 7680x4320 at 30 FPS");
        if (preview == null) reasons.put("Camera2 does not advertise a preview size within 1920x1080");
        if (selectAccelerometer() == null || selectGyroscope() == null) reasons.put("Accelerometer and gyroscope are both required");
        boolean directTestEligible = Build.VERSION.SDK_INT >= 35 && !sizeAvailable
                && facing != null && facing == CameraCharacteristics.LENS_FACING_BACK
                && fixedFps && source != null && source == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
                && encoder != null && preview != null && selectAccelerometer() != null && selectGyroscope() != null;
        JSONObject json = object("id", id, "label", facingLabel(facing) + " camera " + id,
                "supported8k", reasons.length() == 0,
                "direct_session_test_eligible", directTestEligible,
                "reason", reasons.length() == 0 ? "Advertised 8K/30 support; a real capture session must still be tested" : reasons.join("; ").replace("\"", ""),
                "reasons", reasons, "lens_facing", facing, "timestamp_source", timestampSource(source),
                "default_8k_size_available", sizeAvailable,
                "maximum_default_video_size", sizeJson(maximumVideoSize),
                "min_frame_duration_ns", minDurationNs,
                "fixed_30_fps_available", fixedFps,
                "preview_width", preview == null ? 0 : preview.getWidth(), "preview_height", preview == null ? 0 : preview.getHeight(),
                "encoder", encoder == null ? JSONObject.NULL : encoder.json());
        diagnosticField(json, "sensor_orientation_degrees", () -> characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION));
        diagnosticField(json, "physical_camera_ids", () -> {
            List<String> ids = new ArrayList<>(characteristics.getPhysicalCameraIds());
            put(json, "physical_camera_id_count", ids.size());
            put(json, "physical_camera_ids_truncated", ids.size() > PROBE_MAX_LIST_VALUES);
            return boundedStrings(ids);
        });
        diagnosticField(json,"focus_control",()->FocusSettings.metadata(requireCalibrationFocus?FocusSettings.read(context,id):null));
        if (directTestEligible) {
            long previewUseCase = 0;
            long[] useCases = characteristics.get(CameraCharacteristics.SCALER_AVAILABLE_STREAM_USE_CASES);
            if (useCases != null) for (long value : useCases) if (value == 1) previewUseCase = 1;
            put(json, "standard_preview_use_case", previewUseCase);
            JSONObject recorded = CapturePreflight.find(new File(context.getExternalFilesDir(null), "captures"), json, Build.FINGERPRINT, requireCalibrationFocus);
            if (recorded != null) put(json, "recorded_preflight", recorded);
            // Keep automatic walk-mode proof distinct from a saved board-focus
            // proof so changing the capture purpose cannot reuse another route.
            if (requireCalibrationFocus) {
                JSONObject automatic = CapturePreflight.find(new File(context.getExternalFilesDir(null), "captures"), json, Build.FINGERPRINT, false);
                if (automatic != null) put(json, "automatic_recorded_preflight", automatic);
            }
        }
        return new Candidate(id, characteristics, encoder, json, reasons.length() == 0, directTestEligible);
    }

    private JSONObject cameraDiagnostics(String id, CameraCharacteristics c, ProbeBudget budget, boolean topLevel) {
        JSONObject out = object("diagnostic_only", true);
        diagnosticField(out, "hardware_level", () -> c.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL));
        diagnosticField(out, "lens_facing", () -> c.get(CameraCharacteristics.LENS_FACING));
        diagnosticField(out, "sensor_orientation_degrees", () -> c.get(CameraCharacteristics.SENSOR_ORIENTATION));
        diagnosticField(out, "timestamp_source", () -> timestampSource(c.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)));
        diagnosticField(out, "available_capabilities", () -> boundedInts(c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)));
        diagnosticField(out, "ae_target_fps_ranges", () -> fpsRanges(c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)));
        diagnosticField(out, "default_stream_map", () -> streamMapDiagnostics(c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP), budget));
        if (Build.VERSION.SDK_INT >= 31) {
            diagnosticField(out, "sensor_pixel_mode_request_available", () -> c.getAvailableCaptureRequestKeys().contains(CaptureRequest.SENSOR_PIXEL_MODE));
            diagnosticField(out, "maximum_resolution_stream_map", () -> streamMapDiagnostics(c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP_MAXIMUM_RESOLUTION), budget));
            diagnosticField(out, "maximum_resolution_active_array", () -> rect(c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE_MAXIMUM_RESOLUTION)));
        } else put(out, "maximum_resolution_stream_map", object("available", false, "requires_api", 31));
        diagnosticField(out, "default_active_array", () -> rect(c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)));
        if (Build.VERSION.SDK_INT >= 33) diagnosticField(out, "available_stream_use_cases", () -> boundedLongs(c.get(CameraCharacteristics.SCALER_AVAILABLE_STREAM_USE_CASES)));
        if (Build.VERSION.SDK_INT >= 35) {
            diagnosticField(out, "session_configuration_query_version", () -> c.get(CameraCharacteristics.INFO_SESSION_CONFIGURATION_QUERY_VERSION));
            diagnosticField(out, "session_characteristic_keys", () -> {
                List<String> names = new ArrayList<>();
                for (CameraCharacteristics.Key<?> key : c.getAvailableSessionCharacteristicsKeys()) names.add(key.getName());
                return object("values", boundedStrings(names), "count", names.size(), "truncated", names.size() > PROBE_MAX_LIST_VALUES);
            });
        }
        // This profile is evidence of an advertised camcorder path, not proof that
        // its size belongs to the standard Camera2 map or our configured session.
        if (Build.VERSION.SDK_INT >= 31) diagnosticField(out, "camcorder_8kuhd", () -> {
            if (!topLevel || !id.matches("[0-9]+")
                    || !contains(c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES), CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_BACKWARD_COMPATIBLE)
                    || Integer.valueOf(CameraCharacteristics.LENS_FACING_EXTERNAL).equals(c.get(CameraCharacteristics.LENS_FACING))) {
                return object("queried", false, "reason", "Camcorder profiles apply to top-level numeric backward-compatible non-external camera IDs");
            }
            return camcorder8k(id);
        });
        diagnosticField(out, "vendor_video_traits", () -> vendorVideoTraits(c, budget));
        if(topLevel)diagnosticField(out,"focus_control",()->FocusSettings.metadata(FocusSettings.read(context,id)));
        return out;
    }

    private static JSONObject vendorVideoTraits(CameraCharacteristics c, ProbeBudget budget) {
        List<CameraCharacteristics.Key<?>> keys = c.getKeys();
        JSONArray rows = new JSONArray();
        int scanned = 0;
        for (; scanned < Math.min(keys.size(), 512) && rows.length() < 32; scanned++) {
            CameraCharacteristics.Key<?> key = keys.get(scanned);
            String name = key.getName();
            String lower = name.toLowerCase(Locale.ROOT);
            if (lower.startsWith("android.") || lower.matches(".*(serial|unique|uuid|imei|address|deviceid|sensorid).*")
                    || !lower.matches(".*(video|stream|fps|size|mode|resolution).*")) continue;
            JSONObject row = object("key", name);
            diagnosticField(row, "value", () -> vendorNumericValue(c.get(key), budget));
            rows.put(row);
        }
        return object("numeric_values_only", true, "rows", rows, "characteristic_key_count", keys.size(),
                "keys_scanned", scanned, "truncated", scanned < keys.size(),
                "numeric_budget_exhausted", budget.vendorValuesRemaining == 0);
    }

    private static Object vendorNumericValue(Object value, ProbeBudget budget) {
        if (value == null) return JSONObject.NULL;
        if (value instanceof Integer || value instanceof Long || value instanceof Float || value instanceof Double || value instanceof Boolean) {
            if (budget.vendorValuesRemaining == 0) return object("truncated", true);
            budget.vendorValuesRemaining--;
            return finiteNumber(value);
        }
        if (value instanceof int[] || value instanceof long[] || value instanceof float[] || value instanceof double[]) {
            int length = java.lang.reflect.Array.getLength(value);
            int count = Math.min(length, Math.min(PROBE_MAX_LIST_VALUES, budget.vendorValuesRemaining));
            JSONArray rows = new JSONArray();
            for (int i = 0; i < count; i++) rows.put(finiteNumber(java.lang.reflect.Array.get(value, i)));
            budget.vendorValuesRemaining -= count;
            return object("values", rows, "count", length, "truncated", count < length);
        }
        return object("omitted", true, "reason", "Only numeric and boolean camera mode values are included");
    }

    private static Object finiteNumber(Object value) {
        if (value instanceof Float && !Float.isFinite((Float) value)) return JSONObject.NULL;
        if (value instanceof Double && !Double.isFinite((Double) value)) return JSONObject.NULL;
        return value;
    }

    private JSONObject physicalDiagnostics(CameraCharacteristics parent, String[] topLevelIds, ProbeBudget budget) {
        JSONObject out = object("diagnostic_only", true, "recursive_traversal", false);
        try {
            List<String> ids = new ArrayList<>(parent.getPhysicalCameraIds());
            Collections.sort(ids);
            JSONArray rows = new JSONArray();
            int count = Math.min(ids.size(), Math.min(PROBE_MAX_PHYSICAL_PER_CAMERA, budget.physicalRowsRemaining));
            for (int i = 0; i < count; i++) {
                String id = ids.get(i);
                budget.physicalRowsRemaining--;
                JSONObject row = object("id", id, "also_top_level", Arrays.asList(topLevelIds).contains(id));
                try { put(row, "traits", cameraDiagnostics(id, cameras.getCameraCharacteristics(id), budget, Arrays.asList(topLevelIds).contains(id))); }
                catch (Exception error) { put(row, "error", probeError(error)); }
                rows.put(row);
            }
            put(out, "ids", boundedStrings(ids));
            put(out, "count", ids.size());
            put(out, "rows", rows);
            put(out, "truncated", rows.length() < ids.size());
        } catch (Exception error) { put(out, "error", probeError(error)); }
        return out;
    }

    private static JSONObject streamMapDiagnostics(StreamConfigurationMap map, ProbeBudget budget) {
        if (map == null) return object("available", false);
        JSONObject out = object("available", true);
        diagnosticField(out, "media_codec", () -> sizeListDiagnostics(map, map.getOutputSizes(MediaCodec.class), true, budget));
        diagnosticField(out, "private", () -> sizeListDiagnostics(map, map.getOutputSizes(ImageFormat.PRIVATE), false, budget));
        diagnosticField(out, "high_resolution_private", () -> sizeListDiagnostics(map, map.getHighResolutionOutputSizes(ImageFormat.PRIVATE), false, budget));
        diagnosticField(out, "surface_texture", () -> sizeListDiagnostics(map, map.getOutputSizes(SurfaceTexture.class), false, budget));
        return out;
    }

    private static JSONObject sizeListDiagnostics(StreamConfigurationMap map, Size[] sizes, boolean codec, ProbeBudget budget) {
        if (sizes == null) return object("available", false, "count", 0, "sizes", new JSONArray(), "truncated", false);
        List<Size> ordered = new ArrayList<>(Arrays.asList(sizes));
        // Keep the requested mode and largest modes visible if a vendor returns a
        // large list. The counts and exact-presence result cover the entire list.
        Size wanted = new Size(WIDTH, HEIGHT);
        ordered.sort((left, right) -> {
            if (left.equals(wanted)) return right.equals(wanted) ? 0 : -1;
            if (right.equals(wanted)) return 1;
            return Long.compare((long) right.getWidth() * right.getHeight(), (long) left.getWidth() * left.getHeight());
        });
        JSONArray rows = new JSONArray();
        int count = Math.min(ordered.size(), Math.min(PROBE_MAX_LIST_VALUES, budget.sizeRowsRemaining));
        for (int i = 0; i < count; i++) {
            Size size = ordered.get(i);
            JSONObject row = sizeJson(size);
            diagnosticField(row, "min_frame_duration_ns", () -> codec
                    ? map.getOutputMinFrameDuration(MediaCodec.class, size)
                    : map.getOutputMinFrameDuration(ImageFormat.PRIVATE, size));
            rows.put(row);
            budget.sizeRowsRemaining--;
        }
        JSONObject out = object("available", true, "count", sizes.length, "sizes", rows,
                "requested_8k_present", contains(sizes, wanted), "largest_size", sizeJson(largestSize(sizes)),
                "truncated", count < sizes.length);
        if (contains(sizes, wanted)) diagnosticField(out, "requested_8k_min_frame_duration_ns", () -> codec
                ? map.getOutputMinFrameDuration(MediaCodec.class, wanted)
                : map.getOutputMinFrameDuration(ImageFormat.PRIVATE, wanted));
        return out;
    }

    private static JSONObject camcorder8k(String id) {
        EncoderProfiles profiles = CamcorderProfile.getAll(id, CamcorderProfile.QUALITY_8KUHD);
        if (profiles == null) return object("available", false, "diagnostic_only", true);
        List<EncoderProfiles.VideoProfile> video = profiles.getVideoProfiles();
        JSONArray rows = new JSONArray();
        for (int i = 0; i < Math.min(video.size(), 8); i++) {
            EncoderProfiles.VideoProfile profile = video.get(i);
            rows.put(object("width", profile.getWidth(), "height", profile.getHeight(),
                    "frame_rate", profile.getFrameRate(), "bitrate", profile.getBitrate(),
                    "codec", profile.getCodec(), "media_type", profile.getMediaType()));
        }
        return object("available", true, "diagnostic_only", true, "video_profiles", rows,
                "video_profile_count", video.size(), "truncated", video.size() > rows.length());
    }

    private interface DiagnosticValue { Object read() throws Exception; }
    private static void diagnosticField(JSONObject out, String key, DiagnosticValue value) {
        try { put(out, key, value.read()); }
        catch (Exception error) { put(out, key, object("error", probeError(error))); }
    }
    private static String probeError(Exception error) {
        String message = describe(error);
        return message.length() <= 256 ? message : message.substring(0, 256) + " [truncated]";
    }
    private static int jsonBytes(JSONObject json) {
        // BundleTools persists two-space JSON; budget that exact representation.
        try { return json.toString(2).getBytes(StandardCharsets.UTF_8).length; }
        catch (JSONException error) { return Integer.MAX_VALUE; }
    }
    private static JSONObject sizeJson(Size size) {
        return size == null ? null : object("width", size.getWidth(), "height", size.getHeight());
    }
    private static Size largestSize(Size[] sizes) {
        Size largest = null;
        if (sizes != null) for (Size size : sizes) {
            if (largest == null || (long) size.getWidth() * size.getHeight() > (long) largest.getWidth() * largest.getHeight()) largest = size;
        }
        return largest;
    }
    private static String facingLabel(Integer facing) {
        if (facing == null) return "Unknown-facing";
        if (facing == CameraCharacteristics.LENS_FACING_BACK) return "Rear";
        if (facing == CameraCharacteristics.LENS_FACING_FRONT) return "Front";
        if (facing == CameraCharacteristics.LENS_FACING_EXTERNAL) return "External";
        return "Unknown-facing";
    }
    private static JSONArray boundedStrings(List<String> values) {
        JSONArray out = new JSONArray();
        for (int i = 0; i < Math.min(values.size(), PROBE_MAX_LIST_VALUES); i++) out.put(values.get(i));
        return out;
    }
    private static JSONObject boundedInts(int[] values) {
        if (values == null) return null;
        JSONArray out = new JSONArray();
        for (int i = 0; i < Math.min(values.length, PROBE_MAX_LIST_VALUES); i++) out.put(values[i]);
        return object("values", out, "count", values.length, "truncated", values.length > out.length());
    }
    private static JSONObject boundedLongs(long[] values) {
        if (values == null) return null;
        JSONArray out = new JSONArray();
        for (int i = 0; i < Math.min(values.length, PROBE_MAX_LIST_VALUES); i++) out.put(values[i]);
        return object("values", out, "count", values.length, "truncated", values.length > out.length());
    }
    private static JSONObject fpsRanges(Range<Integer>[] values) {
        if (values == null) return null;
        JSONArray out = new JSONArray();
        for (int i = 0; i < Math.min(values.length, PROBE_MAX_LIST_VALUES); i++) out.put(object("min", values[i].getLower(), "max", values[i].getUpper()));
        return object("values", out, "count", values.length, "truncated", values.length > out.length());
    }
    private static final class ProbeBudget {
        int sizeRowsRemaining = PROBE_MAX_SIZE_ROWS;
        int physicalRowsRemaining = PROBE_MAX_PHYSICAL_TOTAL;
        int vendorValuesRemaining = 2048;
    }

    private EncoderChoice chooseEncoder() {
        for (String mime : new String[]{MediaFormat.MIMETYPE_VIDEO_HEVC, MediaFormat.MIMETYPE_VIDEO_AVC}) {
            for (MediaCodecInfo info : new MediaCodecList(MediaCodecList.REGULAR_CODECS).getCodecInfos()) {
                if (!info.isEncoder() || !info.isHardwareAccelerated() || info.isSoftwareOnly()) continue;
                if (!Arrays.asList(info.getSupportedTypes()).contains(mime)) continue;
                try {
                    MediaCodecInfo.CodecCapabilities caps = info.getCapabilitiesForType(mime);
                    boolean surface = false;
                    for (int color : caps.colorFormats) if (color == MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface) surface = true;
                    if (!surface || caps.getVideoCapabilities() == null
                            || !caps.getVideoCapabilities().areSizeAndRateSupported(WIDTH, HEIGHT, FPS)) continue;
                    int bitrate = caps.getVideoCapabilities().getBitrateRange().clamp(REQUESTED_BITRATE);
                    return new EncoderChoice(info.getName(), mime, bitrate);
                } catch (RuntimeException ignored) {
                    // A codec with inconsistent advertised capabilities is not selected.
                }
            }
        }
        return null;
    }

    private void prepare(Session session, String cameraId) throws Exception {
        if (Build.VERSION.SDK_INT < 33) throw new IOException("Recording requires Android 13/API 33 or newer");
        session.candidate = candidate(cameraId, session.useCalibrationFocus);
        session.focus = session.useCalibrationFocus ? FocusSettings.read(context,cameraId) : null;
        if (session.useCalibrationFocus) FocusSettings.validate(session.candidate.characteristics,cameraId,session.focus);
        session.outputCharacteristics=session.candidate.characteristics;
        if(session.focus!=null&&session.focus.physicalId!=null){
            CameraCharacteristics physical=cameras.getCameraCharacteristics(session.focus.physicalId);
            session.outputCharacteristics=physical;
            if(!Integer.valueOf(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME).equals(physical.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)))
                throw new IOException("The locked lens does not report a REALTIME acquisition clock");
            if(!Objects.equals(physical.get(CameraCharacteristics.SENSOR_ORIENTATION),session.candidate.characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION)))
                throw new IOException("The locked lens orientation differs from the selected camera");
        }
        if (!session.dir.isDirectory() && !session.dir.mkdirs()) throw new IOException("Cannot create session directory");
        for (String name : new String[]{"camera.mp4", "accel.csv", "gyro.csv", "camera_results.jsonl",
                "encoder_pts.csv", "timestamps.csv", "capture_result.json", "capabilities.json"}) {
            if (new File(session.dir, name).exists()) throw new IOException("Session output already exists: " + name);
        }
        session.outputOwned = true;
        session.capabilityReport = probe();
        put(session.capabilityReport, "advertised_capabilities_only", false);
        session.sessionSupport = object("schema", "noesis.phone_capture.android_recording_session.v1",
                "camera_id", cameraId, "width", WIDTH, "height", HEIGHT, "fps", FPS,
                "short_test", session.shortTest, "max_duration_ms", session.maxDurationMs,
                "static_8k_advertised", session.candidate.supported,
                "query_required", !session.candidate.supported, "query_called", false,
                "supported", JSONObject.NULL, "same_configuration_used_for_capture", false,
                "native_image_detail_verified", false);
        put(session.capabilityReport, "recording_session", session.sessionSupport);
        writeJson(new File(session.dir, "capabilities.json"), session.capabilityReport);
        if (!session.candidate.supported) {
            if (!session.candidate.directTestEligible) throw new IOException(session.candidate.json.optString("reason"));
            if ((!session.shortTest && !CapturePreflight.qualifiesWalk(session.candidate.json, session.encoderUseCase, session.useCalibrationFocus))
                    || (session.encoderUseCase != 0 && session.encoderUseCase != 3))
                throw new IOException("This camera requires a successful short recording test for the same configuration before a full walk");
            put(session.sessionSupport, "recorded_preflight", session.candidate.json.opt("recorded_preflight"));
        }
        if (new StatFs(session.dir.getAbsolutePath()).getAvailableBytes() < MIN_FREE_BYTES) {
            throw new IOException("At least 512 MiB free storage is required");
        }
        session.accelerometer = selectAccelerometer();
        session.gyroscope = selectGyroscope();

        EncoderChoice choice = session.candidate.encoder;
        session.codec = MediaCodec.createByCodecName(choice.name);
        MediaFormat format = MediaFormat.createVideoFormat(choice.mime, WIDTH, HEIGHT);
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface);
        format.setInteger(MediaFormat.KEY_BIT_RATE, choice.bitrate);
        format.setInteger(MediaFormat.KEY_FRAME_RATE, FPS);
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1);
        format.setInteger(MediaFormat.KEY_MAX_B_FRAMES, 0);
        session.codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
        session.encoderSurface = session.codec.createInputSurface();
        // A static-map omission requires a fresh query of the actual surfaces;
        // full walks additionally require a completed matching short test.
        if (!session.candidate.supported) {
            long startedMs = SystemClock.elapsedRealtime();
            try {
                if (!cameras.isCameraDeviceSetupSupported(cameraId))
                    throw new IOException("The camera no longer supports direct session queries");
                CameraDevice.CameraDeviceSetup setup = cameras.getCameraDeviceSetup(cameraId);
                session.configuration = recordingConfiguration(session, setup.createCaptureRequest(CameraDevice.TEMPLATE_RECORD));
                put(session.sessionSupport, "query_called", true);
                boolean accepted = setup.isSessionConfigurationSupported(session.configuration);
                put(session.sessionSupport, "supported", accepted);
                if (!accepted) throw new IOException("The driver rejected the actual 8K recording surfaces");
            } catch (Exception failure) {
                put(session.sessionSupport, "error", describe(failure));
                throw failure;
            } finally {
                put(session.sessionSupport, "elapsed_ms", SystemClock.elapsedRealtime() - startedMs);
                writeJson(new File(session.dir, "capabilities.json"), session.capabilityReport);
            }
        }

        session.acceptRecords = true;
        session.metadataThread = new Thread(() -> writeMetadata(session), "RoomWalkMetadataWriter");
        session.metadataThread.start();
        session.sensorListener = new SensorEventListener() {
            @Override public void onSensorChanged(SensorEvent event) {
                if (!session.acceptRecords) return;
                int type = event.sensor.getType();
                boolean accel = type == session.accelerometer.getType();
                offer(session, new ImuRecord(accel, event.timestamp, event.values.clone(),
                        event.accuracy, SystemClock.elapsedRealtimeNanos()));
            }
            @Override public void onAccuracyChanged(Sensor sensor, int accuracy) {}
        };
        boolean accelRegistered = sensors.registerListener(session.sensorListener, session.accelerometer,
                1_000_000 / SENSOR_RATE_HZ, 0, sensorHandler);
        boolean gyroRegistered = sensors.registerListener(session.sensorListener, session.gyroscope,
                1_000_000 / SENSOR_RATE_HZ, 0, sensorHandler);
        if (!accelRegistered || !gyroRegistered) throw new IOException("Could not register both IMU acquisition streams");

        session.muxer = new MediaMuxer(new File(session.dir, "camera.mp4").getAbsolutePath(),
                MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);
        session.muxer.setOrientationHint(0);
        session.codec.start();
        session.codecStarted = true;
        session.encoderThread = new Thread(() -> drainEncoder(session), "RoomWalkEncoderDrain");
        session.encoderThread.start();
        cameras.openCamera(cameraId, new CameraDevice.StateCallback() {
            @Override public void onOpened(CameraDevice camera) {
                if (current != session || session.stopping) { camera.close(); return; }
                session.camera = camera;
                try { configureCamera(session); }
                catch (Exception failure) { fail(session, "camera_configuration_failed", failure); }
            }
            @Override public void onDisconnected(CameraDevice camera) {
                camera.close();
                fail(session, "camera_disconnected", null);
            }
            @Override public void onError(CameraDevice camera, int code) {
                camera.close();
                fail(session, "camera_error_" + code, null);
            }
        }, control);
        session.watchdog = () -> checkLimits(session);
        control.postDelayed(session.watchdog, 1000);
    }

    private void configureCamera(Session session) throws Exception {
        if (session.configuration == null)
            session.configuration = recordingConfiguration(session, session.camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD));
        put(session.sessionSupport, "same_configuration_used_for_capture", true);
        session.camera.createCaptureSession(session.configuration);
    }

    private SessionConfiguration recordingConfiguration(Session session, CaptureRequest.Builder request) throws Exception {
        if (session.preview == null || !session.preview.isValid()) throw new IOException("The recording preview surface is unavailable");
        long[] advertised = session.candidate.characteristics.get(CameraCharacteristics.SCALER_AVAILABLE_STREAM_USE_CASES);
        boolean videoAdvertised = false, previewAdvertised = false;
        if (advertised != null) for (long value : advertised) {
            if (value == 3) videoAdvertised = true;
            if (value == 1) previewAdvertised = true;
        }
        if (session.encoderUseCase == 3 && !videoAdvertised)
            throw new IOException("The standard VIDEO_RECORD use case is no longer advertised");
        long encoderUseCase = session.candidate.supported ? 0 : session.encoderUseCase;
        long previewUseCase = !session.candidate.supported && previewAdvertised ? 1 : 0;
        List<OutputConfiguration> outputs = new ArrayList<>();
        outputs.add(SessionProbe.output(session.encoderSurface, encoderUseCase,session.focus));
        outputs.add(SessionProbe.output(session.preview, previewUseCase,session.focus));
        session.timestampBaseConfigured = true;
        put(session.sessionSupport, "encoder_use_case", encoderUseCase);
        put(session.sessionSupport, "preview_use_case", previewUseCase);
        put(session.sessionSupport, "preview_width", session.candidate.json.optInt("preview_width"));
        put(session.sessionSupport, "preview_height", session.candidate.json.optInt("preview_height"));
        SessionProbe.applyRequestSettings(request, session.candidate.characteristics, session.sessionSupport);
        FocusSettings.apply(request,session.candidate.characteristics,session.candidate.id,session.focus);
        put(session.sessionSupport,"focus_control",FocusSettings.metadata(session.focus));
        request.addTarget(session.encoderSurface);
        request.addTarget(session.preview);
        session.repeatingRequest = request.build();
        Integer af = session.repeatingRequest.get(CaptureRequest.CONTROL_AF_MODE);
        session.requestedAfMode = af == null ? -1 : af;
        Integer rotate = session.repeatingRequest.get(CaptureRequest.SCALER_ROTATE_AND_CROP);
        session.rotateAndCropNoneRequested = rotate != null && rotate == CaptureRequest.SCALER_ROTATE_AND_CROP_NONE;
        Integer distortion = session.repeatingRequest.get(CaptureRequest.DISTORTION_CORRECTION_MODE);
        session.distortionCorrectionOffRequested = distortion != null && distortion == CaptureRequest.DISTORTION_CORRECTION_MODE_OFF;
        SessionConfiguration configuration = new SessionConfiguration(SessionConfiguration.SESSION_REGULAR,
                outputs, command -> control.post(command), new CameraCaptureSession.StateCallback() {
            @Override public void onConfigured(CameraCaptureSession capture) {
                if (current != session || session.stopping) { capture.close(); return; }
                session.capture = capture;
                try {
                    put(session.sessionSupport, "capture_session_configured", true);
                    session.repeatingSequence = capture.setRepeatingRequest(session.repeatingRequest, captureCallback(session), control);
                    session.recordingStartedNs = SystemClock.elapsedRealtimeNanos();
                    if (session.shortTest) control.postDelayed(
                            () -> requestStop(session, "short_test_duration_limit_10_seconds", false), session.maxDurationMs);
                    state("recording", object("session_dir", session.dir.getAbsolutePath(),
                            "width", WIDTH, "height", HEIGHT, "fps", FPS,
                            "camera_id", session.candidate.id,
                            "sensor_orientation_degrees", session.candidate.json.opt("sensor_orientation_degrees"),
                            "max_duration_ms", session.maxDurationMs));
                } catch (Exception failure) { fail(session, "capture_request_failed", failure); }
            }
            @Override public void onConfigureFailed(CameraCaptureSession capture) {
                capture.close();
                fail(session, "8k_camera_session_rejected", null);
            }
        });
        configuration.setSessionParameters(session.repeatingRequest);
        return configuration;
    }

    private CameraCaptureSession.CaptureCallback captureCallback(Session session) {
        return new CameraCaptureSession.CaptureCallback() {
            @Override public void onCaptureCompleted(CameraCaptureSession capture, CaptureRequest request, TotalCaptureResult result) {
                if (!session.acceptRecords) return;
                final FrameRecord record;
                try{record=new FrameRecord(result,session.candidate.id,SystemClock.elapsedRealtimeNanos(),session.focus);}
                catch(IOException missing){fail(session,"locked_lens_metadata_missing",missing);return;}
                offer(session, record);
                session.captureCallbacks.incrementAndGet();
                session.lastCameraReceivedNs=record.receivedNs;
                if(session.focus!=null) {
                    if(FocusSettings.matches(session.focus,result))session.focusConfirmedFrames++;
                    else{session.focusUnconfirmedFrames++;if(session.focusConfirmedFrames>0||SystemClock.elapsedRealtimeNanos()-session.recordingStartedNs>3_000_000_000L)fail(session,"manual_focus_not_confirmed",null);}
                }
                if (record.sensorTimestampNs == null) fail(session, "missing_camera_sensor_timestamp", null);
                if (record.physicalCameraId != null) {
                    if (session.activePhysicalCameraId == null) session.activePhysicalCameraId = record.physicalCameraId;
                    else if (!session.activePhysicalCameraId.equals(record.physicalCameraId)) {
                        fail(session, "active_physical_camera_changed", null);
                    }
                }
                if (record.eisMode != null && record.eisMode != CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
                    fail(session, "electronic_stabilization_not_off", null);
                if (record.oisMode != null && record.oisMode != CaptureResult.LENS_OPTICAL_STABILIZATION_MODE_OFF)
                    fail(session, "optical_stabilization_not_off", null);
                if (record.rotateAndCrop != null && record.rotateAndCrop != CaptureResult.SCALER_ROTATE_AND_CROP_NONE)
                    fail(session, "camera_output_rotation_or_crop_enabled", null);
            }
            @Override public void onCaptureFailed(CameraCaptureSession capture, CaptureRequest request, CaptureFailure failure) {
                fail(session, "camera_capture_failed_frame_" + failure.getFrameNumber() + "_reason_" + failure.getReason(), null);
            }
            @Override public void onCaptureSequenceCompleted(CameraCaptureSession capture, int sequenceId, long frameNumber) {
                if (session.stopping && sequenceId == session.repeatingSequence) finishCamera(session);
            }
            @Override public void onCaptureSequenceAborted(CameraCaptureSession capture, int sequenceId) {
                if (session.stopping && sequenceId == session.repeatingSequence) {
                    addFailure(session, "capture_sequence_aborted");
                    session.partial = true;
                    finishCamera(session);
                }
            }
        };
    }

    private void checkLimits(Session session) {
        if (current != session || session.stopping) return;
        long now = SystemClock.elapsedRealtimeNanos();
        if (session.recordingStartedNs == 0 && now - session.createdNs > TimeUnit.SECONDS.toNanos(30)) {
            fail(session, "camera_start_timeout", null);
            return;
        }
        if (session.recordingStartedNs > 0) {
            long elapsedMs = TimeUnit.NANOSECONDS.toMillis(now - session.recordingStartedNs);
            if (elapsedMs >= session.maxDurationMs) {
                requestStop(session, session.shortTest ? "short_test_duration_limit_10_seconds" : "duration_limit_10_minutes", false); return;
            }
            if (elapsedMs >= 10000 && session.captureCallbacks.get() == 0) {
                fail(session, "no_camera_frames_timeout", null); return;
            }
            if (elapsedMs >= 10000 && session.encodedFrameCount.get() == 0) {
                fail(session, "no_encoded_frames_timeout", null); return;
            }
            String stalled=mediaStallReason(now,session.lastCameraReceivedNs,session.lastEncodedReceivedNs);
            if(stalled!=null){fail(session,stalled,null);return;}
            if (elapsedMs >= SENSOR_START_TIMEOUT_MS) {
                if (session.accelStats.count < 2 || session.gyroStats.count < 2) {
                    fail(session, "imu_stream_start_timeout", null); return;
                }
                if (now - session.accelStats.lastReceivedNs > SENSOR_STALL_TIMEOUT_NS
                        || now - session.gyroStats.lastReceivedNs > SENSOR_STALL_TIMEOUT_NS) {
                    fail(session, "imu_stream_stalled_for_2_seconds", null); return;
                }
            }
            state("recording", object("session_dir", session.dir.getAbsolutePath(),
                    "elapsed_seconds", elapsedMs / 1000.0,
                    "encoded_frames", session.encodedFrameCount.get(),
                    "accel_samples", session.accelStats.count,
                    "gyro_samples", session.gyroStats.count,
                    "encoded_bytes", session.encodedBytes.get()));
        }
        if (session.encodedBytes.get() >= MAX_VIDEO_BYTES) { requestStop(session, "video_size_limit_6_gib", false); return; }
        try {
            if (new StatFs(session.dir.getAbsolutePath()).getAvailableBytes() < MIN_FREE_BYTES) {
                requestStop(session, "storage_below_512_mib", true); return;
            }
        } catch (RuntimeException failure) { fail(session, "storage_check_failed", failure); return; }
        control.postDelayed(session.watchdog, 1000);
    }

    static String mediaStallReason(long now,long cameraReceived,long encodedReceived){
        if(cameraReceived>0&&now-cameraReceived>TimeUnit.SECONDS.toNanos(5))return "camera_stream_stalled_for_5_seconds";
        if(encodedReceived>0&&now-encodedReceived>TimeUnit.SECONDS.toNanos(5))return "encoder_stream_stalled_for_5_seconds";
        return null;
    }

    private void requestStop(Session session, String reason, boolean partial) {
        if (current != session || session.stopping) return;
        session.stopping = true;
        session.partial |= partial;
        session.stopReason = reason;
        if (session.watchdog != null) control.removeCallbacks(session.watchdog);
        state("stopping", object("reason", reason, "session_dir", session.dir.getAbsolutePath()));
        if (session.capture != null && session.repeatingSequence >= 0) {
            try {
                session.capture.stopRepeating();
                control.postDelayed(() -> {
                    if (!session.cameraFinished) {
                        addFailure(session, "capture_stop_timeout");
                        session.partial = true;
                        try { session.capture.abortCaptures(); } catch (Exception ignored) {}
                        finishCamera(session);
                    }
                }, 2500);
            } catch (Exception failure) {
                addFailure(session, "stop_repeating_failed: " + describe(failure));
                session.partial = true;
                finishCamera(session);
            }
        } else finishCamera(session);
    }

    private void finishCamera(Session session) {
        if (session.cameraFinished) return;
        session.cameraFinished = true;
        if (session.capture != null) { session.capture.close(); session.capture = null; }
        if (session.camera != null) { session.camera.close(); session.camera = null; }
        session.drainDeadlineNs = SystemClock.elapsedRealtimeNanos() + TimeUnit.SECONDS.toNanos(8);
        if (session.codecStarted && session.encoderThread != null && session.encoderThread.isAlive()) {
            try { session.codec.signalEndOfInputStream(); }
            catch (RuntimeException failure) { addFailure(session, "encoder_eos_failed: " + describe(failure)); session.forceDrainStop = true; }
        }
        // Leave IMU acquisition running briefly past the final camera exposure, then put
        // a barrier on its handler before closing the metadata writer.
        sensorHandler.postDelayed(() -> {
            if (session.sensorListener != null) sensors.unregisterListener(session.sensorListener);
            sensorHandler.post(() -> {
                session.acceptRecords = false;
                session.metadataStop = true;
                new Thread(() -> finalizeSession(session), "RoomWalkFinalize").start();
            });
        }, 500);
    }

    private void drainEncoder(Session session) {
        boolean muxerStarted = false;
        int track = -1;
        try (BufferedWriter pts = writer(new File(session.dir, "encoder_pts.csv"))) {
            pts.write("encoded_index,encoded_pts_us,flags,size_bytes\n");
            MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
            while (!session.forceDrainStop) {
                if (session.drainDeadlineNs > 0 && SystemClock.elapsedRealtimeNanos() > session.drainDeadlineNs) {
                    throw new IOException("Encoder did not return end-of-stream within 8 seconds");
                }
                int index = session.codec.dequeueOutputBuffer(info, 10_000);
                if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    if (muxerStarted) throw new IOException("Encoder output format changed after muxer start");
                    MediaFormat actual = session.codec.getOutputFormat();
                    session.actualEncoderFormat = actual.toString();
                    if (actual.getInteger(MediaFormat.KEY_WIDTH) != WIDTH || actual.getInteger(MediaFormat.KEY_HEIGHT) != HEIGHT)
                        throw new IOException("Encoder changed the required 7680x4320 resolution");
                    track = session.muxer.addTrack(actual);
                    session.muxer.start();
                    muxerStarted = true;
                } else if (index >= 0) {
                    try {
                        boolean config = (info.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0;
                        if (info.size > 0 && !config) {
                            if ((info.flags & MediaCodec.BUFFER_FLAG_PARTIAL_FRAME) != 0)
                                throw new IOException("Encoder returned a partial access unit");
                            if (!muxerStarted) throw new IOException("Encoded frame arrived before its format");
                            if (session.encodedFrames.size() >= MAX_FRAMES) throw new IOException("Encoded frame evidence limit reached");
                            ByteBuffer buffer = session.codec.getOutputBuffer(index);
                            if (buffer == null) throw new IOException("Missing encoder output buffer");
                            buffer.position(info.offset);
                            buffer.limit(info.offset + info.size);
                            // Preserve the original acquisition-derived timestamp. MediaMuxer
                            // may normalize its MP4 track start; encoder_pts.csv does not.
                            session.muxer.writeSampleData(track, buffer, info);
                            int frameIndex = session.encodedFrames.size();
                            pts.write(frameIndex + "," + info.presentationTimeUs + "," + info.flags + "," + info.size + "\n");
                            session.encodedFrames.add(new TimestampAssociation.EncodedFrame(frameIndex, info.presentationTimeUs));
                            session.encodedFrameCount.incrementAndGet();
                            session.lastEncodedReceivedNs=SystemClock.elapsedRealtimeNanos();
                            long bytes = session.encodedBytes.addAndGet(info.size);
                            if (bytes >= MAX_VIDEO_BYTES) control.post(() -> requestStop(session, "video_size_limit_6_gib", false));
                        }
                        if ((info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                            session.encoderEosReceived = true;
                            break;
                        }
                    } finally { session.codec.releaseOutputBuffer(index, false); }
                }
            }
        } catch (Exception failure) {
            fail(session, "encoder_failed", failure);
        } finally {
            if (muxerStarted) {
                try { session.muxer.stop(); session.muxerFinalized = true; }
                catch (RuntimeException failure) { addFailure(session, "mp4_finalize_failed: " + describe(failure)); }
            }
            releaseEncoder(session);
        }
    }

    private void writeMetadata(Session session) {
        try (BufferedWriter accel = writer(new File(session.dir, "accel.csv"));
             BufferedWriter gyro = writer(new File(session.dir, "gyro.csv"));
             BufferedWriter frames = writer(new File(session.dir, "camera_results.jsonl"))) {
            accel.write(SENSOR_HEADER);
            gyro.write(SENSOR_HEADER);
            while (!session.metadataStop || !session.records.isEmpty()) {
                Record record = session.records.poll(100, TimeUnit.MILLISECONDS);
                if (record instanceof ImuRecord) {
                    ImuRecord imu = (ImuRecord) record;
                    SensorStats stats = imu.accel ? session.accelStats : session.gyroStats;
                    if (stats.count >= MAX_SENSOR_SAMPLES) throw new IOException("IMU sample evidence limit reached");
                    stats.add(imu);
                    BufferedWriter out = imu.accel ? accel : gyro;
                    out.write(imu.csv());
                    if (stats.nonmonotonicCount > 0) throw new IOException("IMU acquisition timestamps are not strictly monotonic");
                } else if (record instanceof FrameRecord) {
                    FrameRecord frame = (FrameRecord) record;
                    if (session.sensorFrames.size() >= MAX_FRAMES) throw new IOException("Camera frame evidence limit reached");
                    frames.write(frame.json().toString());
                    frames.newLine();
                    session.cameraResultCount++;
                    if (frame.sensorTimestampNs != null) session.sensorFrames.add(
                            new TimestampAssociation.SensorFrame(frame.frameNumber, frame.sensorTimestampNs));
                }
            }
            session.metadataFinalized = true;
        } catch (Exception failure) {
            fail(session, "metadata_writer_failed", failure);
        }
    }

    private void offer(Session session, Record record) {
        if (!session.acceptRecords) return;
        if (!session.records.offer(record)) {
            session.droppedRecords.incrementAndGet();
            if (session.overflowReported.compareAndSet(false, true)) fail(session, "metadata_queue_overflow", null);
        }
    }

    private void finalizeSession(Session session) {
        try {
            if (session.encoderThread != null) {
                session.encoderThread.join(10000);
                if (session.encoderThread.isAlive()) {
                    session.forceDrainStop = true;
                    session.encoderThread.join(1000);
                }
                if (session.encoderThread.isAlive()) addFailure(session, "encoder_thread_did_not_stop");
            } else releaseEncoder(session);
            if (session.metadataThread != null) {
                session.metadataThread.join(5000);
                if (session.metadataThread.isAlive()) addFailure(session, "metadata_thread_did_not_stop");
            }
            boolean writersStopped = (session.encoderThread == null || !session.encoderThread.isAlive())
                    && (session.metadataThread == null || !session.metadataThread.isAlive());
            TimestampAssociation.Result association = writersStopped
                    ? TimestampAssociation.associate(session.sensorFrames, session.encodedFrames)
                    : new TimestampAssociation.Result();
            boolean exact = association.verified && session.metadataFinalized
                    && session.droppedRecords.get() == 0 && session.timestampBaseConfigured
                    && session.cameraResultCount == session.sensorFrames.size();
            if (exact) {
                try (BufferedWriter stamps = writer(new File(session.dir, "timestamps.csv"))) {
                    stamps.write("encoded_index,encoded_pts_us,frame_number,timestamp_ns\n");
                    for (TimestampAssociation.Match match : association.matches) {
                        stamps.write(match.encoded.index + "," + match.encoded.ptsUs + ","
                                + match.sensor.frameNumber + "," + match.sensor.timestampNs + "\n");
                    }
                }
            }
            JSONObject result = captureResult(session, association, exact, writersStopped);
            try {
                if (session.outputOwned) {
                    writeJson(new File(session.dir, "capture_result.json"), result);
                    if (session.capabilityReport != null) {
                        put(session.capabilityReport, "capture_attempt", result);
                        if (jsonBytes(session.capabilityReport) > 512 * 1024) {
                            session.capabilityReport.remove("capture_attempt");
                            put(session.capabilityReport, "capture_attempt_omitted_for_size", true);
                            put(session.capabilityReport, "capture_attempt_local_file", "capture_result.json");
                        }
                        writeJson(new File(session.dir, "capabilities.json"), session.capabilityReport);
                    }
                }
            } catch (IOException failure) {
                addFailure(session, "result_write_failed: " + describe(failure));
                put(result, "status", "partial");
                put(result, "partial", true);
                put(result, "export_ready", false);
                put(result, "failures", failureArray(session));
            }
            control.post(() -> {
                if (current == session) current = null;
                main.post(() -> listener.onStopped(result));
                if (closing) shutdownThreads();
            });
        } catch (Exception failure) {
            addFailure(session, "finalization_failed: " + describe(failure));
            JSONObject result = object("schema", "noesis.phone_capture.android_result.v1",
                    "android_api_level", Build.VERSION.SDK_INT, "status", "failed", "partial", true,
                    "export_ready", false, "stop_reason", session.stopReason,
                    "session_dir", session.dir.getAbsolutePath(), "failures", failureArray(session),
                    "metric_vio_allowed", false);
            control.post(() -> {
                if (current == session) current = null;
                main.post(() -> listener.onStopped(result));
                if (closing) shutdownThreads();
            });
        }
    }

    private JSONObject captureResult(Session session, TimestampAssociation.Result association,
                                     boolean exact, boolean writersStopped) {
        long first = association.matches.isEmpty() ? 0 : association.matches.get(0).sensor.timestampNs;
        long last = association.matches.isEmpty() ? 0 : association.matches.get(association.matches.size() - 1).sensor.timestampNs;
        boolean imuCoverage = exact && session.accelStats.covers(first, last) && session.gyroStats.covers(first, last);
        JSONObject files = new JSONObject();
        for (String[] file : new String[][]{{"video", "camera.mp4"}, {"accelerometer", "accel.csv"},
                {"gyroscope", "gyro.csv"}, {"encoder_pts", "encoder_pts.csv"},
                {"camera_results", "camera_results.jsonl"}, {"capabilities", "capabilities.json"}}) {
            if (new File(session.dir, file[1]).isFile()) put(files, file[0], file[1]);
        }
        if (exact) put(files, "timestamps", "timestamps.csv");
        JSONObject camera = object("id", session.candidate == null ? JSONObject.NULL : FocusSettings.outputCameraId(session.candidate.id,session.focus),
                "logical_camera_id",session.candidate==null?JSONObject.NULL:session.candidate.id,
                "width", WIDTH, "height", HEIGHT, "fps", FPS, "fps_is_requested_target", true,
                "timestamp_source", session.candidate == null ? "UNKNOWN" : session.candidate.json.optString("timestamp_source"),
                "timestamp_base", "SENSOR", "timestamp_base_configured", session.timestampBaseConfigured,
                "readout_timestamp_enabled", false, "readout_timestamp_explicitly_disabled", Build.VERSION.SDK_INT >= 34 && session.timestampBaseConfigured,
                "sensor_orientation_degrees", session.candidate == null ? JSONObject.NULL : session.candidate.json.opt("sensor_orientation_degrees"),
                "encoded_rotation_degrees", 0, "active_physical_camera_id", session.activePhysicalCameraId,
                "physical_camera_identity_reported", session.activePhysicalCameraId != null,
                "ois_requested", "OFF", "eis_requested", "OFF", "requested_af_mode", session.requestedAfMode,
                "rotate_and_crop_none_requested", session.rotateAndCropNoneRequested,
                "distortion_correction_off_requested", session.distortionCorrectionOffRequested,
                "intrinsics_source", "raw_android_capture_result_unverified_for_encoded_image",
                "distortion_model", "raw_android_lens_distortion_not_opencv_d5");
        put(camera, "recording_session", session.sessionSupport);
        try{put(camera,"focus_control",FocusSettings.metadata(session.focus).put("confirmed_frame_count",session.focusConfirmedFrames).put("unconfirmed_frame_count",session.focusUnconfirmedFrames));}catch(Exception ignored){}
        if (session.candidate != null) {
            CameraCharacteristics c = session.outputCharacteristics==null?session.candidate.characteristics:session.outputCharacteristics;
            put(camera, "distortion_correction_request_key_available", c.getAvailableCaptureRequestKeys().contains(CaptureRequest.DISTORTION_CORRECTION_MODE));
            put(camera, "distortion_correction_result_key_available", c.getAvailableCaptureResultKeys().contains(CaptureResult.DISTORTION_CORRECTION_MODE));
            int[] modes=c.get(CameraCharacteristics.DISTORTION_CORRECTION_AVAILABLE_MODES);
            JSONArray modeValues=new JSONArray();if(modes!=null)for(int mode:modes)modeValues.put(mode);
            put(camera, "distortion_correction_available_modes", modes==null?null:modeValues);
            put(camera, "sensor_geometry_camera_id", session.candidate.id);
            if(session.activePhysicalCameraId!=null){
                try{c=cameras.getCameraCharacteristics(session.activePhysicalCameraId);put(camera,"sensor_geometry_camera_id",session.activePhysicalCameraId);}
                catch(Exception unavailable){put(camera,"physical_sensor_geometry_error",unavailable.toString());}
            }
            put(camera, "sensor_active_array_size", rect(c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)));
            put(camera, "sensor_pre_correction_active_array_size", rect(c.get(CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE)));
            Size array = c.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE);
            put(camera, "sensor_pixel_array_size", array == null ? null : new JSONArray(Arrays.asList(array.getWidth(), array.getHeight())));
            put(camera, "raw_android_characteristics_lens_intrinsic_calibration", floats(c.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)));
            put(camera, "raw_android_characteristics_lens_distortion", floats(c.get(CameraCharacteristics.LENS_DISTORTION)));
        }
        JSONObject timing = object("sensor_timestamp_source", "android_elapsed_realtime_ns",
                "association_method", "encoder_pts_us_equals_sensor_timestamp_ns_div_1000",
                "exact_frame_association_verified", exact,
                "encoded_frame_count", session.encodedFrames.size(), "camera_result_count", session.cameraResultCount,
                "camera_results_with_sensor_timestamp_count", session.sensorFrames.size(),
                "matched_frame_count", association.matches.size(),
                "unmatched_encoded_frame_count", association.unmatchedEncodedFrames,
                "duplicate_encoded_pts_count", association.duplicateEncodedPts,
                "duplicate_sensor_timestamp_us_count", association.duplicateSensorTimestampUs,
                "duplicate_camera_frame_number_count", association.duplicateCameraFrameNumbers,
                "sensor_timestamps_monotonic", association.sensorTimestampsMonotonic,
                "encoder_pts_monotonic", association.encoderPtsMonotonic,
                "imu_coverage_verified", imuCoverage, "first_matched_sensor_timestamp_ns", first,
                "last_matched_sensor_timestamp_ns", last,
                "camera_results_without_encoded_frame_count", Math.max(0, session.sensorFrames.size() - association.matches.size()),
                "camera_to_imu_time_offset_measured", false,
                "observed_encoded_fps", observedEncodedFps(session),
                "observed_camera_fps", observedCameraFps(session));
        JSONObject accelInfo = sensorInfo(session.accelerometer, "accel.csv", "m/s^2");
        JSONObject gyroInfo = sensorInfo(session.gyroscope, "gyro.csv", "rad/s");
        session.accelStats.appendTo(accelInfo);
        session.gyroStats.appendTo(gyroInfo);
        boolean videoPresent = session.muxerFinalized && !session.encodedFrames.isEmpty();
        boolean imuPresent = session.accelStats.count >= 2 && session.gyroStats.count >= 2
                && session.accelStats.nonmonotonicCount == 0 && session.gyroStats.nonmonotonicCount == 0;
        boolean partial = session.partial || !session.failures.isEmpty() || !videoPresent
                || !session.metadataFinalized || !imuPresent || session.encodedFrames.size() < 2;
        JSONObject encoder = session.candidate == null || session.candidate.encoder == null
                ? new JSONObject() : session.candidate.encoder.json();
        put(encoder, "actual_output_format", session.actualEncoderFormat);
        put(encoder, "encoded_bytes", session.encodedBytes.get());
        put(encoder, "eos_received", session.encoderEosReceived);
        put(encoder, "mp4_finalized", session.muxerFinalized);
        put(encoder, "original_pts_preserved_in_sidecar", true);
        return object("schema", "noesis.phone_capture.android_result.v1", "android_api_level", Build.VERSION.SDK_INT,
                "device", device(), "status", partial ? (videoPresent ? "partial" : "failed") : "complete",
                "partial", partial, "export_ready", writersStopped && videoPresent && session.encodedFrames.size() >= 2
                        && session.metadataFinalized && imuPresent,
                "stop_reason", session.stopReason, "session_dir", session.dir.getAbsolutePath(),
                "files", files, "camera", camera, "encoder", encoder,
                "sensors", object("accelerometer", accelInfo, "gyroscope", gyroInfo), "timing", timing,
                "metric_vio_allowed", false,
                "calibration", object("camera_intrinsics_verified", false,
                        "camera_to_imu_extrinsics_verified", false, "time_offset_verified", false, "imu_noise_verified", false),
                "failures", failureArray(session), "dropped_metadata_records", session.droppedRecords.get(),
                "bounds", object("max_duration_ms", session.maxDurationMs, "short_test", session.shortTest,
                        "max_video_bytes", MAX_VIDEO_BYTES,
                        "minimum_free_bytes", MIN_FREE_BYTES, "metadata_queue_capacity", METADATA_QUEUE_CAPACITY,
                        "maximum_camera_frames", MAX_FRAMES, "maximum_samples_per_imu_sensor", MAX_SENSOR_SAMPLES,
                        "imu_start_timeout_ms", SENSOR_START_TIMEOUT_MS,
                        "imu_stall_timeout_ms", TimeUnit.NANOSECONDS.toMillis(SENSOR_STALL_TIMEOUT_NS)));
    }

    private void releaseEncoder(Session session) {
        if (!session.encoderReleased.compareAndSet(false, true)) return;
        if (session.codec != null) {
            if (session.codecStarted) try { session.codec.stop(); } catch (RuntimeException ignored) {}
            try { session.codec.release(); } catch (RuntimeException ignored) {}
        }
        if (session.muxer != null) try { session.muxer.release(); } catch (RuntimeException ignored) {}
        if (session.encoderSurface != null) session.encoderSurface.release();
    }

    private void fail(Session session, String reason, Exception exception) {
        String detail = exception == null ? reason : reason + ": " + describe(exception);
        addFailure(session, detail);
        session.partial = true;
        control.post(() -> {
            if (current != session) return;
            if (session.errorReported.compareAndSet(false, true)) error(detail, object("session_dir", session.dir.getAbsolutePath()));
            requestStop(session, reason, true);
        });
    }

    private static void addFailure(Session session, String message) {
        synchronized (session.failures) {
            if (session.failures.size() < 32 && !session.failures.contains(message)) session.failures.add(message);
        }
    }

    private static JSONArray failureArray(Session session) {
        synchronized (session.failures) { return new JSONArray(new ArrayList<>(session.failures)); }
    }

    private static double observedEncodedFps(Session session) {
        int count = session.encodedFrames.size();
        if (count < 2) return 0;
        long spanUs = session.encodedFrames.get(count - 1).ptsUs - session.encodedFrames.get(0).ptsUs;
        return spanUs > 0 ? (count - 1) * 1e6 / spanUs : 0;
    }

    private static double observedCameraFps(Session session) {
        int count = session.sensorFrames.size();
        if (count < 2) return 0;
        long spanNs = session.sensorFrames.get(count - 1).timestampNs - session.sensorFrames.get(0).timestampNs;
        return spanNs > 0 ? (count - 1) * 1e9 / spanNs : 0;
    }

    private void state(String state, JSONObject details) { main.post(() -> listener.onState(state, details)); }
    private void error(String message, JSONObject details) { main.post(() -> listener.onError(message, details)); }
    private void shutdownThreads() { sensorThread.quitSafely(); controlThread.quitSafely(); }
    private Sensor selectAccelerometer() {
        return NativeSensors.accelerometer(sensors);
    }
    private Sensor selectGyroscope() {
        return NativeSensors.gyroscope(sensors);
    }
    private JSONObject sensorInfo(Sensor sensor, String file, String units) {
        if (sensor == null) return object("available", false, "file", file);
        boolean uncalibrated = sensor.getType() == Sensor.TYPE_ACCELEROMETER_UNCALIBRATED
                || sensor.getType() == Sensor.TYPE_GYROSCOPE_UNCALIBRATED;
        return object("available", true, "file", file, "sensor_id",NativeSensors.sensorId(context,sensor),"android_sensor_id",sensor.getId(),"type", sensor.getType(),
                "type_name", sensor.getStringType(), "name", sensor.getName(), "vendor", sensor.getVendor(),
                "version", sensor.getVersion(), "reporting_mode", sensor.getReportingMode(),
                "requested_rate_hz", SENSOR_RATE_HZ, "minimum_delay_us", sensor.getMinDelay(),
                "maximum_range", sensor.getMaximumRange(), "resolution", sensor.getResolution(),
                "units", units, "axes", AXES, "uncalibrated", uncalibrated,
                "bias_fields_are_sensor_estimates", uncalibrated,
                "timestamp_source", "android_elapsed_realtime_ns",
                "accelerometer_includes_gravity", units.equals("m/s^2"));
    }
    private static JSONObject device() {
        return object("manufacturer", Build.MANUFACTURER, "model", Build.MODEL,
                "device", Build.DEVICE, "build_fingerprint", Build.FINGERPRINT,
                "android_version", Build.VERSION.RELEASE, "android_api_level", Build.VERSION.SDK_INT);
    }
    private static String timestampSource(Integer source) {
        return source != null && source == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME ? "REALTIME" : "UNKNOWN";
    }
    private static Size choosePreview(StreamConfigurationMap map) {
        Size best = null;
        int area = 0;
        Size[] sizes = map == null ? null : map.getOutputSizes(SurfaceTexture.class);
        if (sizes != null) for (Size size : sizes) {
            if (size.getWidth() <= 1920 && size.getHeight() <= 1080
                    && size.getWidth() * 9 == size.getHeight() * 16
                    && size.getWidth() * size.getHeight() > area) {
                best = size; area = size.getWidth() * size.getHeight();
            }
        }
        if (best == null && sizes != null) for (Size size : sizes) {
            if (size.getWidth() <= 1920 && size.getHeight() <= 1080 && size.getWidth() * size.getHeight() > area) {
                best = size; area = size.getWidth() * size.getHeight();
            }
        }
        return best;
    }
    private static int chooseAf(int[] modes) {
        for (int mode : new int[]{CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO,
                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE, CaptureRequest.CONTROL_AF_MODE_AUTO,
                CaptureRequest.CONTROL_AF_MODE_OFF}) if (contains(modes, mode)) return mode;
        return CaptureRequest.CONTROL_AF_MODE_OFF;
    }
    private static boolean contains(int[] values, int wanted) {
        if (values != null) for (int value : values) if (value == wanted) return true;
        return false;
    }
    private static boolean contains(Size[] values, Size wanted) {
        return values != null && Arrays.asList(values).contains(wanted);
    }
    private static String describe(Exception error) {
        return error.getClass().getSimpleName() + (error.getMessage() == null ? "" : ": " + error.getMessage());
    }
    private static BufferedWriter writer(File file) throws IOException {
        return new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), StandardCharsets.UTF_8), 65536);
    }
    private static void writeJson(File file, JSONObject value) throws IOException {
        try (BufferedWriter out = writer(file)) { out.write(value.toString()); out.newLine(); }
    }
    private static JSONObject object(Object... keyValues) {
        JSONObject result = new JSONObject();
        for (int i = 0; i < keyValues.length; i += 2) put(result, (String) keyValues[i], keyValues[i + 1]);
        return result;
    }
    private static void put(JSONObject object, String key, Object value) {
        try { object.put(key, value == null ? JSONObject.NULL : value); }
        catch (JSONException error) { throw new IllegalArgumentException("Invalid JSON field " + key, error); }
    }
    private static JSONArray floats(float[] values) {
        if (values == null) return null;
        JSONArray result = new JSONArray();
        for (float value : values) result.put(Float.isFinite(value) ? Double.valueOf(value) : JSONObject.NULL);
        return result;
    }
    private static JSONArray rect(Rect value) {
        return value == null ? null : new JSONArray(Arrays.asList(value.left, value.top, value.right, value.bottom));
    }

    private interface Record {}
    private static final class ImuRecord implements Record {
        final boolean accel;
        final long timestampNs;
        final float[] values;
        final int accuracy;
        final long receivedNs;
        ImuRecord(boolean accel, long timestampNs, float[] values, int accuracy, long receivedNs) {
            this.accel = accel; this.timestampNs = timestampNs; this.values = values;
            this.accuracy = accuracy; this.receivedNs = receivedNs;
        }
        String csv() throws IOException {
            if (values.length < 3) throw new IOException("IMU event has fewer than three axes");
            StringBuilder row = new StringBuilder().append(timestampNs);
            for (int i = 0; i < 6; i++) {
                row.append(',');
                if (i < values.length) {
                    if (!Float.isFinite(values[i])) throw new IOException("IMU event contains non-finite values");
                    row.append(values[i]);
                }
            }
            return row.append(',').append(accuracy).append(',').append(receivedNs).append('\n').toString();
        }
    }
    private static final class FrameRecord implements Record {
        final long frameNumber;
        final Long sensorTimestampNs, exposureTimeNs, frameDurationNs, rollingShutterSkewNs;
        final Integer sensitivity, oisMode, eisMode, afState, aeState, rotateAndCrop, distortionCorrection, sensorPixelMode;
        final Float focusDistance, focalLength, zoomRatio;
        final float[] intrinsics, distortion;
        final Rect crop;
        final String physicalCameraId, resultCameraId,logicalPhysicalCameraId,outputPhysicalCameraId;
        final Long logicalTimestampNs;
        final PhysicalFrameRecord physicalResult,logicalResult;
        final long receivedNs;
        FrameRecord(TotalCaptureResult total, String logicalCameraId, long receivedNs,FocusSettings.Lock focus) throws IOException {
            CaptureResult result=FocusSettings.resultForOutput(total,focus);
            this.receivedNs = receivedNs;
            outputPhysicalCameraId=focus==null?null:focus.physicalId;
            this.resultCameraId=outputPhysicalCameraId==null?logicalCameraId:outputPhysicalCameraId;
            logicalTimestampNs=total.get(CaptureResult.SENSOR_TIMESTAMP);
            logicalPhysicalCameraId=total.get(CaptureResult.LOGICAL_MULTI_CAMERA_ACTIVE_PHYSICAL_ID);
            logicalResult=outputPhysicalCameraId==null?null:new PhysicalFrameRecord(logicalCameraId,total);
            frameNumber = total.getFrameNumber();
            sensorTimestampNs = result.get(CaptureResult.SENSOR_TIMESTAMP);
            exposureTimeNs = result.get(CaptureResult.SENSOR_EXPOSURE_TIME);
            frameDurationNs = result.get(CaptureResult.SENSOR_FRAME_DURATION);
            rollingShutterSkewNs = result.get(CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW);
            sensitivity = result.get(CaptureResult.SENSOR_SENSITIVITY);
            oisMode = result.get(CaptureResult.LENS_OPTICAL_STABILIZATION_MODE);
            eisMode = result.get(CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE);
            afState = result.get(CaptureResult.CONTROL_AF_STATE);
            aeState = result.get(CaptureResult.CONTROL_AE_STATE);
            focusDistance = result.get(CaptureResult.LENS_FOCUS_DISTANCE);
            focalLength = result.get(CaptureResult.LENS_FOCAL_LENGTH);
            zoomRatio = result.get(CaptureResult.CONTROL_ZOOM_RATIO);
            rotateAndCrop = result.get(CaptureResult.SCALER_ROTATE_AND_CROP);
            distortionCorrection = result.get(CaptureResult.DISTORTION_CORRECTION_MODE);
            sensorPixelMode = result.get(CaptureResult.SENSOR_PIXEL_MODE);
            float[] rawIntrinsics = result.get(CaptureResult.LENS_INTRINSIC_CALIBRATION);
            float[] rawDistortion = result.get(CaptureResult.LENS_DISTORTION);
            intrinsics = rawIntrinsics == null ? null : rawIntrinsics.clone();
            distortion = rawDistortion == null ? null : rawDistortion.clone();
            Rect rawCrop = result.get(CaptureResult.SCALER_CROP_REGION);
            crop = rawCrop == null ? null : new Rect(rawCrop);
            physicalCameraId=outputPhysicalCameraId==null?logicalPhysicalCameraId:outputPhysicalCameraId;
            CaptureResult physical = physicalCameraId == null ? null : total.getPhysicalCameraResults().get(physicalCameraId);
            physicalResult = physical == null ? null : new PhysicalFrameRecord(physicalCameraId, physical);
        }
        JSONObject json() {
            return object("frame_number", frameNumber, "sensor_timestamp_ns", sensorTimestampNs,
                    "received_elapsed_realtime_ns", receivedNs, "exposure_time_ns", exposureTimeNs,
                    "frame_duration_ns", frameDurationNs, "rolling_shutter_skew_ns", rollingShutterSkewNs,
                    "sensor_sensitivity_iso", sensitivity, "lens_focus_distance_diopters", focusDistance,
                    "lens_focal_length_mm", focalLength, "crop_region", rect(crop), "zoom_ratio", zoomRatio,
                    "ois_mode", oisMode, "eis_mode", eisMode, "af_state", afState, "ae_state", aeState,
                    "rotate_and_crop_mode", rotateAndCrop, "distortion_correction_mode", distortionCorrection,
                    "sensor_pixel_mode", sensorPixelMode,
                    "result_camera_id", resultCameraId,
                    "metadata_source",outputPhysicalCameraId==null?"logical_capture_result":"physical_output_capture_result",
                    "output_physical_camera_id",outputPhysicalCameraId,
                    "logical_active_physical_camera_id",logicalPhysicalCameraId,
                    "logical_capture_result",logicalResult==null?null:logicalResult.json(),
                    "active_physical_camera_id", physicalCameraId,
                    "physical_capture_result", physicalResult == null ? null : physicalResult.json(),
                    "physical_capture_result_matches_logical_timestamp", physicalResult != null && logicalTimestampNs != null && logicalTimestampNs.equals(physicalResult.sensorTimestampNs),
                    "raw_android_lens_intrinsic_calibration", floats(intrinsics),
                    "raw_android_lens_distortion", floats(distortion),
                    "calibration_admitted", false);
        }
    }
    /** Preserve each SDK result with its own camera identity; never relabel logical metadata. */
    private static final class PhysicalFrameRecord {
        final String cameraId;
        final Long sensorTimestampNs, exposureTimeNs, rollingShutterSkewNs;
        final Integer sensorPixelMode, distortionMode, oisMode, eisMode, rotateAndCrop,afMode,afState;
        final Float zoomRatio,focusDistance,focalLength;
        final Rect crop;
        PhysicalFrameRecord(String cameraId, CaptureResult result) {
            this.cameraId = cameraId;
            sensorTimestampNs = result.get(CaptureResult.SENSOR_TIMESTAMP);
            exposureTimeNs = result.get(CaptureResult.SENSOR_EXPOSURE_TIME);
            rollingShutterSkewNs = result.get(CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW);
            sensorPixelMode = result.get(CaptureResult.SENSOR_PIXEL_MODE);
            distortionMode = result.get(CaptureResult.DISTORTION_CORRECTION_MODE);
            oisMode = result.get(CaptureResult.LENS_OPTICAL_STABILIZATION_MODE);
            eisMode = result.get(CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE);
            rotateAndCrop = result.get(CaptureResult.SCALER_ROTATE_AND_CROP);
            zoomRatio = result.get(CaptureResult.CONTROL_ZOOM_RATIO);
            focusDistance=result.get(CaptureResult.LENS_FOCUS_DISTANCE);
            focalLength=result.get(CaptureResult.LENS_FOCAL_LENGTH);
            afMode=result.get(CaptureResult.CONTROL_AF_MODE);afState=result.get(CaptureResult.CONTROL_AF_STATE);
            Rect rawCrop = result.get(CaptureResult.SCALER_CROP_REGION);
            crop = rawCrop == null ? null : new Rect(rawCrop);
        }
        JSONObject json() {
            return object("result_camera_id", cameraId, "sensor_timestamp_ns", sensorTimestampNs,
                    "exposure_time_ns", exposureTimeNs, "rolling_shutter_skew_ns", rollingShutterSkewNs,
                    "sensor_pixel_mode", sensorPixelMode, "distortion_correction_mode", distortionMode,
                    "ois_mode", oisMode, "eis_mode", eisMode, "rotate_and_crop_mode", rotateAndCrop,
                    "zoom_ratio", zoomRatio, "crop_region", rect(crop),
                    "lens_focus_distance_diopters",focusDistance,"lens_focal_length_mm",focalLength,"af_mode",afMode,"af_state",afState);
        }
    }
    private static final class SensorStats {
        volatile long count, lastReceivedNs;
        long firstNs, lastNs, nonmonotonicCount, unreliableCount;
        long minDeltaNs = Long.MAX_VALUE, maxDeltaNs;
        void add(ImuRecord record) {
            if (count == 0) firstNs = record.timestampNs;
            else {
                long delta = record.timestampNs - lastNs;
                if (delta <= 0) nonmonotonicCount++;
                else { minDeltaNs = Math.min(minDeltaNs, delta); maxDeltaNs = Math.max(maxDeltaNs, delta); }
            }
            if (record.timestampNs <= 0) nonmonotonicCount++;
            if (record.accuracy == SensorManager.SENSOR_STATUS_UNRELIABLE) unreliableCount++;
            lastNs = record.timestampNs;
            lastReceivedNs = record.receivedNs;
            count++;
        }
        boolean covers(long first, long last) {
            return count >= 2 && nonmonotonicCount == 0 && first > 0 && firstNs <= first && lastNs >= last;
        }
        void appendTo(JSONObject result) {
            put(result, "sample_count", count); put(result, "first_timestamp_ns", firstNs);
            put(result, "last_timestamp_ns", lastNs); put(result, "nonmonotonic_timestamp_count", nonmonotonicCount);
            put(result, "unreliable_accuracy_sample_count", unreliableCount);
            put(result, "observed_rate_hz", count > 1 && lastNs > firstNs ? (count - 1) * 1e9 / (lastNs - firstNs) : 0.0);
            put(result, "minimum_interval_ns", minDeltaNs == Long.MAX_VALUE ? 0 : minDeltaNs);
            put(result, "maximum_interval_ns", maxDeltaNs);
        }
    }
    private static final class EncoderChoice {
        final String name, mime;
        final int bitrate;
        EncoderChoice(String name, String mime, int bitrate) { this.name = name; this.mime = mime; this.bitrate = bitrate; }
        JSONObject json() { return object("name", name, "mime", mime, "hardware_accelerated", true,
                "width", WIDTH, "height", HEIGHT, "fps", FPS, "requested_bitrate", bitrate,
                "preferred_bitrate", REQUESTED_BITRATE, "max_b_frames", 0); }
    }
    private static final class Candidate {
        final String id;
        final CameraCharacteristics characteristics;
        final EncoderChoice encoder;
        final JSONObject json;
        final boolean supported, directTestEligible;
        Candidate(String id, CameraCharacteristics characteristics, EncoderChoice encoder, JSONObject json, boolean supported, boolean directTestEligible) {
            this.id = id; this.characteristics = characteristics; this.encoder = encoder; this.json = json;
            this.supported = supported; this.directTestEligible = directTestEligible;
        }
    }
    private static final class Session {
        CameraCharacteristics outputCharacteristics;
        final File dir;
        final Surface preview;
        final boolean shortTest;
        final long maxDurationMs, encoderUseCase;
        final long createdNs = SystemClock.elapsedRealtimeNanos();
        final ArrayBlockingQueue<Record> records = new ArrayBlockingQueue<>(METADATA_QUEUE_CAPACITY);
        final List<TimestampAssociation.SensorFrame> sensorFrames = new ArrayList<>();
        final List<TimestampAssociation.EncodedFrame> encodedFrames = new ArrayList<>();
        final List<String> failures = Collections.synchronizedList(new ArrayList<>());
        final SensorStats accelStats = new SensorStats(), gyroStats = new SensorStats();
        final AtomicLong droppedRecords = new AtomicLong(), captureCallbacks = new AtomicLong(), encodedBytes = new AtomicLong(), encodedFrameCount = new AtomicLong();
        final AtomicBoolean overflowReported = new AtomicBoolean(), errorReported = new AtomicBoolean(), encoderReleased = new AtomicBoolean();
        volatile boolean acceptRecords, metadataStop, stopping, partial, forceDrainStop;
        volatile boolean metadataFinalized, muxerFinalized, encoderEosReceived;
        volatile long drainDeadlineNs;
        volatile long lastCameraReceivedNs,lastEncodedReceivedNs;
        boolean outputOwned, cameraFinished, codecStarted, timestampBaseConfigured, rotateAndCropNoneRequested, distortionCorrectionOffRequested;
        int repeatingSequence = -1, requestedAfMode = -1, cameraResultCount;
        long recordingStartedNs;
        String activePhysicalCameraId, stopReason = "unknown";
        volatile String actualEncoderFormat;
        FocusSettings.Lock focus;long focusConfirmedFrames,focusUnconfirmedFrames;
        Candidate candidate;
        Sensor accelerometer, gyroscope;
        SensorEventListener sensorListener;
        CameraDevice camera;
        CameraCaptureSession capture;
        SessionConfiguration configuration;
        CaptureRequest repeatingRequest;
        JSONObject capabilityReport, sessionSupport;
        MediaCodec codec;
        MediaMuxer muxer;
        Surface encoderSurface;
        Thread metadataThread, encoderThread;
        Runnable watchdog;
        final boolean useCalibrationFocus;
        Session(File dir, Surface preview, boolean shortTest, long encoderUseCase, boolean useCalibrationFocus) {
            this.dir = dir; this.preview = preview; this.shortTest = shortTest; this.encoderUseCase = encoderUseCase; this.useCalibrationFocus = useCalibrationFocus;
            this.maxDurationMs = shortTest ? SHORT_TEST_DURATION_MS : MAX_DURATION_MS;
        }
    }
}

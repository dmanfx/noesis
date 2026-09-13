package org.noesis.roomwalk;

import android.content.Context;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.OutputConfiguration;
import android.hardware.camera2.params.SessionConfiguration;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import android.os.Build;
import android.os.SystemClock;
import android.util.Range;
import android.view.Surface;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/** Bounded setup queries only: this helper never opens a camera or starts an encoder. */
public final class SessionProbe {
    private static final int MAX_CAMERAS = 16;
    private static final int MAX_OBSERVED_USE_CASES = 64;
    private static final int MAX_VENDOR_USE_CASES = 12;
    private static final int MAX_QUERIES = 28;
    private static final long ELAPSED_BUDGET_MS = 25_000;
    private static final int MAX_REPORT_BYTES = 32 * 1024;
    private static final int REQUESTED_BITRATE = 128_000_000;
    // The public SDK exposes standard IDs only; the Camera HAL metadata defines this boundary.
    private static final long VENDOR_START = 0x10000L;
    private static final long DEFAULT = CameraCharacteristics.SCALER_AVAILABLE_STREAM_USE_CASES_DEFAULT;
    private static final long PREVIEW = CameraCharacteristics.SCALER_AVAILABLE_STREAM_USE_CASES_PREVIEW;
    private static final long VIDEO_RECORD = CameraCharacteristics.SCALER_AVAILABLE_STREAM_USE_CASES_VIDEO_RECORD;

    private SessionProbe() {}

    /** A standard positive result permits a short test only; it never promotes a walk. */
    public static long standardTestUseCase(JSONObject report, JSONObject camera) {
        if (report == null || camera == null || !camera.optBoolean("direct_session_test_eligible")) return -1;
        if (!camera.optString("id").equals(report.optString("camera_id"))
                || report.optInt("requested_width") != CaptureEngine.WIDTH
                || report.optInt("requested_height") != CaptureEngine.HEIGHT
                || report.optInt("requested_fps") != CaptureEngine.FPS) return -1;
        JSONObject preview = report.optJSONObject("preview_surface");
        if (preview == null || preview.optInt("width") != camera.optInt("preview_width")
                || preview.optInt("height") != camera.optInt("preview_height")) return -1;
        JSONArray rows = report.optJSONArray("combinations");
        if (rows == null) return -1;
        for (long wanted : new long[]{VIDEO_RECORD, DEFAULT}) {
            for (int i = 0; i < Math.min(rows.length(), MAX_QUERIES); i++) {
                JSONObject row = rows.optJSONObject(i);
                if (row != null && row.optBoolean("query_called") && row.optBoolean("supported")
                        && "encoder_and_preview".equals(row.optString("output_mode"))
                        && row.optLong("encoder_use_case", -1) == wanted
                        && (row.optLong("preview_use_case", -1) == PREVIEW
                            || row.optLong("preview_use_case", -1) == DEFAULT)) return wanted;
            }
        }
        return -1;
    }

    /** Invoke on the Check phone worker, independently of the recording readiness decision. */
    public static JSONObject run(Context context, JSONObject capabilityReport) {
        long startedMs = SystemClock.elapsedRealtime();
        JSONArray combinations = new JSONArray();
        JSONObject result = object("schema", "noesis.phone_capture.android_session_queries.v1",
                "status", "skipped", "supported_count", 0, "query_count", 0,
                "combinations", combinations, "reason", "No eligible rear camera was found",
                "android_api_level", Build.VERSION.SDK_INT, "diagnostic_only", true,
                "recording_enabled_by_query", false, "report_truncated", false,
                "requested_width", CaptureEngine.WIDTH, "requested_height", CaptureEngine.HEIGHT,
                "requested_fps", CaptureEngine.FPS,
                "timestamp_base", "SENSOR", "timestamp_base_value", OutputConfiguration.TIMESTAMP_BASE_SENSOR,
                "readout_timestamp_enabled", false, "session_type", "REGULAR",
                "request_template", "RECORD", "request_template_value", CameraDevice.TEMPLATE_RECORD,
                "limits", object("cameras", 1, "vendor_use_cases", MAX_VENDOR_USE_CASES,
                        "queries", MAX_QUERIES, "elapsed_budget_ms", ELAPSED_BUDGET_MS,
                        "budget_checked_between_queries", true, "individual_call_timeout_enforced", false,
                        "maximum_report_bytes", MAX_REPORT_BYTES),
                "interpretation", "A positive query reports HAL acceptance of the submitted session. "
                        + "It does not prove native 8K detail, lack of upscaling, actual frame cadence, "
                        + "acquisition timestamps, stabilization state, or calibration. Vendor IDs have no inferred meaning.");
        MediaCodec codec = null;
        Surface encoderSurface = null;
        Surface previewSurface = null;
        SurfaceTexture previewTexture = null;
        String stage = "eligibility";
        try {
            if (Build.VERSION.SDK_INT < 35) {
                put(result, "reason", "CameraDeviceSetup session queries require Android API 35 or newer");
                return result;
            }
            JSONArray cameras = capabilityReport == null ? null : capabilityReport.optJSONArray("cameras");
            if (cameras == null) return result;
            JSONObject target = null;
            for (int i = 0; i < Math.min(cameras.length(), MAX_CAMERAS); i++) {
                JSONObject row = cameras.optJSONObject(i);
                if (row == null || row.optInt("lens_facing", -1) != CameraCharacteristics.LENS_FACING_BACK) continue;
                if (row.optBoolean("supported8k", false)) {
                    put(result, "reason", "The standard recording path already advertises 8K/30; extra mode queries are unnecessary");
                    return result;
                }
                JSONObject encoder = row.optJSONObject("encoder");
                if (target == null && !row.optBoolean("default_8k_size_available", true)
                        && row.optBoolean("fixed_30_fps_available", false)
                        && "REALTIME".equals(row.optString("timestamp_source"))
                        && encoder != null && encoder.optBoolean("hardware_accelerated", false)
                        && encoder.optInt("width") == CaptureEngine.WIDTH
                        && encoder.optInt("height") == CaptureEngine.HEIGHT
                        && encoder.optInt("fps") == CaptureEngine.FPS) target = row;
            }
            if (target == null) return result;
            String cameraId = requiredString(target, "id", 128);
            put(result, "camera_id", cameraId);
            put(result, "default_8k_size_available", false);
            put(result, "timestamp_source", "REALTIME");
            CameraManager manager = (CameraManager) context.getApplicationContext().getSystemService(Context.CAMERA_SERVICE);
            if (manager == null) throw new IllegalStateException("Camera service is unavailable");
            stage = "camera_device_setup";
            boolean setupSupported = manager.isCameraDeviceSetupSupported(cameraId);
            put(result, "camera_device_setup_supported", setupSupported);
            if (!setupSupported) {
                put(result, "reason", "The selected camera does not support CameraDeviceSetup queries");
                return result;
            }
            CameraDevice.CameraDeviceSetup setup = manager.getCameraDeviceSetup(cameraId);
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            put(result, "session_configuration_query_version",
                    characteristics.get(CameraCharacteristics.INFO_SESSION_CONFIGURATION_QUERY_VERSION));
            int previewWidth = target.optInt("preview_width");
            int previewHeight = target.optInt("preview_height");
            if (previewWidth <= 0 || previewWidth > 1920 || previewHeight <= 0 || previewHeight > 1080) {
                put(result, "reason", "The selected camera has no advertised preview size within 1920x1080");
                return result;
            }
            put(result, "preview_surface", object("width", previewWidth, "height", previewHeight,
                    "type", "detached_SurfaceTexture"));

            // Only IDs present in this probe's actual camera trait list may enter the vendor sweep.
            JSONObject diagnostics = target.optJSONObject("diagnostics");
            JSONObject observed = diagnostics == null ? null : diagnostics.optJSONObject("available_stream_use_cases");
            JSONArray values = observed == null ? null : observed.optJSONArray("values");
            Set<Long> advertised = new LinkedHashSet<>();
            if (values != null) for (int i = 0; i < Math.min(values.length(), MAX_OBSERVED_USE_CASES); i++) {
                Object value = values.opt(i);
                if (value instanceof Integer || value instanceof Long) advertised.add(((Number) value).longValue());
            }
            Set<Long> selected = new LinkedHashSet<>();
            selected.add(DEFAULT);
            if (advertised.contains(VIDEO_RECORD)) selected.add(VIDEO_RECORD);
            int vendorCount = 0;
            for (long value : advertised) if (value >= VENDOR_START) {
                if (vendorCount < MAX_VENDOR_USE_CASES) selected.add(value);
                vendorCount++;
            }
            long previewUseCase = advertised.contains(PREVIEW) ? PREVIEW : DEFAULT;
            put(result, "observed_stream_use_cases", longs(advertised));
            put(result, "selected_encoder_use_cases", longs(selected));
            put(result, "vendor_use_cases_observed", vendorCount);
            put(result, "vendor_use_cases_truncated", vendorCount > MAX_VENDOR_USE_CASES
                    || (observed != null && observed.optBoolean("truncated", false))
                    || (values != null && values.length() > MAX_OBSERVED_USE_CASES));
            put(result, "planned_query_count", selected.size() * 2);

            stage = "session_parameters";
            CaptureRequest parameters = sessionParameters(context,cameraId,setup, characteristics, result);
            stage = "encoder_allocation";
            JSONObject encoder = target.optJSONObject("encoder");
            String codecName = requiredString(encoder, "name", 256);
            String mime = requiredString(encoder, "mime", 128);
            if (!MediaFormat.MIMETYPE_VIDEO_HEVC.equals(mime) && !MediaFormat.MIMETYPE_VIDEO_AVC.equals(mime))
                throw new IllegalArgumentException("The capability report did not select a supported video codec");
            codec = MediaCodec.createByCodecName(codecName);
            MediaCodecInfo info = codec.getCodecInfo();
            if (!info.isEncoder() || !info.isHardwareAccelerated() || info.isSoftwareOnly())
                throw new IllegalStateException("The selected codec is not a hardware encoder");
            MediaCodecInfo.CodecCapabilities capabilities = info.getCapabilitiesForType(mime);
            if (!contains(capabilities.colorFormats, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
                    || capabilities.getVideoCapabilities() == null
                    || !capabilities.getVideoCapabilities().areSizeAndRateSupported(
                            CaptureEngine.WIDTH, CaptureEngine.HEIGHT, CaptureEngine.FPS))
                throw new IllegalStateException("The selected encoder no longer advertises an 8K/30 input surface");
            int bitrate = capabilities.getVideoCapabilities().getBitrateRange().clamp(REQUESTED_BITRATE);
            put(result, "encoder_surface", object("name", codecName, "mime", mime,
                    "width", CaptureEngine.WIDTH, "height", CaptureEngine.HEIGHT, "fps", CaptureEngine.FPS,
                    "requested_bitrate", REQUESTED_BITRATE, "configured_bitrate", bitrate,
                    "max_b_frames", 0, "hardware_accelerated", true, "codec_started", false));
            MediaFormat format = MediaFormat.createVideoFormat(mime, CaptureEngine.WIDTH, CaptureEngine.HEIGHT);
            format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface);
            format.setInteger(MediaFormat.KEY_BIT_RATE, bitrate);
            format.setInteger(MediaFormat.KEY_FRAME_RATE, CaptureEngine.FPS);
            format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1);
            format.setInteger(MediaFormat.KEY_MAX_B_FRAMES, 0);
            stage = "encoder_configuration";
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
            stage = "encoder_surface";
            encoderSurface = codec.createInputSurface();
            stage = "preview_surface";
            previewTexture = new SurfaceTexture(false);
            previewTexture.setDefaultBufferSize(previewWidth, previewHeight);
            previewSurface = new Surface(previewTexture);
            int queryCount = 0;
            int supportedCount = 0;
            int errorCount = 0;
            boolean budgetExhausted = false;
            stage = "session_queries";
            queries: for (long encoderUseCase : selected) {
                for (boolean includePreview : new boolean[]{false, true}) {
                    if (queryCount >= MAX_QUERIES || SystemClock.elapsedRealtime() - startedMs >= ELAPSED_BUDGET_MS) {
                        budgetExhausted = true;
                        break queries;
                    }
                    long combinationStartedMs = SystemClock.elapsedRealtime();
                    JSONObject row = object("output_mode", includePreview ? "encoder_and_preview" : "encoder_only",
                            "encoder_use_case", encoderUseCase,
                            "preview_use_case", includePreview ? previewUseCase : JSONObject.NULL,
                            "supported", JSONObject.NULL, "query_called", false);
                    combinations.put(row);
                    String queryStage = "output_configuration";
                    try {
                        List<OutputConfiguration> outputs = new ArrayList<>();
                        outputs.add(output(encoderSurface, encoderUseCase));
                        if (includePreview) outputs.add(output(previewSurface, previewUseCase));
                        SessionConfiguration session = new SessionConfiguration(SessionConfiguration.SESSION_REGULAR,
                                outputs, Runnable::run, new CameraCaptureSession.StateCallback() {
                                    @Override public void onConfigured(CameraCaptureSession capture) {}
                                    @Override public void onConfigureFailed(CameraCaptureSession capture) {}
                                });
                        session.setSessionParameters(parameters);
                        queryStage = "support_query";
                        queryCount++;
                        put(row, "query_called", true);
                        boolean supported = setup.isSessionConfigurationSupported(session);
                        put(row, "supported", supported);
                        if (supported) supportedCount++;
                    } catch (Exception failure) {
                        errorCount++;
                        put(row, "error_stage", queryStage);
                        put(row, "error", describe(failure));
                    } finally {
                        put(row, "elapsed_ms", SystemClock.elapsedRealtime() - combinationStartedMs);
                        put(result, "query_count", queryCount);
                        put(result, "supported_count", supportedCount);
                    }
                }
            }
            put(result, "error_count", errorCount);
            put(result, "budget_exhausted", budgetExhausted);
            put(result, "status", budgetExhausted ? "budget_exhausted" : errorCount > 0 ? "completed_with_errors" : "completed");
            String reason = supportedCount > 0
                    ? "The HAL accepted " + supportedCount + " queried combinations; direct capture, image detail, and timing remain unverified"
                    : "No queried combination returned support; this report concerns the companion configurations only";
            if (budgetExhausted) reason += "; the elapsed budget stopped further queries";
            if (errorCount > 0) reason += "; " + errorCount + " combinations returned errors";
            put(result, "reason", reason);
        } catch (Exception failure) {
            put(result, "status", "error");
            put(result, "error_stage", stage);
            put(result, "error", describe(failure));
            put(result, "reason", "The diagnostic could not complete at " + stage + "; existing recording readiness is unchanged");
        } finally {
            JSONArray cleanupErrors = new JSONArray();
            if (previewSurface != null) try { previewSurface.release(); }
                catch (RuntimeException failure) { cleanupErrors.put(object("resource", "preview_surface", "error", describe(failure))); }
            if (previewTexture != null) try { previewTexture.release(); }
                catch (RuntimeException failure) { cleanupErrors.put(object("resource", "preview_texture", "error", describe(failure))); }
            if (encoderSurface != null) try { encoderSurface.release(); }
                catch (RuntimeException failure) { cleanupErrors.put(object("resource", "encoder_surface", "error", describe(failure))); }
            if (codec != null) try { codec.release(); }
                catch (RuntimeException failure) { cleanupErrors.put(object("resource", "encoder", "error", describe(failure))); }
            if (cleanupErrors.length() > 0) put(result, "cleanup_errors", cleanupErrors);
            put(result, "elapsed_ms", SystemClock.elapsedRealtime() - startedMs);
            boundReport(result, combinations);
        }
        return result;
    }

    private static CaptureRequest sessionParameters(Context context,String cameraId,CameraDevice.CameraDeviceSetup setup,
            CameraCharacteristics characteristics, JSONObject result) throws Exception {
        CaptureRequest.Builder request = setup.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
        applyRequestSettings(request, characteristics, result);
        FocusSettings.Lock focus=FocusSettings.read(context,cameraId);
        FocusSettings.apply(request,characteristics,cameraId,focus);
        put(result,"focus_control",FocusSettings.metadata(focus));
        return request.build();
    }

    static void applyRequestSettings(CaptureRequest.Builder request,
            CameraCharacteristics characteristics, JSONObject result) {
        List<CaptureRequest.Key<?>> requestKeys = characteristics.getAvailableCaptureRequestKeys();
        List<CaptureRequest.Key<?>> sessionKeys = characteristics.getAvailableSessionKeys();
        JSONArray settings = new JSONArray();
        put(result, "explicit_session_parameters", settings);
        set(request, requestKeys, sessionKeys, settings, CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                new Range<>(CaptureEngine.FPS, CaptureEngine.FPS), new JSONArray().put(CaptureEngine.FPS).put(CaptureEngine.FPS));
        set(request, requestKeys, sessionKeys, settings, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF);
        set(request, requestKeys, sessionKeys, settings, CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF, CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF);
        int[] afModes = characteristics.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES);
        for (int mode : new int[]{CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO,
                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE, CaptureRequest.CONTROL_AF_MODE_AUTO,
                CaptureRequest.CONTROL_AF_MODE_OFF}) if (contains(afModes, mode)) {
            set(request, requestKeys, sessionKeys, settings, CaptureRequest.CONTROL_AF_MODE, mode, mode);
            break;
        }
        Range<Float> zoom = characteristics.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE);
        if (zoom != null && zoom.contains(1.0f))
            set(request, requestKeys, sessionKeys, settings, CaptureRequest.CONTROL_ZOOM_RATIO, 1.0f, 1.0f);
        if (contains(characteristics.get(CameraCharacteristics.SCALER_AVAILABLE_ROTATE_AND_CROP_MODES), CaptureRequest.SCALER_ROTATE_AND_CROP_NONE))
            set(request, requestKeys, sessionKeys, settings, CaptureRequest.SCALER_ROTATE_AND_CROP,
                    CaptureRequest.SCALER_ROTATE_AND_CROP_NONE, CaptureRequest.SCALER_ROTATE_AND_CROP_NONE);
        if (contains(characteristics.get(CameraCharacteristics.DISTORTION_CORRECTION_AVAILABLE_MODES), CaptureRequest.DISTORTION_CORRECTION_MODE_OFF))
            set(request, requestKeys, sessionKeys, settings, CaptureRequest.DISTORTION_CORRECTION_MODE,
                    CaptureRequest.DISTORTION_CORRECTION_MODE_OFF, CaptureRequest.DISTORTION_CORRECTION_MODE_OFF);
    }

    private static <T> void set(CaptureRequest.Builder request, List<CaptureRequest.Key<?>> requestKeys,
            List<CaptureRequest.Key<?>> sessionKeys, JSONArray settings, CaptureRequest.Key<T> key,
            T value, Object jsonValue) {
        boolean available = requestKeys != null && requestKeys.contains(key);
        JSONObject row = object("key", key.getName(), "value", jsonValue, "request_key_available", available,
                "advertised_session_key", sessionKeys != null && sessionKeys.contains(key), "submitted", false);
        settings.put(row);
        if (available) {
            request.set(key, value);
            put(row, "submitted", true);
        }
    }

    static OutputConfiguration output(Surface surface, long useCase) {
        OutputConfiguration output = new OutputConfiguration(surface);
        output.setStreamUseCase(useCase);
        output.setTimestampBase(OutputConfiguration.TIMESTAMP_BASE_SENSOR);
        if (Build.VERSION.SDK_INT >= 34) output.setReadoutTimestampEnabled(false);
        return output;
    }

    private static String requiredString(JSONObject object, String key, int maxLength) {
        String value = object == null ? "" : object.optString(key, "");
        if (value.isEmpty() || value.length() > maxLength) throw new IllegalArgumentException("Invalid capability field " + key);
        return value;
    }

    private static boolean contains(int[] values, int wanted) {
        if (values != null) for (int value : values) if (value == wanted) return true;
        return false;
    }

    private static JSONArray longs(Set<Long> values) {
        JSONArray result = new JSONArray();
        for (long value : values) result.put(value);
        return result;
    }

    private static String describe(Exception failure) {
        String text = failure.getClass().getSimpleName() + (failure.getMessage() == null ? "" : ": " + failure.getMessage());
        return text.length() <= 256 ? text : text.substring(0, 253) + "...";
    }

    private static void boundReport(JSONObject report, JSONArray combinations) {
        put(report, "combination_count", combinations.length());
        put(report, "retained_combination_count", combinations.length());
        // Include the wrapper so pretty-print indentation matches the containing capability report.
        while (reportBytes(report) > MAX_REPORT_BYTES && combinations.length() > 0) {
            combinations.remove(combinations.length() - 1);
            put(report, "report_truncated", true);
            put(report, "retained_combination_count", combinations.length());
        }
        if (reportBytes(report) > MAX_REPORT_BYTES) {
            report.remove("explicit_session_parameters");
            report.remove("observed_stream_use_cases");
            put(report, "report_truncated", true);
        }
    }

    private static int reportBytes(JSONObject report) {
        try { return object("session_queries", report).toString(2).getBytes(StandardCharsets.UTF_8).length; }
        catch (JSONException failure) { return Integer.MAX_VALUE; }
    }

    private static JSONObject object(Object... values) {
        JSONObject result = new JSONObject();
        for (int i = 0; i < values.length; i += 2) put(result, (String) values[i], values[i + 1]);
        return result;
    }

    private static void put(JSONObject object, String key, Object value) {
        try { object.put(key, value == null ? JSONObject.NULL : value); }
        catch (JSONException failure) { throw new IllegalArgumentException("Invalid JSON field " + key, failure); }
    }
}

package org.noesis.roomwalk;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.util.Arrays;

/** A retained, completed short take can qualify the same mode for longer recording. */
public final class CapturePreflight {
    private static final int MAX_RESULTS = 64;
    private CapturePreflight() {}

    public static JSONObject find(File captureRoot, JSONObject candidate, String buildFingerprint) {
        File[] directories = captureRoot.listFiles(file -> file.isDirectory()
                && file.getName().matches("\\d{8}-\\d{6}-[0-9a-f]{8}"));
        if (directories == null) return null;
        Arrays.sort(directories, (left, right) -> Long.compare(right.lastModified(), left.lastModified()));
        for (int i = 0; i < Math.min(directories.length, MAX_RESULTS); i++) {
            File result = new File(directories[i], "capture_result.json");
            if (!result.isFile()) continue;
            try {
                JSONObject recorded = BundleTools.readJson(result);
                if (matches(recorded, candidate, buildFingerprint)) return reference(directories[i].getName(), recorded);
            } catch (Exception ignored) {
                // An unreadable or incomplete take cannot qualify a camera mode.
            }
        }
        return null;
    }

    public static JSONObject reference(String captureId, JSONObject result) throws org.json.JSONException {
        JSONObject session = result.getJSONObject("camera").getJSONObject("recording_session");
        return new JSONObject().put("capture_id", captureId)
                .put("encoder_use_case", session.getLong("encoder_use_case"))
                .put("preview_use_case", session.getLong("preview_use_case"))
                .put("focus_control",result.getJSONObject("camera").opt("focus_control"));
    }

    public static boolean qualifiesWalk(JSONObject candidate, long encoderUseCase) {
        if (candidate == null || !candidate.optBoolean("direct_session_test_eligible")) return false;
        JSONObject recorded = candidate.optJSONObject("recorded_preflight");
        return recorded != null && FocusSettings.sameConfiguration(candidate.optJSONObject("focus_control"),recorded.optJSONObject("focus_control")) && !recorded.optString("capture_id").isEmpty()
                && (encoderUseCase == 0 || encoderUseCase == 3)
                && recorded.optLong("encoder_use_case", -1) == encoderUseCase
                && recorded.optLong("preview_use_case", -1) == candidate.optLong("standard_preview_use_case", -2);
    }

    public static boolean matches(JSONObject result, JSONObject candidate, String buildFingerprint) {
        if (result == null || candidate == null || buildFingerprint == null || buildFingerprint.isEmpty()) return false;
        JSONObject device = result.optJSONObject("device"), camera = result.optJSONObject("camera"),
                encoder = result.optJSONObject("encoder"), expectedEncoder = candidate.optJSONObject("encoder"),
                timing = result.optJSONObject("timing"), bounds = result.optJSONObject("bounds");
        JSONArray failures = result.optJSONArray("failures");
        if (device == null || camera == null || encoder == null || expectedEncoder == null
                || timing == null || bounds == null || failures == null) return false;
        if (!"noesis.phone_capture.android_result.v1".equals(result.optString("schema"))
                || !"complete".equals(result.optString("status")) || result.optBoolean("partial", true)
                || !result.optBoolean("export_ready") || failures.length() != 0
                || result.optLong("dropped_metadata_records", -1) != 0
                || !buildFingerprint.equals(device.optString("build_fingerprint"))
                || !bounds.optBoolean("short_test") || bounds.optLong("max_duration_ms") != 10_000
                || !"short_test_duration_limit_10_seconds".equals(result.optString("stop_reason"))) return false;
        if (!candidate.optBoolean("direct_session_test_eligible")
                || !camera.optString("id").equals(candidate.optString("id"))
                || camera.optInt("width") != CaptureEngine.WIDTH || camera.optInt("height") != CaptureEngine.HEIGHT
                || camera.optInt("fps") != CaptureEngine.FPS
                || !"REALTIME".equals(camera.optString("timestamp_source"))
                || !"SENSOR".equals(camera.optString("timestamp_base")) || !camera.optBoolean("timestamp_base_configured")
                || camera.optInt("sensor_orientation_degrees", -1) != candidate.optInt("sensor_orientation_degrees", -2)
                || !"OFF".equals(camera.optString("ois_requested")) || !"OFF".equals(camera.optString("eis_requested"))) return false;
        JSONObject focus=camera.optJSONObject("focus_control");
        if(!FocusSettings.sameConfiguration(candidate.optJSONObject("focus_control"),focus))return false;
        if(focus!=null&&"manual_locked".equals(focus.optString("mode"))&&(focus.optLong("confirmed_frame_count")<=0||focus.optLong("unconfirmed_frame_count",-1)!=0))return false;
        JSONObject session = camera.optJSONObject("recording_session");
        if (session == null || !session.optBoolean("query_called") || !session.optBoolean("supported")
                || !candidate.optString("id").equals(session.optString("camera_id"))
                || session.optInt("width") != CaptureEngine.WIDTH || session.optInt("height") != CaptureEngine.HEIGHT
                || session.optInt("fps") != CaptureEngine.FPS
                || !session.optBoolean("capture_session_configured") || !session.optBoolean("same_configuration_used_for_capture")
                || !session.optBoolean("short_test")
                || (session.optLong("encoder_use_case", -1) != 0 && session.optLong("encoder_use_case", -1) != 3)
                || (session.optLong("preview_use_case", -1) != 0 && session.optLong("preview_use_case", -1) != 1)
                || session.optLong("preview_use_case", -1) != candidate.optLong("standard_preview_use_case", -2)
                || session.optInt("preview_width") != candidate.optInt("preview_width")
                || session.optInt("preview_height") != candidate.optInt("preview_height")) return false;
        if (!encoder.optString("name").equals(expectedEncoder.optString("name"))
                || !encoder.optString("mime").equals(expectedEncoder.optString("mime"))
                || encoder.optInt("requested_bitrate") != expectedEncoder.optInt("requested_bitrate")
                || encoder.optInt("fps") != CaptureEngine.FPS || encoder.optInt("max_b_frames", -1) != 0
                || encoder.optInt("width") != CaptureEngine.WIDTH || encoder.optInt("height") != CaptureEngine.HEIGHT
                || !encoder.optBoolean("hardware_accelerated") || !encoder.optBoolean("mp4_finalized")
                || !encoder.optBoolean("eos_received")) return false;
        String physicalId = camera.optString("active_physical_camera_id", "");
        JSONArray physicalIds = candidate.optJSONArray("physical_camera_ids");
        if (physicalIds != null && physicalIds.length() > 0) {
            boolean present = false;
            for (int i = 0; i < physicalIds.length(); i++) if (physicalId.equals(physicalIds.optString(i))) present = true;
            if (!present) return false;
        }
        long frames = timing.optLong("encoded_frame_count", -1);
        if (!"encoder_pts_us_equals_sensor_timestamp_ns_div_1000".equals(timing.optString("association_method"))
                || !timing.optBoolean("exact_frame_association_verified") || !timing.optBoolean("imu_coverage_verified")
                || !timing.optBoolean("sensor_timestamps_monotonic") || !timing.optBoolean("encoder_pts_monotonic")
                || frames < 2 || timing.optLong("matched_frame_count", -2) != frames
                || timing.optLong("camera_result_count", -2) != frames
                || timing.optLong("camera_results_with_sensor_timestamp_count", -2) != frames
                || timing.optLong("camera_results_without_encoded_frame_count", -1) != 0
                || timing.optLong("duplicate_camera_frame_number_count", -1) != 0
                || timing.optLong("unmatched_encoded_frame_count", -1) != 0
                || timing.optLong("duplicate_encoded_pts_count", -1) != 0
                || timing.optLong("duplicate_sensor_timestamp_us_count", -1) != 0) return false;
        long first = timing.optLong("first_matched_sensor_timestamp_ns"), last = timing.optLong("last_matched_sensor_timestamp_ns");
        double duration = (last - first) / 1e9;
        double measuredFps = duration > 0 ? (frames - 1) / duration : 0;
        // Require enough of the ten-second take to measure cadence; allow startup
        // latency and fractional camera clocks around the requested nominal 30 FPS.
        return first > 0 && duration >= 8 && duration <= 12
                && measuredFps >= CaptureEngine.FPS - 1 && measuredFps <= CaptureEngine.FPS + 1;
    }
}

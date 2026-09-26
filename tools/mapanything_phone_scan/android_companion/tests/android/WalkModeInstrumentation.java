package org.noesis.roomwalk;

import android.app.Instrumentation;
import android.os.Bundle;
import org.json.JSONObject;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.UUID;

/** Device checks for walk intent boundaries, retained metadata, and focus semantics. */
public final class WalkModeInstrumentation extends Instrumentation {
    private static final String TARGET = "20260920-123456-deadbeef";
    private static final String PRIOR = "pcf_prior_living_room_v3";
    private static final String MANIFEST = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    private static final String BINDING = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    private int checks;

    @Override public void onCreate(Bundle arguments) { super.onCreate(arguments); start(); }

    @Override public void onStart() {
        Bundle result = new Bundle();
        try {
            checkWalkIntentContract();
            checkSavedMetadata();
            checkFocusSemanticBinding();
            checkModeSpecificTimingProof();
            result.putString("stream", "Walk mode instrumentation: " + checks + " checks passed\n");
            finish(-1, result);
        } catch (Throwable error) {
            result.putString("stream", "FAILED: " + android.util.Log.getStackTraceString(error));
            finish(1, result);
        }
    }

    private void checkWalkIntentContract() throws Exception {
        JSONObject legacy = WalkIntent.create("walk", null);
        require(WalkIntent.RECONSTRUCTION.equals(legacy.getString("mode")), "Legacy walk action normalizes to reconstruction");
        require(WalkIntent.isWalkMode("walk") && !WalkIntent.isWalkMode("camera"), "Only walk actions enter the walk boundary");
        JSONObject path = WalkIntent.create(WalkIntent.PATH_REFINEMENT, TARGET);
        require(TARGET.equals(path.getString("target_scan_id")), "Valid backend-shaped target is retained");
        require(WalkIntent.PATH_REFINEMENT.equals(WalkIntent.validate(path).getString("mode")), "Strict path intent validates");
        JSONObject selection = pcfSelection();
        JSONObject selectedPath = WalkIntent.create(WalkIntent.PATH_REFINEMENT, TARGET, selection);
        require(selection.toString().equals(selectedPath.getJSONObject("target_reference").toString()), "Exact retained PCF selection is preserved");
        require(!WalkIntent.create(WalkIntent.RECONSTRUCTION, TARGET).has("target_reference"), "Reconstruction remains unbound to PCF selection");
        expectRejected(WalkIntent.create(WalkIntent.PATH_REFINEMENT, TARGET, selection).put("target_reference", new JSONObject(selection.toString()).put("extra", true)), "Unsupported PCF selection fields are rejected");
        expectRejected(new JSONObject(selectedPath.toString()).put("target_reference", new JSONObject(selection.toString()).put("manifest_sha256", MANIFEST.toUpperCase())), "Uppercase PCF digest is rejected");
        expectRejected(new JSONObject(WalkIntent.create(WalkIntent.RECONSTRUCTION, TARGET).toString()).put("target_reference", selection), "Reconstruction cannot carry PCF selection");
        expectRejected(new JSONObject(legacy.toString()).put("mode", "walk"), "Legacy alias is not accepted inside persisted schema");
        for (String invalid : new String[]{"scan-123", "20260920-123456-ABCDEF12", "20260920-123456-deadbeeg"}) {
            try { WalkIntent.create(WalkIntent.RECONSTRUCTION, invalid); throw new AssertionError("Invalid target accepted: " + invalid); }
            catch (IllegalArgumentException expected) { checks++; }
        }
        expectRejected(new JSONObject(path.toString()).put("unexpected", true), "Unknown persisted walk field is rejected");
    }

    private void checkSavedMetadata() throws Exception {
        File directory = new File(getTargetContext().getCacheDir(), "walk-mode-" + UUID.randomUUID());
        if (!directory.mkdirs()) throw new AssertionError("Could not create isolated metadata fixture");
        try {
            JSONObject intent = WalkIntent.create(WalkIntent.PATH_REFINEMENT, TARGET, pcfSelection());
            BundleTools.writeJson(new File(directory, "walk_intent.json"), intent);
            JSONObject files = new JSONObject().put("video", "camera.mp4")
                    .put("accelerometer", "accel.csv").put("gyroscope", "gyro.csv")
                    .put("encoder_pts", "encoder_pts.csv").put("camera_results", "camera_results.jsonl");
            JSONObject capture = new JSONObject()
                    .put("camera", new JSONObject().put("id", "0").put("width", CaptureEngine.WIDTH).put("height", CaptureEngine.HEIGHT))
                    .put("files", files).put("timing", new JSONObject().put("exact_frame_association_verified", false));
            BundleTools.writeCaptureManifest(getTargetContext(), directory, capture);
            JSONObject manifest = BundleTools.readJson(new File(directory, "capture_manifest.json"));
            JSONObject saved = manifest.getJSONObject("walk_intent");
            require(WalkIntent.PATH_REFINEMENT.equals(saved.getString("mode")), "Saved manifest preserves walk mode");
            require(TARGET.equals(saved.getString("target_scan_id")), "Saved manifest preserves exact target identity");
            require("close_body".equals(saved.getString("carry_protocol")), "Saved manifest preserves close-body protocol");
            require(MANIFEST.equals(saved.getJSONObject("target_reference").getString("manifest_sha256")), "Saved manifest preserves exact PCF manifest digest");
            require(BINDING.equals(saved.getJSONObject("target_reference").getString("frame_binding_sha256")), "Saved manifest preserves exact PCF frame binding digest");
        } finally {
            delete(directory);
        }
    }

    private void checkFocusSemanticBinding() throws Exception {
        FocusSettings.Lock lock = new FocusSettings.Lock("0", "5", "test-build", 2.631579f, 1000, 2000);
        JSONObject guard = lock.json();
        JSONObject request = new JSONObject().put("mode", "imu").put("calibration_focus_guard", guard);
        require(FocusSettings.matchesMotionGuard(request, new JSONObject(guard.toString())), "Exact logical, physical, build and focus settings match");
        for (String key : new String[]{"logical_camera_id", "physical_camera_id", "build_fingerprint", "output_routing_policy", "capture_width", "capture_fps"}) {
            JSONObject changed = new JSONObject(guard.toString());
            Object value = changed.opt(key);
            changed.put(key, value instanceof Number ? ((Number) value).intValue() + 1 : value + "-changed");
            require(!FocusSettings.matchesMotionGuard(request, changed), "Focus guard rejects changed " + key);
        }
        JSONObject changedDistance = new JSONObject(guard.toString()).put("focus_distance_diopters", 2.63158);
        require(!FocusSettings.matchesMotionGuard(request, changedDistance), "Focus guard rejects changed lens distance");
        require(!FocusSettings.matchesMotionGuard(new JSONObject().put("mode", "imu"), guard), "Missing focus guard fails closed");
        require(!FocusSettings.matchesMotionGuard(new JSONObject().put("mode", "reconstruction").put("calibration_focus_guard", guard), guard), "Non-IMU capture does not claim motion focus binding");
    }

    private void checkModeSpecificTimingProof() throws Exception {
        JSONObject locked = new FocusSettings.Lock("0", "5", "test-build", 2.631579f, 1000, 2000).json();
        JSONObject reference = new JSONObject().put("capture_id", "retained-board-test")
                .put("encoder_use_case", 3).put("preview_use_case", 1).put("focus_control", locked);
        JSONObject candidate = new JSONObject().put("direct_session_test_eligible", true)
                .put("standard_preview_use_case", 1).put("focus_control", locked).put("recorded_preflight", reference);
        require(CapturePreflight.qualifiesWalk(candidate, 3, true), "Matching calibration proof remains usable");
        require(!CapturePreflight.qualifiesWalk(candidate, 3, false), "Pinned calibration proof cannot qualify automatic capture");
        JSONObject automatic = new JSONObject(reference.toString()).put("capture_id", "retained-automatic-test")
                .put("focus_control", new JSONObject().put("mode", "automatic"));
        candidate.put("automatic_recorded_preflight", automatic);
        require(CapturePreflight.qualifiesWalk(candidate, 3, false), "Automatic walk proof is usable with saved calibration focus unchanged");
        require(CapturePreflight.qualifiesWalk(candidate, 3, true), "Both modes retain their separate timing proof");
        require(!CapturePreflight.qualifiesWalk(candidate, 0, false), "Another use case cannot reuse automatic timing proof");
    }

    private void expectRejected(JSONObject value, String label) throws Exception {
        try { WalkIntent.validate(value); throw new AssertionError(label); }
        catch (IllegalArgumentException expected) { checks++; }
    }

    private JSONObject pcfSelection() throws Exception {
        return new JSONObject().put("kind", "scene_prior_pcf").put("prior_id", PRIOR)
                .put("manifest_sha256", MANIFEST).put("camera_id", "living_room")
                .put("frame_binding_sha256", BINDING);
    }

    private void require(boolean passed, String label) { if (!passed) throw new AssertionError(label); checks++; }

    private static void delete(File file) {
        File[] children = file.listFiles();
        if (children != null) for (File child : children) delete(child);
        file.delete();
    }
}

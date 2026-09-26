package org.noesis.roomwalk;

import org.json.JSONObject;

/** Native-only intent metadata for reconstruction and paired path-refinement evidence. */
public final class WalkIntent {
    public static final String SCHEMA = "roomwalk.walk_intent.v1";
    public static final String RECONSTRUCTION = "reconstruction";
    public static final String PATH_REFINEMENT = "path_refinement";
    public static final double ACCURACY_TARGET_M = 0.1;
    private static final String TARGET_SCAN_ID_PATTERN = "^[0-9]{8}-[0-9]{6}-[a-f0-9]{8}$";
    private static final String PRIOR_ID_PATTERN = "^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$";
    private static final String CAMERA_ID_PATTERN = "^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$";
    private static final String SHA256_PATTERN = "^[0-9a-f]{64}$";

    private WalkIntent() {}

    /** Normalize the legacy native action without changing the caller's other arguments. */
    public static String normalizeMode(String mode) {
        if (mode == null || mode.trim().isEmpty() || "walk".equals(mode)) return RECONSTRUCTION;
        if (RECONSTRUCTION.equals(mode) || PATH_REFINEMENT.equals(mode)) return mode;
        throw new IllegalArgumentException("Choose reconstruction or path refinement");
    }

    public static boolean isWalkMode(String mode) {
        return RECONSTRUCTION.equals(mode) || PATH_REFINEMENT.equals(mode) || "walk".equals(mode);
    }

    public static boolean requiresPair(String mode) {
        return isWalkMode(mode) && PATH_REFINEMENT.equals(normalizeMode(mode));
    }

    public static JSONObject create(String mode, String targetScanId) throws Exception {
        return create(mode, targetScanId, null);
    }

    /** Create an intent while retaining the exact selected PCF binding for path mode only. */
    public static JSONObject create(String mode, String targetScanId, JSONObject targetReference) throws Exception {
        String normalized = normalizeMode(mode);
        String target = targetScanId == null ? null : targetScanId.trim();
        if (target != null && target.isEmpty()) target = null;
        if (target != null && !target.matches(TARGET_SCAN_ID_PATTERN))
            throw new IllegalArgumentException("The target reconstruction ID is invalid");
        if (PATH_REFINEMENT.equals(normalized) && target == null)
            throw new IllegalArgumentException("Path refinement must name an existing reconstruction");
        JSONObject result = new JSONObject().put("schema", SCHEMA).put("mode", normalized)
                .put("target_scan_id", target == null ? JSONObject.NULL : target)
                .put("carry_protocol", PATH_REFINEMENT.equals(normalized) ? "close_body" : "coverage")
                .put("accuracy_target_m", ACCURACY_TARGET_M);
        if (targetReference != null) {
            if (!PATH_REFINEMENT.equals(normalized))
                throw new IllegalArgumentException("A retained PCF selection is valid only for path refinement");
            result.put("target_reference", validateTargetReference(targetReference));
        }
        return validate(result);
    }

    private static boolean isTargetReferenceKey(String key) {
        return "kind".equals(key) || "prior_id".equals(key) || "manifest_sha256".equals(key)
                || "camera_id".equals(key) || "frame_binding_sha256".equals(key);
    }

    private static JSONObject validateTargetReference(JSONObject value) throws Exception {
        if (value == null) throw new IllegalArgumentException("The retained PCF selection is invalid");
        int count = 0;
        java.util.Iterator<String> keys = value.keys();
        while (keys.hasNext()) {
            count++;
            String key = keys.next();
            if (!isTargetReferenceKey(key))
                throw new IllegalArgumentException("Unknown retained PCF selection field: " + key);
        }
        if (count != 5) throw new IllegalArgumentException("The retained PCF selection must contain exactly five fields");
        if (!"scene_prior_pcf".equals(value.optString("kind", null)))
            throw new IllegalArgumentException("The retained PCF selection kind is unsupported");
        String priorId = value.optString("prior_id", null);
        String cameraId = value.optString("camera_id", null);
        String manifest = value.optString("manifest_sha256", null);
        String frameBinding = value.optString("frame_binding_sha256", null);
        if (priorId == null || !priorId.matches(PRIOR_ID_PATTERN))
            throw new IllegalArgumentException("The retained PCF prior_id is invalid");
        if (cameraId == null || !cameraId.matches(CAMERA_ID_PATTERN))
            throw new IllegalArgumentException("The retained PCF camera_id is invalid");
        if (manifest == null || !manifest.matches(SHA256_PATTERN))
            throw new IllegalArgumentException("The retained PCF manifest_sha256 must be lowercase SHA-256");
        if (frameBinding == null || !frameBinding.matches(SHA256_PATTERN))
            throw new IllegalArgumentException("The retained PCF frame_binding_sha256 must be lowercase SHA-256");
        return new JSONObject(value.toString());
    }

    public static JSONObject validate(JSONObject value) throws Exception {
        Object rawSchema = value == null ? null : value.opt("schema");
        if (!(rawSchema instanceof String) || !SCHEMA.equals(rawSchema))
            throw new IllegalArgumentException("Invalid walk intent schema");
        String[] allowed = {"schema", "mode", "target_scan_id", "carry_protocol", "accuracy_target_m", "target_reference"};
        java.util.Iterator<String> keys = value.keys();
        while (keys.hasNext()) {
            String key = keys.next();
            boolean known = false;
            for (String candidate : allowed) if (candidate.equals(key)) { known = true; break; }
            if (!known) throw new IllegalArgumentException("Unknown walk intent field: " + key);
        }
        Object rawModeValue = value.opt("mode");
        if (!(rawModeValue instanceof String)) throw new IllegalArgumentException("Walk intent mode is invalid");
        String rawMode = (String) rawModeValue;
        if (!RECONSTRUCTION.equals(rawMode) && !PATH_REFINEMENT.equals(rawMode))
            throw new IllegalArgumentException("Walk intent mode must be reconstruction or path_refinement");
        String mode = rawMode;
        Object rawCarry = value.opt("carry_protocol");
        if (!(rawCarry instanceof String)) throw new IllegalArgumentException("Walk intent carry protocol is invalid");
        String carry = (String) rawCarry;
        String expectedCarry = PATH_REFINEMENT.equals(mode) ? "close_body" : "coverage";
        if (!expectedCarry.equals(carry)) throw new IllegalArgumentException("Walk intent carry protocol does not match its mode");
        Object rawAccuracy = value.opt("accuracy_target_m");
        if (!(rawAccuracy instanceof Number) || ((Number) rawAccuracy).doubleValue() != ACCURACY_TARGET_M)
            throw new IllegalArgumentException("Walk intent accuracy target must remain 0.1 m");
        if (!value.has("target_scan_id")) throw new IllegalArgumentException("Walk intent target_scan_id is required");
        Object rawTarget = value.opt("target_scan_id");
        if (rawTarget != null && rawTarget != JSONObject.NULL && !(rawTarget instanceof String))
            throw new IllegalArgumentException("The target reconstruction ID must be a string or null");
        String target = rawTarget == null || rawTarget == JSONObject.NULL ? null : (String) rawTarget;
        if (PATH_REFINEMENT.equals(mode) && (target == null || target.trim().isEmpty()))
            throw new IllegalArgumentException("Path refinement must name an existing reconstruction");
        if (target != null && !target.matches(TARGET_SCAN_ID_PATTERN))
            throw new IllegalArgumentException("The target reconstruction ID is invalid");
        if (value.has("target_reference")) {
            if (!PATH_REFINEMENT.equals(mode))
                throw new IllegalArgumentException("A retained PCF selection is valid only for path refinement");
            Object rawReference = value.opt("target_reference");
            if (!(rawReference instanceof JSONObject))
                throw new IllegalArgumentException("The retained PCF selection must be an object");
            validateTargetReference((JSONObject) rawReference);
        }
        return new JSONObject(value.toString());
    }
}

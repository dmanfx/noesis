package org.noesis.roomwalk;

/** Timed instructions, not a claim that the camera saw a target or that a walk is accurate. */
public final class CaptureGuide {
    private CaptureGuide() {}

    public static int targetSeconds(String mode) {
        return "camera".equals(mode) ? 60 : "imu".equals(mode) ? 90 : 0;
    }

    public static final class Step {
        public final String phase, title, instruction;
        public final boolean stationary, completed;
        Step(String phase, String title, String instruction, boolean stationary, boolean completed) {
            this.phase = phase; this.title = title; this.instruction = instruction;
            this.stationary = stationary; this.completed = completed;
        }
    }

    public static Step at(String mode, long seconds) { return at(mode, seconds, null); }

    public static Step at(String mode, long seconds, String targetScanId) {
        seconds = Math.max(0, seconds);
        int target = targetSeconds(mode);
        if (target > 0 && seconds >= target)
            return new Step("saving", "Take finished — saving", "Keep RoomWalk open. Upload this take next; processing checks whether the data is sufficient.", true, true);
        if ("camera".equals(mode)) {
            if (seconds < 5) return new Step("settle", "Keep the board fixed and sharp", "Hold the phone at the distance where the board was sharp when you locked focus. Focus stays locked throughout this take.", true, false);
            if (seconds < 20) return new Step("coverage", "Move the phone · cover the image", "Move sideways and up/down so the board appears near the center, edges and corners. Keep the whole board visible; pause briefly at each view.", false, false);
            if (seconds < 40) return new Step("angles", "Change angle; keep the board sharp", "View the fixed board from left, right, above and below. Stay near the focused distance; reduce the movement if any board edge becomes blurry. Do not refocus during the take.", false, false);
            return new Step("repeat", "Repeat varied, sharp views", "Visit the image corners again from different angles while staying within the sharp focus range. Move the phone, not the board. The take saves automatically at 60 seconds.", false, false);
        }
        if ("imu".equals(mode)) {
            if (seconds < 8) return new Step("settle", "Hold the phone still", "Keep the fixed board sharp and in view at the locked-focus distance. Do not refocus. This quiet start helps initialize camera and motion tracking.", true, false);
            if (seconds < 30) return new Step("rotation", "Gently tilt and turn the phone", "Tilt left/right, nod up/down, and turn side to side while keeping the board visible and sharp. Add small sideways and forward/backward movements only within the sharp focus range.", false, false);
            if (seconds < 55) return new Step("translation", "Move in three directions", "Move left/right, up/down, and a little toward/away from the fixed board, staying within the sharp focus range. Reduce movement if the board blurs; do not refocus. Use smooth motion, not jerks.", false, false);
            if (seconds < 82) return new Step("validation", "Repeat the mixed movements", "Repeat tilts and movement in every direction within the sharp focus range. This part checks the calibration on separate observations; keep the board fixed and visible. Do not refocus.", false, false);
            return new Step("finish", "Hold still to finish", "Keep the board in view and the phone still. The take saves automatically at 90 seconds.", true, false);
        }
        if (WalkIntent.PATH_REFINEMENT.equals(mode)) {
            if (seconds < 8) return new Step("settle", "Hold the phone against your own torso", "Keep the phone against or near your own torso with your elbows tucked while the paired room camera records. Establish a fixed phone-to-body relation; this is evidence for refinement, not a live accuracy certificate.", true, false);
            if (seconds < 45) return new Step("close_body", "Walk with the phone held to your torso", "Hold the phone against or near your own torso with elbows tucked. Turn and move your whole body with the phone while the paired room camera sees the same person. Do not pan with an independent arm or sweep the phone; do not convert the phone pose into a body-ground point.", false, false);
            if (seconds < 120) return new Step("whole_body", "Continue the whole-body walk", "Continue along the occupied route with the same fixed phone-to-body relation. Make whole-body turns with the phone, keep the static pairing active, and do not sweep the phone or vary its distance from your body.", false, false);
            return new Step("return", "Finish with the same body-held phone", "Return through the useful part of the occupied route while keeping the phone against or near your own torso and moving your whole body with it. Hold still briefly and tap Stop & save. The 10 cm target is retained as a validation target, not asserted by capture timing.", false, false);
        }
        boolean supplement = targetScanId != null && !targetScanId.trim().isEmpty();
        if (seconds < 8) return new Step("settle", supplement ? "Begin the retained reconstruction supplement" : "Begin with a steady room view", supplement ? "Hold still briefly, then add views that cover areas missing from the selected reconstruction. Keep the phone moving slowly and preserve the target ID in this capture." : "Hold still briefly, then begin a slow room walk. The phone video and motion evidence stay separate from any later static-camera alignment.", true, false);
        if (seconds < 60) return new Step("coverage", supplement ? "Add useful overlapping views" : "Build overlapping room coverage", supplement ? "Move through the selected reconstruction's under-covered areas. Add sharp, overlapping views of corners, doorways and transitions; do not replace the retained scan's identity." : "Cover the room from varied positions with overlapping views. Include corners, doorways and transitions, and move slowly enough to keep the scene useful.", false, false);
        if (seconds < 240) return new Step("revisit", "Create overlap and parallax", supplement ? "Revisit the supplement route from a second nearby angle and preserve overlap with the retained room geometry. Avoid empty repeated frames." : "Revisit important areas from a second nearby angle. Keep overlap and parallax, and avoid spending the whole walk on near-identical frames.", false, false);
        return new Step("return", "Revisit the starting area", "Finish the room coverage, return to the starting area, then hold still briefly and tap Stop & save. Static pairing is optional for reconstruction capture.", false, false);
    }
}

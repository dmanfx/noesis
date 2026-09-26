import org.noesis.roomwalk.CaptureGuide;
import org.noesis.roomwalk.WalkIntent;

public final class CaptureGuideTest {
    private static void check(boolean value, String message) {
        if (!value) throw new AssertionError(message);
    }
    public static void main(String[] args) {
        check(CaptureGuide.targetSeconds("camera") == 60, "short camera take");
        check(CaptureGuide.targetSeconds("imu") == 90, "short IMU take");
        check(CaptureGuide.targetSeconds(WalkIntent.RECONSTRUCTION) == 0, "reconstruction is stopped by the operator");
        check(CaptureGuide.targetSeconds(WalkIntent.PATH_REFINEMENT) == 0, "path refinement is stopped by the operator");

        check(CaptureGuide.at(WalkIntent.RECONSTRUCTION, 0).stationary, "reconstruction starts steady");
        check(CaptureGuide.at(WalkIntent.RECONSTRUCTION, 10).phase.equals("coverage"), "reconstruction coverage guidance");
        check(CaptureGuide.at(WalkIntent.RECONSTRUCTION, 10).instruction.contains("corners, doorways"), "reconstruction asks for useful room views");
        check(CaptureGuide.at(WalkIntent.RECONSTRUCTION, 10, "20260920-123456-deadbeef").instruction.contains("selected reconstruction"), "supplement guidance names the retained target");
        check(CaptureGuide.at(WalkIntent.RECONSTRUCTION, 600).instruction.contains("Static pairing is optional"), "reconstruction does not require pairing");

        check(CaptureGuide.at(WalkIntent.PATH_REFINEMENT, 0).stationary, "path refinement starts steady");
        check(CaptureGuide.at(WalkIntent.PATH_REFINEMENT, 10).phase.equals("close_body"), "path refinement close-body guidance");
        String pathGuide = CaptureGuide.at(WalkIntent.PATH_REFINEMENT, 10).instruction + " "
                + CaptureGuide.at(WalkIntent.PATH_REFINEMENT, 60).instruction + " "
                + CaptureGuide.at(WalkIntent.PATH_REFINEMENT, 130).instruction;
        check(pathGuide.contains("your own torso") && pathGuide.contains("elbows tucked"), "path refinement keeps the phone against the walker's own torso");
        check(pathGuide.contains("whole body") && pathGuide.contains("same person"), "path refinement moves the whole body with the phone");
        check(pathGuide.contains("paired room camera"), "path refinement calls for pairing");
        check(pathGuide.contains("do not convert"), "path refinement does not infer body-ground points");
        check(pathGuide.contains("Do not pan") && pathGuide.contains("do not sweep") && pathGuide.contains("vary its distance"), "path refinement forbids arm sweeps and changing phone-body distance");
        check(!pathGuide.contains("nearby angles") && !pathGuide.contains("paired_sweep"), "path refinement has no reconstruction sweep guidance");
        check(CaptureGuide.at(WalkIntent.PATH_REFINEMENT, 130).instruction.contains("10 cm"), "accuracy target is not claimed by the guide");

        check(CaptureGuide.at("camera", 0).instruction.contains("locked focus"), "camera calibration keeps its board guidance");
        check(CaptureGuide.at("imu", 0).instruction.contains("Do not refocus"), "IMU calibration keeps its focus guidance");
        check(CaptureGuide.at("camera", 60).completed, "camera duration reached");
        check(CaptureGuide.at("imu", 90).completed, "IMU duration reached");
        System.out.println("CaptureGuideTest passed");
    }
}

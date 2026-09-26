import org.json.JSONObject;
import org.noesis.roomwalk.WalkIntent;

public final class WalkIntentTest {
    private static int checks;
    private static final String TARGET = "20260920-123456-deadbeef";
    private static final String MANIFEST = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    private static final String BINDING = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    private static void check(boolean value, String message) { checks++; if (!value) throw new AssertionError(message); }

    public static void main(String[] args) throws Exception {
        JSONObject reconstruction=WalkIntent.create("reconstruction",null);
        check(WalkIntent.RECONSTRUCTION.equals(reconstruction.getString("mode")), "reconstruction mode");
        check(reconstruction.isNull("target_scan_id"), "unbound reconstruction target is null");
        check("coverage".equals(reconstruction.getString("carry_protocol")), "coverage protocol");
        check(reconstruction.getDouble("accuracy_target_m")==0.1, "exact validation target");
        check(WalkIntent.RECONSTRUCTION.equals(WalkIntent.create("walk",null).getString("mode")), "legacy walk aliases reconstruction");
        check(!WalkIntent.requiresPair("reconstruction"), "reconstruction pairing is optional");
        check(WalkIntent.requiresPair("path_refinement"), "path refinement pairing is required");

        JSONObject supplement=WalkIntent.create("reconstruction",TARGET);
        check(TARGET.equals(supplement.getString("target_scan_id")), "reconstruction target retained");
        check(WalkIntent.PATH_REFINEMENT.equals(WalkIntent.create("path_refinement",TARGET).getString("mode")), "path refinement mode");
        check("close_body".equals(WalkIntent.create("path_refinement",TARGET).getString("carry_protocol")), "close-body protocol");
        JSONObject selection=new JSONObject().put("kind","scene_prior_pcf").put("prior_id","pcf_prior_living_room_v3")
                .put("manifest_sha256",MANIFEST).put("camera_id","living_room").put("frame_binding_sha256",BINDING);
        JSONObject selectedPath=WalkIntent.create("path_refinement",TARGET,selection);
        check(selection.toString().equals(selectedPath.getJSONObject("target_reference").toString()), "exact PCF selection retained");
        check(!supplement.has("target_reference"), "reconstruction has no PCF selection");

        boolean missing=false; try { WalkIntent.create("path_refinement",null); } catch (IllegalArgumentException expected) { missing=true; }
        check(missing, "path refinement requires a target");
        boolean extra=false; try { WalkIntent.validate(new JSONObject(reconstruction.toString()).put("capture_id","wrong")); } catch (IllegalArgumentException expected) { extra=true; }
        check(extra, "sidecar fields are strict");
        boolean wrongTarget=false; try { WalkIntent.validate(WalkIntent.create("path_refinement",TARGET).put("accuracy_target_m",0.2)); } catch (IllegalArgumentException expected) { wrongTarget=true; }
        check(wrongTarget, "accuracy target cannot drift");
        for (String invalid : new String[]{"scan-123", "20260920-123456-ABCDEF12", "2026092-123456-deadbeef", "20260920-12345-deadbeef", "20260920-123456-deadbeeg", "20260920-123456-deadbeef/extra"}) {
            boolean rejected=false; try { WalkIntent.create("reconstruction",invalid); } catch (IllegalArgumentException expected) { rejected=true; }
            check(rejected, "backend-shaped target regex rejects " + invalid);
        }
        check(TARGET.equals(WalkIntent.validate(WalkIntent.create("path_refinement",TARGET)).getString("target_scan_id")), "backend-shaped target validates");
        boolean targetType=false; try { WalkIntent.validate(new JSONObject(reconstruction.toString()).put("target_scan_id",42)); } catch (IllegalArgumentException expected) { targetType=true; }
        check(targetType, "target ID remains a string or null");
        boolean unknown=false; try { WalkIntent.validate(new JSONObject(reconstruction.toString()).put("mode","walk")); } catch (IllegalArgumentException expected) { unknown=true; }
        check(unknown, "legacy alias is accepted only at the action boundary");
        boolean extraReference=false; try { WalkIntent.validate(new JSONObject(selectedPath.toString()).put("target_reference",new JSONObject(selection.toString()).put("extra",true))); } catch (IllegalArgumentException expected) { extraReference=true; }
        check(extraReference, "PCF selection key set is exact");
        boolean uppercaseDigest=false; try { WalkIntent.validate(new JSONObject(selectedPath.toString()).put("target_reference",new JSONObject(selection.toString()).put("manifest_sha256",MANIFEST.toUpperCase()))); } catch (IllegalArgumentException expected) { uppercaseDigest=true; }
        check(uppercaseDigest, "PCF manifest digest is lowercase hex");
        boolean reconstructionReference=false; try { WalkIntent.create("reconstruction",TARGET,selection); } catch (IllegalArgumentException expected) { reconstructionReference=true; }
        check(reconstructionReference, "reconstruction rejects PCF selection");
        System.out.println("WalkIntentTest: "+checks+" checks passed");
    }
}

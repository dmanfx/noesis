package org.noesis.roomwalk;

import org.json.JSONObject;

/** Run on Android's actual JSON runtime; a timing test cannot cross focus bindings. */
public final class FocusConfigurationTest {
    private static int checks;
    public static void main(String[] args) throws Exception {
        JSONObject automatic=new JSONObject().put("mode","automatic");
        require(FocusSettings.sameConfiguration(automatic,null),"Legacy automatic test remains automatic");
        JSONObject lock=new JSONObject().put("mode","manual_locked").put("logical_camera_id","0").put("physical_camera_id","5")
                .put("build_fingerprint","test-os").put("ois_requested","OFF").put("eis_requested","OFF")
                .put("capture_width",7680).put("capture_height",4320).put("capture_fps",30).put("focus_distance_diopters",1.25)
                .put("output_routing_policy",FocusSettings.PHYSICAL_OUTPUT_POLICY);
        require(FocusSettings.sameConfiguration(lock,new JSONObject(lock.toString())),"Identical persistent focus configuration matches");
        require(!FocusSettings.sameConfiguration(lock,automatic),"Automatic test cannot qualify manual lock");
        require(!FocusSettings.sameConfiguration(automatic,lock),"Manual test cannot qualify automatic focus");
        for(String key:new String[]{"logical_camera_id","physical_camera_id","build_fingerprint","ois_requested","eis_requested","output_routing_policy"}){
            JSONObject changed=new JSONObject(lock.toString()).put(key,"different");require(!FocusSettings.sameConfiguration(lock,changed),"Changed "+key+" invalidates preflight");
        }
        for(String key:new String[]{"capture_width","capture_height","capture_fps"}){
            JSONObject changed=new JSONObject(lock.toString()).put(key,1);require(!FocusSettings.sameConfiguration(lock,changed),"Changed "+key+" invalidates preflight");
        }
        require(!FocusSettings.sameConfiguration(lock,new JSONObject(lock.toString()).put("focus_distance_diopters",1.3)),"Changed measured focus invalidates preflight");
        JSONObject missing=new JSONObject(lock.toString());missing.remove("focus_distance_diopters");require(!FocusSettings.sameConfiguration(lock,missing),"Missing focus cannot qualify");
        require(!FocusSettings.sameConfiguration(lock,new JSONObject()),"Unknown mode cannot qualify");
        JSONObject legacy=new JSONObject(lock.toString());legacy.remove("output_routing_policy");
        require(!FocusSettings.sameConfiguration(lock,legacy),"A logical-output timing test cannot qualify physical-output recording");
        JSONObject motion=new JSONObject().put("mode","imu").put("calibration_focus_guard",lock);
        require(FocusSettings.matchesMotionGuard(motion,new JSONObject(lock.toString())),"An unchanged focus survives opening the motion viewer");
        require(!FocusSettings.matchesMotionGuard(motion,new JSONObject(lock.toString()).put("focus_distance_diopters",1.3)),"Refocusing inside the motion viewer blocks Record");
        require(!FocusSettings.matchesMotionGuard(motion,automatic),"Unlocking focus inside the motion viewer blocks Record");
        require(!FocusSettings.matchesMotionGuard(new JSONObject().put("mode","imu"),lock),"Missing motion guard fails closed");
        System.out.println("Focus configuration: "+checks+" checks passed");
    }
    private static void require(boolean condition,String message){checks++;if(!condition)throw new AssertionError(message);}
}

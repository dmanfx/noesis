package org.noesis.roomwalk;

import android.content.Context;
import android.hardware.camera2.*;
import android.hardware.camera2.params.OutputConfiguration;
import android.os.Build;
import android.os.SystemClock;
import org.json.JSONObject;
import java.io.IOException;
import java.util.List;

/** Explicit measured focus settings, bound to a camera and the installed OS. */
final class FocusSettings {
    static final String PHYSICAL_OUTPUT_POLICY = "physical_camera_output_v1";
    static final class Lock {
        final String cameraId, physicalId, fingerprint;
        final float distance;
        final long measuredTimestampNs, receivedNs;
        Lock(String cameraId, String physicalId, String fingerprint, float distance, long measuredTimestampNs, long receivedNs) {
            this.cameraId=cameraId; this.physicalId=physicalId; this.fingerprint=fingerprint;
            this.distance=distance; this.measuredTimestampNs=measuredTimestampNs; this.receivedNs=receivedNs;
        }
        JSONObject json() throws Exception {
            return new JSONObject().put("mode","manual_locked").put("logical_camera_id",cameraId)
                    .put("physical_camera_id",physicalId==null?JSONObject.NULL:physicalId)
                    .put("output_routing_policy",physicalId==null?"single_camera_output_v1":PHYSICAL_OUTPUT_POLICY)
                    .put("build_fingerprint",fingerprint).put("focus_distance_diopters",distance)
                    .put("measured_sensor_timestamp_ns",measuredTimestampNs).put("received_elapsed_realtime_ns",receivedNs)
                    .put("maximum_focus_error_diopters",Math.max(0.01f,Math.abs(distance)*0.01f)).put("capture_width",CaptureEngine.WIDTH).put("capture_height",CaptureEngine.HEIGHT).put("capture_fps",CaptureEngine.FPS)
                    .put("ois_requested","OFF").put("eis_requested","OFF").put("source","confirmed_preview_capture_result");
        }
    }
    static Lock read(Context context,String cameraId) throws Exception {
        String raw=context.getSharedPreferences("focus-locks",0).getString(cameraId,null);
        if(raw==null)return null;
        JSONObject j=new JSONObject(raw);
        if(j.optInt("capture_width")!=CaptureEngine.WIDTH||j.optInt("capture_height")!=CaptureEngine.HEIGHT||j.optInt("capture_fps")!=CaptureEngine.FPS
                ||!"OFF".equals(j.optString("ois_requested"))||!"OFF".equals(j.optString("eis_requested")))throw new IOException("Saved focus uses another capture mode. Unlock focus before continuing.");
        return new Lock(j.getString("logical_camera_id"),j.isNull("physical_camera_id")?null:j.getString("physical_camera_id"),
                j.getString("build_fingerprint"),(float)j.getDouble("focus_distance_diopters"),j.getLong("measured_sensor_timestamp_ns"),j.getLong("received_elapsed_realtime_ns"));
    }
    static void save(Context context,Lock lock) throws Exception {
        if(!context.getSharedPreferences("focus-locks",0).edit().putString(lock.cameraId,lock.json().toString()).commit())
            throw new IOException("Could not save focus lock");
    }
    static void clear(Context context,String cameraId) throws IOException {
        if(!context.getSharedPreferences("focus-locks",0).edit().remove(cameraId).commit())throw new IOException("Could not clear saved focus lock");
    }
    static JSONObject metadata(Lock lock) throws Exception {
        return lock==null?new JSONObject().put("mode","automatic"):lock.json();
    }
    static void validate(CameraCharacteristics c,String cameraId,Lock lock) throws IOException {
        if(lock==null)return;
        if(!cameraId.equals(lock.cameraId)||!Build.FINGERPRINT.equals(lock.fingerprint))throw new IOException("Saved focus lock belongs to another camera or OS build. Unlock focus before continuing.");
        Float maximum=c.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE);
        List<CaptureRequest.Key<?>> keys=c.getAvailableCaptureRequestKeys();
        boolean manual=false;int[] modes=c.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES);
        if(modes!=null)for(int mode:modes)if(mode==CaptureRequest.CONTROL_AF_MODE_OFF)manual=true;
        if(!manual||keys==null||!keys.contains(CaptureRequest.LENS_FOCUS_DISTANCE)||!keys.contains(CaptureRequest.CONTROL_AF_MODE)
                ||maximum==null||!Float.isFinite(maximum)||maximum<=0)throw new IOException("This camera does not support manual focus distance");
        if(!Float.isFinite(lock.distance)||lock.distance<0||lock.distance>maximum)throw new IOException("Measured focus distance is outside this camera's range");
        if(c.getPhysicalCameraIds().size()>0&&(lock.physicalId==null||lock.physicalId.isEmpty()))throw new IOException("The logical camera did not report the active physical camera; focus cannot be bound safely");
        if(lock.physicalId!=null&&!c.getPhysicalCameraIds().contains(lock.physicalId))throw new IOException("The saved lens is not part of this camera. Unlock focus and select it again.");
    }
    static void apply(CaptureRequest.Builder request,CameraCharacteristics c,String cameraId,Lock lock) throws Exception {
        validate(c,cameraId,lock);
        if(lock!=null){request.set(CaptureRequest.CONTROL_AF_MODE,CaptureRequest.CONTROL_AF_MODE_OFF);request.set(CaptureRequest.LENS_FOCUS_DISTANCE,lock.distance);}
    }
    /** Focus distance alone does not prevent a logical camera's automatic lens switching. */
    static void bindOutput(OutputConfiguration output,Lock lock) {
        if(lock!=null&&lock.physicalId!=null)output.setPhysicalCameraId(lock.physicalId);
    }
    static String outputCameraId(String logicalId,Lock lock){return lock!=null&&lock.physicalId!=null?lock.physicalId:logicalId;}
    static String outputCameraId(String logicalId,JSONObject focus){
        return focus!=null&&PHYSICAL_OUTPUT_POLICY.equals(focus.optString("output_routing_policy"))?focus.optString("physical_camera_id",logicalId):logicalId;
    }
    static CaptureResult resultForOutput(TotalCaptureResult total,Lock lock) throws IOException {
        return selectOutputMetadata(total,total.getPhysicalCameraResults(),lock==null?null:lock.physicalId);
    }
    static <T> T selectOutputMetadata(T logical,java.util.Map<String,? extends T> physicalResults,String physicalId) throws IOException {
        if(physicalId==null)return logical;
        T physical=physicalResults.get(physicalId);
        if(physical==null)throw new IOException("The camera did not return metadata for the locked physical lens "+physicalId);
        return physical;
    }
    static Lock measured(CameraCharacteristics c,String cameraId,TotalCaptureResult result,long receivedNs) throws Exception {
        if(result==null||SystemClock.elapsedRealtimeNanos()-receivedNs>2_000_000_000L)throw new IOException("Wait for a fresh, focused preview image");
        Float distance=result.get(CaptureResult.LENS_FOCUS_DISTANCE);Long timestamp=result.get(CaptureResult.SENSOR_TIMESTAMP);
        Integer state=result.get(CaptureResult.CONTROL_AF_STATE),lens=result.get(CaptureResult.LENS_STATE);
        if(distance==null||timestamp==null||timestamp<=0||state==null||(state!=CaptureResult.CONTROL_AF_STATE_PASSIVE_FOCUSED&&state!=CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED)
                ||(lens!=null&&lens!=CaptureResult.LENS_STATE_STATIONARY))throw new IOException("Wait until autofocus has settled on the calibration target, then lock focus");
        Lock lock=new Lock(cameraId,result.get(CaptureResult.LOGICAL_MULTI_CAMERA_ACTIVE_PHYSICAL_ID),Build.FINGERPRINT,distance,timestamp,receivedNs);
        validate(c,cameraId,lock);return lock;
    }
    static boolean sameConfiguration(JSONObject expected,JSONObject recorded) {
        String a=expected==null?"automatic":expected.optString("mode"),b=recorded==null?"automatic":recorded.optString("mode");
        if(!a.equals(b))return false;
        if("automatic".equals(a))return true;
        if(!"manual_locked".equals(a)||expected==null||recorded==null)return false;
        for(String key:new String[]{"logical_camera_id","physical_camera_id","build_fingerprint","ois_requested","eis_requested","output_routing_policy"})
            if(!expected.optString(key,"").equals(recorded.optString(key,"")))return false;
        for(String key:new String[]{"capture_width","capture_height","capture_fps"})if(expected.optInt(key,-1)!=recorded.optInt(key,-2))return false;
        double distance=expected.optDouble("focus_distance_diopters",Double.NaN),actual=recorded.optDouble("focus_distance_diopters",Double.NaN);
        return Double.isFinite(distance)&&Double.isFinite(actual)&&distance==actual;
    }
    static boolean matchesMotionGuard(JSONObject request,JSONObject current) {
        if(request==null||!"imu".equals(request.optString("mode")))return false;
        JSONObject guard=request.optJSONObject("calibration_focus_guard");
        return guard!=null&&"manual_locked".equals(guard.optString("mode"))&&sameConfiguration(guard,current);
    }
    static boolean matches(Lock lock,TotalCaptureResult result) {
        if(lock==null)return true;
        try{return matchesResult(lock,resultForOutput(result,lock));}catch(IOException missing){return false;}
    }
    private static boolean matchesResult(Lock lock,CaptureResult result) {
        Float distance=result.get(CaptureResult.LENS_FOCUS_DISTANCE);Integer af=result.get(CaptureResult.CONTROL_AF_MODE),lens=result.get(CaptureResult.LENS_STATE);
        return af!=null&&af==CaptureResult.CONTROL_AF_MODE_OFF&&distance!=null&&Float.isFinite(distance)
                &&Math.abs(distance-lock.distance)<=Math.max(0.01f,Math.abs(lock.distance)*0.01f)
                &&(lens==null||lens==CaptureResult.LENS_STATE_STATIONARY);
    }
}

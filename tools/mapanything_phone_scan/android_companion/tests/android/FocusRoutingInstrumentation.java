package org.noesis.roomwalk;

import android.app.Instrumentation;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.params.OutputConfiguration;
import android.os.Bundle;
import android.view.Surface;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.json.JSONObject;

/** Android output configuration and metadata routing; not a physical-phone 8K test. */
public final class FocusRoutingInstrumentation extends Instrumentation {
    private int checks;
    @Override public void onCreate(Bundle arguments){super.onCreate(arguments);start();}
    @Override public void onStart(){
        Bundle result=new Bundle();SurfaceTexture texture=null;Surface surface=null;
        try{
            texture=new SurfaceTexture(false);texture.setDefaultBufferSize(1920,1080);surface=new Surface(texture);
            FocusSettings.Lock lock=new FocusSettings.Lock("0","5","test-os",3.215434f,1000,2000);
            OutputConfiguration actual=SessionProbe.output(surface,3,lock);
            OutputConfiguration expected=SessionProbe.output(surface,3);expected.setPhysicalCameraId("5");
            require(actual.equals(expected),"Recorder and support probe outputs target the measured physical lens");
            require(!actual.equals(SessionProbe.output(surface,3)),"A pinned output is distinct from an automatic logical output");
            OutputConfiguration preview=new OutputConfiguration(surface),expectedPreview=new OutputConfiguration(surface);
            FocusSettings.bindOutput(preview,lock);expectedPreview.setPhysicalCameraId("5");
            require(preview.equals(expectedPreview),"Preview uses the same explicit physical lens mapping");
            Map<String,String> physical=new HashMap<>();physical.put("5","lens-5-time-1000");physical.put("2","lens-2-time-1010");
            require("lens-5-time-1000".equals(FocusSettings.selectOutputMetadata("logical-lens-2-time-1010",physical,"5")),"Logical auto-switch metadata cannot replace the pinned output's actual sensor evidence");
            require("logical".equals(FocusSettings.selectOutputMetadata("logical",physical,null)),"An unlocked logical stream keeps its own metadata");
            boolean rejected=false;try{FocusSettings.selectOutputMetadata("logical",physical,"6");}catch(IOException expectedFailure){rejected=true;}
            require(rejected,"Missing physical metadata fails closed without a logical substitute");
            JSONObject current=lock.json(),legacy=new JSONObject(current.toString());legacy.remove("output_routing_policy");
            require("5".equals(FocusSettings.outputCameraId("0",lock))&&"5".equals(FocusSettings.outputCameraId("0",current)),"Pinned recording identifies the actual output camera in both manifest and timing preflight");
            require("0".equals(FocusSettings.outputCameraId("0",legacy)),"A legacy logical stream retains its distinct calibration identity");
            require(!FocusSettings.sameConfiguration(current,legacy),"Old logical-output timing proof cannot qualify a new pinned session");
            require(FocusSettings.sameConfiguration(current,new JSONObject(current.toString())),"Matching physical-output timing configuration remains reusable");
            JSONObject motion=new JSONObject().put("mode","imu").put("calibration_focus_guard",current);
            require(FocusSettings.matchesMotionGuard(motion,new JSONObject(current.toString())),"Motion preserves the exact lock used at viewer entry");
            require(!FocusSettings.matchesMotionGuard(motion,new JSONObject(current.toString()).put("focus_distance_diopters",3.4364262)),"Refocus inside the viewer cannot start a motion take");
            require(!FocusSettings.matchesMotionGuard(motion,new JSONObject().put("mode","automatic")),"Unlocking focus cannot start a motion take");
            require(!FocusSettings.matchesMotionGuard(new JSONObject().put("mode","imu"),current),"A missing native motion guard fails closed");
            require(CaptureEngine.mediaStallReason(10_000_000_000L,9_000_000_000L,9_000_000_000L)==null,"Progressing media stays active");
            require(CaptureEngine.mediaStallReason(10_000_000_000L,0,0)==null,"Startup has its separate existing deadline");
            require("camera_stream_stalled_for_5_seconds".equals(CaptureEngine.mediaStallReason(10_000_000_000L,1_000_000_000L,9_000_000_000L)),"A mid-recording camera stall is bounded");
            require("encoder_stream_stalled_for_5_seconds".equals(CaptureEngine.mediaStallReason(10_000_000_000L,9_000_000_000L,1_000_000_000L)),"A mid-recording encoder stall is bounded");
            result.putString("stream","Focus routing instrumentation: "+checks+" checks passed\n");finish(-1,result);
        }catch(Throwable error){result.putString("stream","FAILED: "+android.util.Log.getStackTraceString(error));finish(1,result);}
        finally{if(surface!=null)surface.release();if(texture!=null)texture.release();}
    }
    private void require(boolean passed,String message){if(!passed)throw new AssertionError(message);checks++;}
}

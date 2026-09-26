package org.noesis.roomwalk;

import android.app.Instrumentation;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import org.json.*;
import java.io.File;
import java.lang.reflect.*;
import java.util.concurrent.*;

/** Exercises the actual packaged WebView and native boundary, without a GPU model or 8K encoder. */
public final class WebShellInstrumentation extends Instrumentation {
    private MainActivity activity;private WebShell shell;private int checks;
    @Override public void onCreate(Bundle arguments){super.onCreate(arguments);start();}
    @Override public void onStart(){
        Bundle result=new Bundle();
        try{
            activity=(MainActivity)startActivitySync(new Intent(getTargetContext(),MainActivity.class).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK|Intent.FLAG_ACTIVITY_CLEAR_TASK));
            shell=(WebShell)field("shell");require(shell!=null,"Packaged web shell exists");
            await("typeof window.RoomWalkNative==='object' && !!document.querySelector('#native-configuration')","Native web UI initializes independently of server connectivity");
            await("document.body.classList.contains('roomwalk-android')","Android capture is selected by the packaged application");
            await("document.querySelector('#native-status').textContent!=='Connecting to the native recorder…'","Native status reaches the web presentation");
            require(Boolean.TRUE.equals(script("document.querySelectorAll('[data-workspace-tab]').length===3")),"Capture, Library and Setup are available");
            require(Boolean.TRUE.equals(script("!document.querySelector('#browser-tools').open && !document.querySelector('#capture-intent').closest('details')")),"Walk purposes remain visible outside collapsed browser imports");
            script("document.querySelector('#capture-mode-path').click()");
            await("!document.querySelector('#path-options').classList.contains('hidden')","Body-held path options are selectable in the packaged app");
            require(Boolean.TRUE.equals(script("document.querySelector('#path-options').textContent.includes('own torso')")),"Own-body guidance is packaged");
            screenshot("web-path-mode.png");
            script("document.querySelector('#capture-mode-reconstruction').click();document.querySelector('#tab-setup').click();document.querySelector('#calibration-secondary').open=true");
            await("!document.querySelector('#panel-setup').classList.contains('hidden')","Optional calibration opens inside Setup");
            require(Boolean.TRUE.equals(script("document.querySelector('#board-squares-x')?.value==='10' || document.querySelector('[name=squares_x]')?.value==='10' || document.querySelector('#calibration-panel').textContent.includes('ChArUco')")),"Calibration controls are rendered");
            require(Boolean.TRUE.equals(script("document.querySelector('#calibration-panel').textContent.toLowerCase().includes('board fixed')")),"Fixed-board guidance reaches the packaged application");
            require(Boolean.TRUE.equals(script("!!document.querySelector('#motion-profile-form')")),"The reusable motion-profile check is present");
            require(Boolean.TRUE.equals(script("!document.querySelector('#calibration-settings').open && !document.querySelector('#calibration-profile-details').open && !document.querySelector('#calibration-history').open")),"Extra settings and checks begin collapsed");
            require(CaptureGuide.targetSeconds("camera")==60&&CaptureGuide.targetSeconds("imu")==90,"Short board durations match the native guide");
            require(CaptureGuide.at("imu",0).stationary&&CaptureGuide.at("imu",85).stationary,"Camera–IMU guide starts and ends still");
            require(CaptureGuide.at("camera",60).completed&&!CaptureGuide.at("walk",600).completed,"Guided board takes complete without truncating normal walks");
            script("document.querySelector('[data-calibration-step=noise]').click()");
            await("!document.querySelector('#calibration-guide-action').disabled","Stationary recording is ready without camera calibration");
            script("window.__nativeReplies=0;window.__lastNativeState=null;window.__receiveNative=window.RoomWalkNative.receive;window.RoomWalkNative.receive=(s)=>{window.__nativeReplies++;window.__lastNativeState=s;return window.__receiveNative(s)}");
            script("document.querySelector('#tab-capture').click();document.querySelector('[data-native-action=snapshot]').click()");
            await("window.__nativeReplies>0 && !document.querySelector('#noise-start').disabled","Unchanged refresh replies and restores recording controls");
            script("window.__nativeReplies=0;document.querySelector('[data-native-action=snapshot]').click()");
            await("window.__nativeReplies>0 && !document.querySelector('#noise-start').disabled","Repeated unchanged refresh also acknowledges the command");
            script("window.__nativeReplies=0;document.dispatchEvent(new Event('visibilitychange'))");
            await("window.__nativeReplies>0 && !document.querySelector('#noise-start').disabled","Foreground refresh does not leave capture controls waiting");
            script("document.querySelector('#tab-setup').click();document.querySelector('#calibration-guide-action').click()");
            await("window.__lastNativeState?.imu_preparing===true && window.__lastNativeState?.imu_active===false","Five-second preparation is not counted as sensor recording");
            require(Boolean.TRUE.equals(script("window.__lastNativeState.imu_telemetry.accel_samples===0 && window.__lastNativeState.imu_telemetry.gyro_samples===0")),"Countdown does not invent acquired samples");
            require(Boolean.TRUE.equals(script("document.querySelector('#calibration-run').getBoundingClientRect().top>=0 && document.querySelector('#calibration-run').getBoundingClientRect().bottom<=innerHeight")),"Countdown and sensor telemetry are visible without touching the phone");
            script("document.querySelector('#noise-stop-top').click()");
            await("!window.__lastNativeState?.imu_preparing && !window.__lastNativeState?.imu_active && !document.querySelector('#noise-start').disabled","Cancelling preparation leaves the recorder idle");
            script("document.querySelector('#calibration-guide-action').click()");
            await("window.__lastNativeState?.imu_active===true","Step two starts the actual native stationary recorder after refresh");
            require(Boolean.TRUE.equals(script("document.querySelector('#calibration-guide-action').disabled && !document.querySelector('#noise-stop').disabled")),"Recording remains gated while stationary sensors are active");
            await("/0:00:0[1-9]/.test(window.__lastNativeState?.imu_progress || '')","Native stationary capture reports elapsed progress");
            require(Boolean.TRUE.equals(script("window.__lastNativeState.imu_telemetry.accel_samples>0 && window.__lastNativeState.imu_telemetry.gyro_samples>0")),"The visible telemetry contains native sample counts");
            screenshot("web-stationary-running.png");
            script("document.querySelector('#noise-stop').click()");
            await("window.__lastNativeState?.imu_active===false && window.__lastNativeState?.busy===false && window.__lastNativeState?.artifact?.name?.endsWith('.zip')","Stationary stop retains a packaged recording");
            require(Boolean.TRUE.equals(script("document.querySelector('#tab-setup').getAttribute('aria-selected')==='true' && document.querySelector('#calibration-guide-action').textContent==='Upload recording'")),"Saved calibration stays in its workflow with Upload as the next action");
            script("document.querySelector('#tab-setup').click();document.querySelector('[data-calibration-step=noise]').click()");
            await("!document.querySelector('#calibration-guide-action').disabled","Step two becomes ready again after packaging");
            screenshot("web-calibration.png");
            script("document.querySelector('#tab-library').click()");
            await("!document.querySelector('#panel-library').classList.contains('hidden')","Local and server library share one app");
            screenshot("web-library.png");
            script("document.querySelector('#tab-capture').click()");
            screenshot("web-capture.png");
            Method origin=WebShell.class.getDeclaredMethod("sameOrigin",Uri.class);origin.setAccessible(true);
            String loaded=(String)script("location.origin");
            require(Boolean.TRUE.equals(origin.invoke(shell,Uri.parse(loaded.toUpperCase(java.util.Locale.ROOT).replace("HTTPS:","https:")+"/"))),"Host comparison is case-insensitive");
            require(Boolean.FALSE.equals(origin.invoke(shell,Uri.parse("http://example.invalid/"))),"Cleartext origins are rejected");
            require(Boolean.FALSE.equals(origin.invoke(shell,Uri.parse("https://example.invalid/"))),"Foreign origins are rejected");
            require(Boolean.FALSE.equals(origin.invoke(shell,Uri.parse(loaded.replace("https://","https://user@")+"/"))),"Credential-bearing origins are rejected");
            Method request=MainActivity.class.getDeclaredMethod("calibrationRequest",String.class,JSONObject.class);request.setAccessible(true);
            JSONObject defaults=(JSONObject)request.invoke(activity,"camera",new JSONObject());JSONObject board=defaults.getJSONObject("board");
            require(board.getInt("squares_x")==10&&board.getInt("squares_y")==14&&board.getJSONArray("marker_ids").length()==70,"Default board retains 10×14 squares and 70 markers");
            require(board.getJSONArray("marker_ids").getInt(0)==300&&board.getJSONArray("marker_ids").getInt(69)==369,"Default marker IDs retain the user's target");
            require(Math.abs(board.getDouble("square_length_m")-.018)<1e-12&&Math.abs(board.getDouble("marker_length_m")-.0132)<1e-12&&!board.getBoolean("legacy_pattern"),"Default physical dimensions and legacy mode are unchanged");
            JSONObject focus=new FocusSettings.Lock("0","5","test-build",2.631579f,1000,2000).json();
            JSONObject motionArgs=new JSONObject().put("board",new JSONObject(board.toString())).put("board_geometry_confirmed",true)
                .put("camera_calibration_id","qualified-camera").put("calibration_focus_guard",focus);
            JSONObject motion=(JSONObject)request.invoke(activity,"imu",motionArgs);
            require(FocusSettings.matchesMotionGuard(motion,focus),"Motion request retains the exact native focus guard");
            for(String missing:new String[]{"camera_calibration_id","board_geometry_confirmed","calibration_focus_guard"}){
                JSONObject incomplete=new JSONObject(motionArgs.toString());incomplete.remove(missing);
                boolean rejected=false;try{request.invoke(activity,"imu",incomplete);}catch(InvocationTargetException expected){rejected=true;}
                require(rejected,"Native motion refuses missing "+missing+" before opening the camera");
            }
            board.getJSONArray("marker_ids").put(1,300);
            boolean invalid=false;try{request.invoke(activity,"imu",new JSONObject().put("board",board));}catch(InvocationTargetException expected){invalid=true;}
            require(invalid,"Duplicate board IDs are rejected before capture");
            require(ImuCalibrationRecorder.MAX_DURATION_SECONDS==10800,"Stationary noise protocol is explicitly bounded to three hours");
            script("window.__networkProbe=null;fetch('/api/health',{signal:AbortSignal.timeout(15000)}).then(r=>r.json()).then(x=>window.__networkProbe={ok:true,device:x.device}).catch(e=>window.__networkProbe={ok:false,error:String(e)})");
            await("window.__networkProbe!==null","Real WebView server request terminates");
            System.out.println("WebView network: "+script("window.__networkProbe"));
            String unknown="roomwalk-native://action?request="+Uri.encode(new JSONObject().put("action","notAnAction").toString());
            script("location.href="+JSONObject.quote(unknown));
            await("document.querySelector('#native-detail').textContent.includes('unavailable')","Unknown native actions fail closed");
            script("window.__nativeReplies=0;location.href="+JSONObject.quote(unknown));
            await("window.__nativeReplies>0 && !document.querySelector('#noise-start').disabled","Repeated unchanged command errors still acknowledge without blocking controls");
            require(Boolean.TRUE.equals(script("location.pathname==='/'")),"Native action handling does not navigate away from the app");
            verifySavedCaptureRecovery();
            BundleTools.writeJson(new File(getTargetContext().getExternalFilesDir(null),"web-shell-validation.json"),new JSONObject().put("schema","roomwalk.web_shell_validation.v1").put("checks",checks).put("real_phone_8k_tested",false).put("server_calibration_solved",false));
            result.putString("stream","Web shell instrumentation: "+checks+" checks passed\n");finish(-1,result);
        }catch(Throwable error){result.putString("stream","FAILED: "+android.util.Log.getStackTraceString(error));finish(1,result);}
        finally{if(activity!=null)runOnMainSync(()->activity.finish());}
    }
    private Object field(String name)throws Exception{Field field=MainActivity.class.getDeclaredField(name);field.setAccessible(true);return field.get(activity);}
    private void setField(String name,Object value)throws Exception{Field field=MainActivity.class.getDeclaredField(name);field.setAccessible(true);field.set(activity,value);}
    private void verifySavedCaptureRecovery()throws Exception{
        final File previous=(File)field("captureDir");
        final File fixture=new File(previous.getParentFile(),"calibration-fixture-"+java.util.UUID.randomUUID());
        if(!fixture.mkdirs())throw new AssertionError("Could not create isolated calibration fixture");
        Method request=MainActivity.class.getDeclaredMethod("calibrationRequest",String.class,JSONObject.class);request.setAccessible(true);
        JSONObject intent=(JSONObject)request.invoke(activity,"camera",new JSONObject());
        intent.put("capture_id",fixture.getName()).put("short_test",false);
        BundleTools.writeJson(new File(fixture,"calibration_request.json"),intent);
        Method select=MainActivity.class.getDeclaredMethod("selectSaved",File.class);select.setAccessible(true);
        runOnMainSync(()->{try{select.invoke(activity,fixture);Method refresh=MainActivity.class.getDeclaredMethod("refreshSavedList");refresh.setAccessible(true);refresh.invoke(activity);}catch(Exception error){throw new RuntimeException(error);}});
        require("camera".equals(((JSONObject)field("calibrationRequest")).getString("mode")),"Selecting a retained Camera take restores its calibration context");
        JSONArray saved=(JSONArray)field("savedCaptures");JSONObject row=null;
        for(int i=0;i<saved.length();i++)if(fixture.getName().equals(saved.getJSONObject(i).getString("id")))row=saved.getJSONObject(i);
        require(row!=null&&"camera".equals(row.optString("calibration_mode"))&&!row.optBoolean("short_test"),"Phone recovery inventory identifies full Camera takes");
        runOnMainSync(()->{try{select.invoke(activity,previous);}catch(Exception error){throw new RuntimeException(error);}});
        require(field("calibrationRequest")==null,"Selecting a different non-calibration take clears stale calibration context");
        final java.util.concurrent.atomic.AtomicBoolean busyAtDismiss=new java.util.concurrent.atomic.AtomicBoolean();
        final Object oldResult=field("pendingNativeResult"),oldPairFinished=field("pairFinished"),oldOutcome=field("captureOutcome");
        // Exercise the actual stop producer's ordering without an 8K encoder or
        // a server session. A fake Dialog observes the exact dismissal boundary.
        runOnMainSync(()->{try{
            android.app.Dialog observer=new android.app.Dialog(activity){@Override public void dismiss(){try{busyAtDismiss.set(Boolean.TRUE.equals(field("busy")));}catch(Exception error){throw new RuntimeException(error);}super.dismiss();}};
            setField("captureDialog",observer);setField("active",true);setField("busy",false);setField("pairFinished",false);
            activity.onStopped(new JSONObject().put("stop_reason","manual_focus_not_confirmed").put("partial",true).put("export_ready",true)
                .put("timing",new JSONObject().put("first_matched_sensor_timestamp_ns",1_000_000_000L).put("last_matched_sensor_timestamp_ns",27_900_000_000L)));
        }catch(Exception error){throw new RuntimeException(error);}});
        require(busyAtDismiss.get(),"Native stop remains busy before closing the camera viewer");
        require(Boolean.TRUE.equals(field("busy")),"Native finalization remains busy while waiting for packaging");
        JSONObject outcome=(JSONObject)field("captureOutcome");
        require(outcome.getBoolean("partial")&&Math.abs(outcome.getDouble("duration_seconds")-26.9)<1e-9,"An interrupted native take reports actual acquisition duration");
        require(outcome.getString("message").contains("locked lens/focus")&&previous.getName().equals(outcome.getString("capture_id")),"The early-stop explanation belongs to the exact retained take");
        runOnMainSync(()->{try{setField("busy",false);setField("pendingNativeResult",oldResult);setField("pairFinished",oldPairFinished);setField("captureOutcome",oldOutcome);Method update=MainActivity.class.getDeclaredMethod("updateButtons");update.setAccessible(true);update.invoke(activity);}catch(Exception error){throw new RuntimeException(error);}});
    }
    private Object script(String source)throws Exception{
        CountDownLatch ready=new CountDownLatch(1);String[] result={null};
        runOnMainSync(()->shell.view.evaluateJavascript(source,value->{result[0]=value;ready.countDown();}));
        if(!ready.await(10,TimeUnit.SECONDS))throw new AssertionError("WebView script did not respond");
        return result[0]==null?null:new JSONTokener(result[0]).nextValue();
    }
    private void await(String source,String label)throws Exception{long deadline=SystemClock.elapsedRealtime()+20000;while(SystemClock.elapsedRealtime()<deadline){if(Boolean.TRUE.equals(script(source))){require(true,label);return;}Thread.sleep(100);}throw new AssertionError(label);}
    private void require(boolean passed,String label){if(!passed)throw new AssertionError(label);checks++;}
    private void screenshot(String name)throws Exception{
        CountDownLatch rendered=new CountDownLatch(1);
        runOnMainSync(()->shell.view.postVisualStateCallback(SystemClock.elapsedRealtime(),new android.webkit.WebView.VisualStateCallback(){public void onComplete(long id){shell.view.invalidate();rendered.countDown();}}));
        if(!rendered.await(10,TimeUnit.SECONDS))throw new AssertionError("WebView did not present its visual state");
        waitForIdleSync();Thread.sleep(1000);android.graphics.Bitmap bitmap=getUiAutomation().takeScreenshot();
        if(bitmap==null)throw new AssertionError("Screenshot unavailable");
        try(java.io.FileOutputStream output=new java.io.FileOutputStream(new File(getTargetContext().getExternalFilesDir(null),name))){bitmap.compress(android.graphics.Bitmap.CompressFormat.PNG,100,output);}finally{bitmap.recycle();}
    }
}

package org.noesis.roomwalk;

import android.app.Instrumentation;
import android.content.Intent;
import android.content.res.Configuration;
import android.os.Bundle;
import android.os.SystemClock;
import java.io.File;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import org.json.JSONArray;
import org.json.JSONObject;

/** Real Activity/preview transitions; never opens an encoder or changes 8K gates. */
public final class OrientationInstrumentation extends Instrumentation {
    private MainActivity activity;
    private int checks;
    private final JSONArray observations=new JSONArray();
    @Override public void onCreate(Bundle arguments){super.onCreate(arguments);start();}
    @Override public void onStart(){
        Bundle result=new Bundle();
        try{
            activity=(MainActivity)startActivitySync(new Intent(getTargetContext(),MainActivity.class).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK|Intent.FLAG_ACTIVITY_CLEAR_TASK));
            waitFor(()->state().getInt("orientation")==Configuration.ORIENTATION_PORTRAIT,"Home opens in portrait");
            screenshot("orientation-portrait-home.png");
            JSONObject report=((CaptureEngine)field("engine")).probe();JSONArray cameras=report.getJSONArray("cameras");
            int found=-1;for(int i=0;i<cameras.length();i++)if(cameras.getJSONObject(i).optInt("preview_width")>0&&cameras.getJSONObject(i).optInt("preview_height")>0){found=i;break;}
            require(found>=0,"A real camera preview is available for the orientation smoke");
            final int selection=found;
            runOnMainSync(()->{try{
                Field values=MainActivity.class.getDeclaredField("cameras");values.setAccessible(true);values.set(activity,cameras);
                Object camera=field("camera");Method size=camera.getClass().getDeclaredMethod("setSize",int.class);size.setAccessible(true);size.invoke(camera,cameras.length());
                Method select=camera.getClass().getDeclaredMethod("setSelection",int.class);select.setAccessible(true);select.invoke(camera,selection);
                call("updateButtons");
            }catch(Exception error){throw new RuntimeException(error);}});
            for(int cycle=0;cycle<2;cycle++){
                invoke("openCaptureViewer");
                waitFor(()->{JSONObject s=state();return s.getInt("orientation")==Configuration.ORIENTATION_LANDSCAPE&&s.getBoolean("viewer")&&s.getBoolean("preview_ready");},"Capture opens a live preview in landscape");
                require(!state().getBoolean("destroyed"),"Orientation change preserves the Activity and selected camera");
                if(cycle==0){
                    Object originalViewer=field("captureDialog");
                    CameraPreview controller=(CameraPreview)field("previewController");
                    Field current=CameraPreview.class.getDeclaredField("current");current.setAccessible(true);
                    Object session=current.get(controller);
                    Field control=CameraPreview.class.getDeclaredField("control");control.setAccessible(true);
                    Method fail=CameraPreview.class.getDeclaredMethod("fail",session.getClass(),String.class);fail.setAccessible(true);
                    ((android.os.Handler)control.get(controller)).post(()->{try{fail.invoke(controller,session,"Injected lens/focus interruption");}catch(Exception error){throw new RuntimeException(error);}});
                    waitFor(()->!state().getBoolean("preview_ready"),"Actual preview error clears ready state");
                    android.widget.Button retry=(android.widget.Button)field("previewRetryButton");
                    require(retry.isEnabled()&&retry.getVisibility()==android.view.View.VISIBLE,"Preview error exposes an enabled in-place restart action");
                    screenshot("preview-recovery.png");
                    runOnMainSync(()->retry.performClick());
                    waitFor(()->state().getBoolean("preview_ready"),"Restart reopens a live camera after confirmed release");
                    require(field("captureDialog")==originalViewer,"Recovery does not require leaving the viewer or reloading the app");
                    Object restored=current.get(controller);
                    Field captureField=restored.getClass().getDeclaredField("capture");captureField.setAccessible(true);
                    ((android.os.Handler)control.get(controller)).post(()->{try{((android.hardware.camera2.CameraCaptureSession)captureField.get(restored)).stopRepeating();}catch(Exception error){throw new RuntimeException(error);}});
                    waitFor(()->!state().getBoolean("preview_ready"),"A real mid-preview frame stall stops within the bounded watchdog");
                    require(field("captureDialog")==originalViewer&&retry.isEnabled(),"A stalled preview remains recoverable without an app reload");
                    runOnMainSync(()->retry.performClick());
                    waitFor(()->state().getBoolean("preview_ready"),"Preview can restart after a frame stall");
                }
                observations.put(state());if(cycle==0)screenshot("orientation-landscape-preview.png");invoke("requestCloseCaptureViewer");
                waitFor(()->{JSONObject s=state();return s.getInt("orientation")==Configuration.ORIENTATION_PORTRAIT&&!s.getBoolean("viewer");},"Closing capture returns to portrait");
                observations.put(state());if(cycle==0)screenshot("orientation-portrait-return.png");
            }
            JSONObject evidence=new JSONObject().put("schema","noesis.android.orientation_validation.v1").put("checks",checks).put("phone_video_recorded",false).put("observations",observations);
            BundleTools.writeJson(new File(getTargetContext().getExternalFilesDir(null),"orientation-validation.json"),evidence);
            result.putString("stream","Orientation instrumentation: "+checks+" checks passed\n");finish(-1,result);
        }catch(Throwable error){result.putString("stream","FAILED: "+android.util.Log.getStackTraceString(error));finish(1,result);}
        finally{if(activity!=null)runOnMainSync(()->activity.finish());}
    }
    private void screenshot(String name)throws Exception{
        waitForIdleSync();Thread.sleep(750);android.graphics.Bitmap bitmap=getUiAutomation().takeScreenshot();
        if(bitmap==null)throw new java.io.IOException("Could not capture orientation screenshot");
        try(java.io.FileOutputStream out=new java.io.FileOutputStream(new File(getTargetContext().getExternalFilesDir(null),name))){if(!bitmap.compress(android.graphics.Bitmap.CompressFormat.PNG,100,out))throw new java.io.IOException("Could not save screenshot");}finally{bitmap.recycle();}
    }
    private Object field(String name)throws Exception{Field field=MainActivity.class.getDeclaredField(name);field.setAccessible(true);return field.get(activity);}
    private void call(String name)throws Exception{Method method=MainActivity.class.getDeclaredMethod(name);method.setAccessible(true);method.invoke(activity);}
    private void invoke(String name){runOnMainSync(()->{try{call(name);}catch(Exception error){throw new RuntimeException(error);}});}
    private JSONObject state(){
        JSONObject value=new JSONObject();runOnMainSync(()->{try{value.put("orientation",activity.getResources().getConfiguration().orientation).put("viewer",field("captureDialog")!=null).put("preview_ready",field("previewReady")).put("destroyed",field("destroyed"));}catch(Exception error){throw new RuntimeException(error);}});return value;
    }
    private interface Condition{boolean get()throws Exception;}
    private void waitFor(Condition condition,String label)throws Exception{long deadline=SystemClock.elapsedRealtime()+15000;while(SystemClock.elapsedRealtime()<deadline){if(condition.get()){require(true,label);return;}Thread.sleep(50);}throw new AssertionError(label+": "+state());}
    private void require(boolean passed,String label){if(!passed)throw new AssertionError(label);checks++;}
}

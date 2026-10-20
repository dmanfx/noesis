package org.noesis.roomwalk;

import android.app.Instrumentation;
import android.content.Intent;
import android.os.Bundle;
import android.os.SystemClock;
import org.json.JSONTokener;
import java.lang.reflect.Field;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/** Actual packaged DOM/bridge smoke; fixture snapshots never start acquisition. */
public final class CameraSelectionInstrumentation extends Instrumentation {
    private MainActivity activity;
    private WebShell shell;
    private int checks;

    @Override public void onCreate(Bundle arguments) { super.onCreate(arguments); start(); }
    @Override public void onStart() {
        Bundle result = new Bundle();
        try {
            activity = (MainActivity) startActivitySync(new Intent(getTargetContext(), MainActivity.class)
                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK));
            Field field = MainActivity.class.getDeclaredField("shell"); field.setAccessible(true);
            shell = (WebShell) field.get(activity);
            await("!!document.querySelector('#native-configuration')", "Packaged native UI loads");
            script("window.__cameraSmokeError=null;(async()=>{try{"
                    + "const m=await import(new URL('/static/native_capture.js?checkbox-smoke=1',location.href).href);"
                    + "const root=document.createElement('section');root.id='camera-smoke';root.style='position:fixed;inset:0;z-index:99999;background:var(--bg);color:var(--text);padding:16px;overflow:auto';document.body.append(root);"
                    + "root.innerHTML='<p>Camera checklist test fixture · no recording started</p><div id=\"camera-smoke-capture\"></div>';"
                    + "const platform={__ROOMWALK_ANDROID__:true,navigator:{userAgent:'RoomWalkAndroid/test'},document,location:{origin:'https://fixture.invalid',protocol:'https:',href:'https://fixture.invalid/'}};"
                    + "window.cu=new m.NativeCaptureUI({captureRoot:root.querySelector('#camera-smoke-capture'),platform,onError:e=>window.__cameraSmokeError=String(e)});"
                    + "cu.bridge.navigate=url=>{window.cameraCommand=JSON.parse(new URL(url).searchParams.get('request'))};"
                    + "window.cameraFixture={schema:'roomwalk.native_state.v1',server:'https://fixture.invalid',camera_index:0,cameras:[{id:'0',label:'Test phone',full_walk_available:true}],"
                    + "room_cameras:[{camera_id:'family-room',label:'Family Room',available:true},{camera_id:'kitchen',label:'Kitchen',available:true},{camera_id:'living-room',label:'Living Room',available:true},{camera_id:'offline',label:'Offline camera',available:false}],"
                    + "room_camera_index:-1,extra_room_camera_index:-1,multi_camera_capture:true,supports_camera_ids:true,imu_minutes:1,saved:[],enabled:{configure:true,capture:true},active:false,busy:false};"
                    + "window.cameraReceive=()=>cu.bridge.receive(JSON.parse(JSON.stringify(cameraFixture)));cameraReceive();"
                    + "window.cameraCheck=id=>root.querySelector('input[data-camera-id=\"'+id+'\"]');"
                    + "window.cameraPrimary=()=>root.querySelector('#native-primary-camera');"
                    + "root.querySelector('#native-setup').open=true;window.__cameraSmokeReady=true;"
                    + "}catch(e){window.__cameraSmokeError=String(e)}})()");
            await("window.__cameraSmokeReady===true", "Fixture uses the actual packaged camera checklist");
            check("cu.configuration().room_camera_index===-1 && cu.configuration().extra_room_camera_index===-1", "Phone-only starts with no hidden camera selection");
            check("cameraCheck('offline').disabled", "Unavailable camera is visibly unavailable");
            script("cameraCheck('kitchen').click();cameraCheck('family-room').click()");
            check("cu.configuration().room_camera_index===1 && cu.configuration().extra_room_camera_index===0", "Two checked cameras keep the first selected primary");
            check("cameraPrimary().options.length===2 && cameraCheck('living-room').disabled", "Primary contains only checked cameras and max two is enforced");
            script("cameraPrimary().value='family-room';cameraPrimary().dispatchEvent(new Event('change',{bubbles:true}))");
            check("cu.configuration().room_camera_index===0 && cu.configuration().extra_room_camera_index===1 && cameraCheck('kitchen').checked", "Changing primary does not change the recording set");
            script("cameraCheck('family-room').click()");
            check("cu.configuration().room_camera_index===1 && cu.configuration().extra_room_camera_index===-1", "Unchecking primary selects the remaining camera");
            script("cameraCheck('kitchen').click()");
            check("cu.configuration().room_camera_index===-1 && cu.configuration().extra_room_camera_index===-1", "Unchecking all genuinely returns to phone-only");
            script("cameraCheck('family-room').click();cameraCheck('kitchen').click();document.querySelector('#camera-smoke #native-configuration').requestSubmit()");
            check("cameraCommand.action==='configure' && cameraCommand.args.room_camera_index===0 && cameraCommand.args.extra_room_camera_index===1", "Apply sends the exact existing native configuration contract");
            script("cameraFixture.room_camera_index=1;cameraFixture.extra_room_camera_index=0;cameraReceive()");
            check("cu.configDirty===true", "A mismatched primary acknowledgement does not discard edits");
            script("cameraFixture.room_camera_index=0;cameraFixture.extra_room_camera_index=1;cameraReceive()");
            check("cu.configDirty===false && cameraCheck('family-room').checked && cameraCheck('kitchen').checked", "Exact acknowledgement restores the retained checked pair");
            screenshot();
            script("cameraFixture.active=true;cameraReceive()");
            check("[...document.querySelectorAll('#camera-smoke input[data-camera-id]')].every(x=>x.disabled) && cameraPrimary().disabled", "Recording locks both checklist and primary reference");
            script("cameraFixture.active=false;cameraFixture.room_cameras[0].available=false;cameraReceive()");
            check("cameraCheck('family-room').checked && !cameraCheck('family-room').disabled", "A selected offline camera remains removable rather than silently disappearing");
            script("cameraFixture.room_cameras[0].available=true;cameraReceive()");
            check("!window.__cameraSmokeError", "No client errors during the changed path");
            result.putString("stream", "Camera checkbox UI: " + checks + " checks passed\n"); finish(-1, result);
        } catch (Throwable error) {
            result.putString("stream", "FAILED: " + error + "\n" + android.util.Log.getStackTraceString(error)); finish(1, result);
        } finally {
            if (activity != null) runOnMainSync(() -> activity.finish());
        }
    }

    private Object script(String source) throws Exception {
        CountDownLatch ready = new CountDownLatch(1); String[] value = {null};
        runOnMainSync(() -> shell.view.evaluateJavascript(source, result -> { value[0] = result; ready.countDown(); }));
        if (!ready.await(10, TimeUnit.SECONDS)) throw new AssertionError("WebView script did not respond");
        return value[0] == null ? null : new JSONTokener(value[0]).nextValue();
    }
    private void check(String source, String label) throws Exception {
        if (!Boolean.TRUE.equals(script(source))) throw new AssertionError(label + ": " + script("window.__cameraSmokeError"));
        checks++;
    }
    private void await(String source, String label) throws Exception {
        long deadline = SystemClock.elapsedRealtime() + 20000;
        while (SystemClock.elapsedRealtime() < deadline) {
            if (Boolean.TRUE.equals(script(source))) { checks++; return; }
            Thread.sleep(100);
        }
        throw new AssertionError(label + ": " + script("window.__cameraSmokeError"));
    }
    private void screenshot() throws Exception {
        script("document.querySelector('#camera-smoke #native-camera-picker').scrollIntoView({block:'center'})");
        CountDownLatch ready = new CountDownLatch(1);
        runOnMainSync(() -> shell.view.postVisualStateCallback(SystemClock.elapsedRealtime(),
                new android.webkit.WebView.VisualStateCallback() {
                    @Override public void onComplete(long id) { shell.view.invalidate(); ready.countDown(); }
                }));
        if (!ready.await(10, TimeUnit.SECONDS)) throw new AssertionError("Checklist did not render");
        waitForIdleSync();
        // Export the tested WebView, not unrelated emulator System UI overlays.
        // This is component evidence, not a physical-device screenshot.
        android.graphics.Bitmap[] rendered = {null};
        runOnMainSync(() -> {
            rendered[0] = android.graphics.Bitmap.createBitmap(shell.view.getWidth(), shell.view.getHeight(), android.graphics.Bitmap.Config.ARGB_8888);
            shell.view.draw(new android.graphics.Canvas(rendered[0]));
        });
        android.graphics.Bitmap bitmap = rendered[0];
        try (java.io.FileOutputStream output = new java.io.FileOutputStream(
                new java.io.File(getTargetContext().getExternalFilesDir(null), "camera-checkboxes-webview.png"))) {
            bitmap.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, output);
        } finally { bitmap.recycle(); }
    }
}

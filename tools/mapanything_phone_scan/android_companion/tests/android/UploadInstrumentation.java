package org.noesis.roomwalk;

import android.app.Instrumentation;
import android.os.Bundle;
import android.os.Debug;
import android.os.SystemClock;
import java.io.*;
import java.net.URL;
import java.security.MessageDigest;
import java.util.Random;
import javax.net.ssl.HttpsURLConnection;
import org.json.JSONObject;

/** Bounded synthetic transport fixtures; never imports a real product scan. */
final class UploadInstrumentation {
    static int benchmark(Instrumentation test,Bundle args)throws Exception {
        int mib=Integer.parseInt(args.getString("uploadMiB","128"));
        if(mib<1||mib>128)throw new IOException("Benchmark bound is1..128MiB");
        File file=new File(test.getTargetContext().getCacheDir(),"upload-benchmark.bin");
        byte[] buffer=new byte[262144];new Random(711).nextBytes(buffer);MessageDigest digest=MessageDigest.getInstance("SHA-256");
        try(OutputStream out=new BufferedOutputStream(new FileOutputStream(file),262144)){
            for(int i=0;i<mib*4;i++){out.write(buffer);digest.update(buffer);}
        }
        String expected=hex(digest.digest());long total=file.length();
        java.lang.reflect.Method factory=BundleTools.class.getDeclaredMethod("connection",android.content.Context.class,URL.class);factory.setAccessible(true);
        HttpsURLConnection connection=(HttpsURLConnection)factory.invoke(null,test.getTargetContext(),BundleTools.endpoint(args.getString("uploadBenchServer"),"/sink"));
        String tuning=args.getString("socketTuning","none");JSONObject socketOptions=new JSONObject();
        javax.net.ssl.SSLSocketFactory delegate=connection.getSSLSocketFactory();
        connection.setSSLSocketFactory(new javax.net.ssl.SSLSocketFactory(){
            public String[] getDefaultCipherSuites(){return delegate.getDefaultCipherSuites();}
            public String[] getSupportedCipherSuites(){return delegate.getSupportedCipherSuites();}
            public java.net.Socket createSocket(java.net.Socket raw,String host,int port,boolean close)throws IOException{
                try{socketOptions.put("send_buffer_before",raw.getSendBufferSize()).put("tcp_nodelay_before",raw.getTcpNoDelay());
                    if(tuning.equals("buffer")||tuning.equals("both"))raw.setSendBufferSize(1024*1024);
                    if(tuning.equals("nodelay")||tuning.equals("both"))raw.setTcpNoDelay(true);
                    socketOptions.put("send_buffer_after",raw.getSendBufferSize()).put("tcp_nodelay_after",raw.getTcpNoDelay());
                }catch(org.json.JSONException error){throw new IOException(error);}
                return delegate.createSocket(raw,host,port,close);
            }
            public java.net.Socket createSocket()throws IOException{return delegate.createSocket();}
            public java.net.Socket createSocket(String h,int p)throws IOException{return delegate.createSocket(h,p);}
            public java.net.Socket createSocket(String h,int p,java.net.InetAddress l,int q)throws IOException{return delegate.createSocket(h,p,l,q);}
            public java.net.Socket createSocket(java.net.InetAddress h,int p)throws IOException{return delegate.createSocket(h,p);}
            public java.net.Socket createSocket(java.net.InetAddress h,int p,java.net.InetAddress l,int q)throws IOException{return delegate.createSocket(h,p,l,q);}
        });
        long started=SystemClock.elapsedRealtime(),cpu=Debug.threadCpuTimeNanos(),firstByteMs=0;int progressEvents=0;JSONObject reply;
        try{
            connection.setRequestMethod("POST");connection.setDoOutput(true);connection.setRequestProperty("Content-Type","application/octet-stream");connection.setFixedLengthStreamingMode(total);
            // The pre-change production upload loop, including its buffering and
            // callback cadence, so server receive measurements have a baseline.
            try(InputStream in=new BufferedInputStream(new FileInputStream(file),262144);OutputStream out=new BufferedOutputStream(connection.getOutputStream(),262144)){
                long sent=0,last=0;int n;
                while((n=in.read(buffer))!=-1){out.write(buffer,0,n);sent+=n;long now=System.currentTimeMillis();if(firstByteMs==0)firstByteMs=SystemClock.elapsedRealtime()-started;if(now-last>500){progressEvents++;last=now;}}
            }
            int code=connection.getResponseCode();ByteArrayOutputStream response=new ByteArrayOutputStream();
            try(InputStream in=connection.getInputStream()){int n;while((n=in.read(buffer))!=-1){if(response.size()+n>65536)throw new IOException("Fixture response bound exceeded");response.write(buffer,0,n);}}
            if(code<200||code>=300)throw new AssertionError("Fixture upload failed: "+code);reply=new JSONObject(response.toString("UTF-8"));
        }finally{connection.disconnect();}
        long elapsed=SystemClock.elapsedRealtime()-started;JSONObject report=new JSONObject().put("synthetic_fixture",true).put("kind","baseline_256k_https_stream").put("socket_tuning",tuning).put("socket_options",socketOptions).put("bytes",total).put("elapsed_ms",elapsed).put("first_write_ms",firstByteMs).put("client_mib_per_second",total*1000.0/Math.max(1,elapsed)/1048576).put("thread_cpu_ms",(Debug.threadCpuTimeNanos()-cpu)/1e6).put("progress_events",progressEvents).put("expected_sha256",expected).put("server",reply);
        BundleTools.writeJson(new File(test.getTargetContext().getExternalFilesDir(null),"upload-benchmark-"+tuning+".json"),report);System.out.println(report.toString(2));
        if(reply.getLong("bytes_received")!=total||!expected.equals(reply.getString("sha256")))throw new AssertionError("Fixture bytes or digest differ");
        if(!file.delete())throw new IOException("Could not remove synthetic benchmark payload");
        return 3;
    }
    static int service(Instrumentation test,Bundle args)throws Exception{
        android.content.Context context=test.getTargetContext();String server=args.getString("uploadServiceServer");
        android.app.Activity activity=visible(test);
        File directory=new File(context.getExternalFilesDir(null),"captures/upload-test-"+java.util.UUID.randomUUID());
        if(!directory.mkdirs())throw new IOException("Could not create test capture directory");
        File archive=syntheticArchive(directory,60);String hash=digest(archive);long size=archive.length(),modified=archive.lastModified();
        JSONObject report=new JSONObject().put("synthetic_fixture",true).put("phone_video_recorded",false).put("archive_sha256",hash).put("archive_bytes",size)
                .put("notification_permission_granted",context.checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS)==android.content.pm.PackageManager.PERMISSION_GRANTED);
        test.runOnMainSync(()->{try{UploadService.start(context,archive,server,false);}catch(Exception error){throw new RuntimeException(error);}});
        require(UploadService.snapshot(context).optBoolean("active"),"Start synchronously reserves the upload");
        JSONObject first=awaitState(context,"uploading",10000);require(first.getString("archive_path").equals(archive.getPath()),"State binds the selected archive");
        JSONObject copy=UploadService.snapshot(context);copy.put("state","tampered");require(!"tampered".equals(UploadService.snapshot(context).getString("state")),"Snapshots are deep copies");
        boolean duplicate=false;try{UploadService.start(context,archive,server,false);}catch(IOException expected){duplicate=true;}require(duplicate,"A second upload is not queued");
        test.runOnMainSync(activity::finish);test.waitForIdleSync();
        shell(test,"input keyevent 223");Thread.sleep(1200);
        require(!context.getSystemService(android.os.PowerManager.class).isInteractive(),"Screen is off during transfer");
        long before=UploadService.snapshot(context).getLong("sent_bytes");Thread.sleep(2500);JSONObject background=UploadService.snapshot(context);
        require(background.getLong("sent_bytes")>before,"Bytes advance after activity destruction with screen off");
        report.put("screen_off_progress",background).put("screen_off_before_bytes",before).put("activity_destroyed",activity.isDestroyed());
        require(activity.isDestroyed(),"The activity was actually destroyed");
        require(background.getLong("sent_bytes")<size,"Cancel is exercised while the body is still being written");
        BundleTools.writeJson(new File(context.getExternalFilesDir(null),"upload-service-test.json"),report);
        final long[] cancelMs={0};test.runOnMainSync(()->{long started=SystemClock.elapsedRealtime();try{UploadService.cancel(context);}catch(Exception error){throw new RuntimeException(error);}cancelMs[0]=SystemClock.elapsedRealtime()-started;});
        JSONObject cancelled=awaitState(context,"cancelled",8000);require(cancelMs[0]<250,"Cancel never waits for network I/O on UI");
        require(archive.length()==size&&archive.lastModified()==modified&&hash.equals(digest(archive)),"Cancellation preserves the exact ZIP");
        require(!cancelled.has("receipt")&&cancelled.getBoolean("can_retry"),"Cancelled state cannot assert a receipt and permits explicit retry");
        JSONObject durable=BundleTools.readJson(new File(directory,UploadService.STATE_FILE));require("cancelled".equals(durable.getString("state")),"Cancellation is durable");
        report.put("cancel_ui_ms",cancelMs[0]).put("cancelled",cancelled);
        BundleTools.writeJson(new File(context.getExternalFilesDir(null),"upload-service-test.json"),report);
        long idleDeadline=SystemClock.elapsedRealtime()+40000;
        while(BundleTools.health(context,server).optBoolean("fixture_active")){if(SystemClock.elapsedRealtime()>idleDeadline)throw new AssertionError("Slow receiver did not release its cancelled request");Thread.sleep(500);}
        shell(test,"input keyevent 224");Thread.sleep(200);visible(test);
        if(!archive.setLastModified(modified+2000))throw new IOException("Could not test modified archive guard");
        boolean changedRejected=false;try{UploadService.retryLast(context);}catch(IOException expected){changedRejected=true;}
        require(changedRejected,"Retry rejects a changed ZIP");if(!archive.setLastModified(modified))throw new IOException("Could not restore fixture mtime");
        // Give the main-thread service teardown its bounded completion turn.
        Thread.sleep(300);test.runOnMainSync(()->{try{UploadService.retryLast(context);}catch(Exception error){throw new RuntimeException(error);}});
        shell(test,"input keyevent 3");shell(test,"input keyevent 223");
        JSONObject complete=awaitState(context,"complete",60000);require(!context.getSystemService(android.os.PowerManager.class).isInteractive(),"Retry also completes with screen off");
        JSONObject receipt=complete.getJSONObject("receipt");require(receipt.getLong("bytes_received")==size&&hash.equals(receipt.getString("sha256")),"Real receiver confirms exact bytes and digest");
        require(!complete.getBoolean("active")&&!complete.getBoolean("can_retry"),"Complete is terminal");
        require(new File(directory,"upload_receipt.json").isFile(),"Verified receipt is retained on disk");
        require(hash.equals(digest(archive)),"Retry sends the retained archive unchanged");
        report.put("complete",complete);BundleTools.writeJson(new File(context.getExternalFilesDir(null),"upload-service-test.json"),report);
        System.out.println(report.toString(2));shell(test,"input keyevent 224");
        return 18;
    }
    static int storedService(Instrumentation test,Bundle args)throws Exception{
        android.content.Context context=test.getTargetContext();android.app.Activity activity=visible(test);
        boolean existing=args.getString("uploadArchive")!=null;File dir,archive;
        if(existing){archive=new File(args.getString("uploadArchive"));dir=archive.getParentFile();}
        else{dir=new File(context.getExternalFilesDir(null),"captures/upload-stored-"+java.util.UUID.randomUUID());if(!dir.mkdirs())throw new IOException("fixture directory");archive=syntheticArchive(dir,16);}
        final File selected=archive;String hash=digest(archive);long started=SystemClock.elapsedRealtime();
        test.runOnMainSync(()->{try{UploadService.start(context,selected,args.getString("uploadStoredServer"),false);}catch(Exception error){throw new RuntimeException(error);}});
        if(!existing){awaitState(context,"uploading",10000);Thread.sleep(1500);
        test.runOnMainSync(()->{try{java.lang.reflect.Field field=MainActivity.class.getDeclaredField("transferPanel");field.setAccessible(true);android.view.View panel=(android.view.View)field.get(activity);panel.requestRectangleOnScreen(new android.graphics.Rect(0,0,panel.getWidth(),panel.getHeight()),true);}catch(Exception error){throw new RuntimeException(error);}});
        screenshot(test,"upload-progress-portrait.png");}
        JSONObject complete=awaitState(context,"complete",20000);JSONObject receipt=complete.getJSONObject("receipt"),stored=receipt.getJSONObject("upload_receipt");
        require("pending".equals(receipt.getString("validation_status")),"202 means transferred; validation stays pending");
        require("noesis.phone_capture.upload_receipt.v1".equals(stored.getString("schema"))&&"stored".equals(stored.getString("status")),"Receipt is the explicit durable storage contract");
        require(stored.getLong("size_bytes")==archive.length()&&hash.equals(stored.getString("sha256")),"Stored receipt matches streamed ZIP size and digest");
        require(dir.getName().equals(stored.getString("capture_id")),"Stored receipt confirms the manifest capture identity");
        JSONObject durable=BundleTools.readJson(new File(dir,"upload_receipt.json"));require("pending".equals(durable.getString("validation_status")),"Durable receipt preserves pending import status");
        Thread.sleep(400);require(!UploadService.isActive(context),"Foreground transfer is complete before server validation");
        int checks=6;
        if(args.getString("assertNotification")!=null){
            android.service.notification.StatusBarNotification[] notifications=context.getSystemService(android.app.NotificationManager.class).getActiveNotifications();android.app.Notification notification=null;
            for(android.service.notification.StatusBarNotification row:notifications)if(row.getId()==41)notification=row.getNotification();
            require(notification!=null&&"Transferred to RoomWalk".equals(notification.extras.getString(android.app.Notification.EXTRA_TITLE)),"Stored completion notification remains after the service stops");
            require((notification.flags&android.app.Notification.FLAG_ONGOING_EVENT)==0&&notification.actions==null,"Completion notification is dismissible and has no stale Cancel button");
            context.startService(new android.content.Intent(context,UploadService.class).setAction("org.noesis.roomwalk.CANCEL_UPLOAD").putExtra("id",complete.getString("id")));
            Thread.sleep(500);boolean serviceRunning=false;
            for(android.app.ActivityManager.RunningServiceInfo row:context.getSystemService(android.app.ActivityManager.class).getRunningServices(30))if(UploadService.class.getName().equals(row.service.getClassName()))serviceRunning=true;
            require(!serviceRunning,"A late Cancel notification intent cannot leave an idle service running");checks+=3;
        }
        JSONObject report=new JSONObject().put("synthetic_fixture",true).put("phone_video_recorded",false).put("elapsed_ms",SystemClock.elapsedRealtime()-started).put("expected_sha256",hash).put("state",complete).put("checks",checks).put("production_endpoint",existing);
        BundleTools.writeJson(new File(context.getExternalFilesDir(null),(existing?"upload-actual-endpoint-test.json":"upload-stored-service-test.json")),report);System.out.println(report.toString(2));return checks;
    }
    private static void screenshot(Instrumentation test,String name)throws Exception{
        test.waitForIdleSync();Thread.sleep(200);android.graphics.Bitmap bitmap=test.getUiAutomation().takeScreenshot();if(bitmap==null)throw new IOException("Screenshot unavailable");
        try(FileOutputStream out=new FileOutputStream(new File(test.getTargetContext().getExternalFilesDir(null),name))){if(!bitmap.compress(android.graphics.Bitmap.CompressFormat.PNG,100,out))throw new IOException("Screenshot save failed");}finally{bitmap.recycle();}
    }
    static int recovery(Instrumentation test,Bundle args)throws Exception{
        android.content.Context context=test.getTargetContext();
        if(args.getString("uploadRecoveryServer")!=null){
            visible(test);File dir=new File(context.getExternalFilesDir(null),"captures/upload-recovery-"+java.util.UUID.randomUUID());if(!dir.mkdirs())throw new IOException("fixture directory");
            File archive=syntheticArchive(dir,60);
            test.runOnMainSync(()->{try{UploadService.start(context,archive,args.getString("uploadRecoveryServer"),false);}catch(Exception error){throw new RuntimeException(error);}});
            awaitState(context,"uploading",10000);Thread.sleep(3000);
            BundleTools.writeJson(new File(context.getExternalFilesDir(null),"upload-recovery-ready.json"),UploadService.snapshot(context));
            Thread.sleep(60000);throw new AssertionError("Host must force-stop the fixture after ready marker");
        }
        JSONObject state=UploadService.snapshot(context);require("interrupted".equals(state.optString("state"))&&!state.optBoolean("active")&&state.optBoolean("can_retry"),"Restart exposes interrupted state without automatic transfer: "+state);
        require(!state.has("receipt"),"Interrupted upload has no asserted receipt");
        JSONObject stored=BundleTools.readJson(new File(state.getString("capture_path"),UploadService.STATE_FILE));require("interrupted".equals(stored.getString("state")),"Recovery state is durable");
        Thread.sleep(1500);require(!UploadService.isActive(context),"Recovery does not resubmit");
        BundleTools.writeJson(new File(context.getExternalFilesDir(null),"upload-recovery-test.json"),state);System.out.println(state.toString(2));return 4;
    }
    static int receipts(Instrumentation test)throws Exception{
        File dir=new File(test.getTargetContext().getCacheDir(),"receipt-test");dir.mkdirs();File archive=syntheticArchive(dir,1);String hash=digest(archive);
        JSONObject stored=new JSONObject().put("schema","noesis.phone_capture.upload_receipt.v1").put("status","stored").put("size_bytes",archive.length()).put("sha256",hash).put("capture_id",dir.getName()).put("companion_session_id",JSONObject.NULL).put("companion_camera_id",JSONObject.NULL);
        JSONObject receipt=new JSONObject().put("id","20000101-000000-00000000").put("upload_receipt",stored);
        BundleTools.validateStoredReceipt(receipt,archive,hash,null);int checks=1;
        for(String key:new String[]{"schema","status","size_bytes","sha256","capture_id"}){JSONObject broken=new JSONObject(receipt.toString());broken.getJSONObject("upload_receipt").put(key,key.equals("size_bytes")?3:"wrong");boolean failed=false;try{BundleTools.validateStoredReceipt(broken,archive,hash,null);}catch(IOException expected){failed=true;}require(failed,"Reject wrong stored receipt "+key);checks++;}
        JSONObject paired=new JSONObject().put("session_id","companion-20000101-000000-00000000").put("camera_id","fixture-camera").put("phone_capture_id",dir.getName());
        stored.put("companion_session_id",paired.getString("session_id")).put("companion_camera_id",paired.getString("camera_id"));BundleTools.validateStoredReceipt(receipt,archive,hash,paired);checks++;
        stored.put("companion_camera_id","wrong");boolean failed=false;try{BundleTools.validateStoredReceipt(receipt,archive,hash,paired);}catch(IOException expected){failed=true;}require(failed,"Reject mismatched paired stored receipt");checks++;
        archive.delete();dir.delete();return checks;
    }
    private static android.app.Activity visible(Instrumentation test)throws Exception{
        shell(test,"input keyevent 224");shell(test,"wm dismiss-keyguard");
        android.app.Activity activity=test.startActivitySync(new android.content.Intent(test.getTargetContext(),MainActivity.class).addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK));test.waitForIdleSync();Thread.sleep(300);return activity;
    }
    private static File syntheticArchive(File dir,int mib)throws Exception{
        File archive=new File(dir,"roomwalk-"+dir.getName()+".zip");byte[] bytes=new byte[262144];new Random(915).nextBytes(bytes);
        try(java.util.zip.ZipOutputStream out=new java.util.zip.ZipOutputStream(new BufferedOutputStream(new FileOutputStream(archive),262144))){out.setLevel(0);
            out.putNextEntry(new java.util.zip.ZipEntry("capture_manifest.json"));out.write(new JSONObject().put("schema","noesis.phone_capture.v1").put("capture_id",dir.getName()).toString().getBytes(java.nio.charset.StandardCharsets.UTF_8));out.closeEntry();
            out.putNextEntry(new java.util.zip.ZipEntry("synthetic.bin"));for(int i=0;i<mib*4;i++)out.write(bytes);out.closeEntry();}
        return archive;
    }
    private static JSONObject awaitState(android.content.Context context,String wanted,long timeout)throws Exception{
        long deadline=SystemClock.elapsedRealtime()+timeout;JSONObject last;
        do{last=UploadService.snapshot(context);if(wanted.equals(last.optString("state")))return last;
            if(!last.optBoolean("active"))throw new AssertionError("Expected "+wanted+", got "+last);Thread.sleep(150);
        }while(SystemClock.elapsedRealtime()<deadline);throw new AssertionError("Timed out waiting for "+wanted+": "+last);
    }
    private static void shell(Instrumentation test,String command)throws Exception{try(android.os.ParcelFileDescriptor fd=test.getUiAutomation().executeShellCommand(command);InputStream in=new android.os.ParcelFileDescriptor.AutoCloseInputStream(fd)){byte[] bytes=new byte[4096];while(in.read(bytes)>=0){}}}
    private static String digest(File file)throws Exception{MessageDigest hash=MessageDigest.getInstance("SHA-256");try(InputStream in=new FileInputStream(file)){byte[] bytes=new byte[262144];int n;while((n=in.read(bytes))!=-1)hash.update(bytes,0,n);}return hex(hash.digest());}
    private static void require(boolean condition,String message){if(!condition)throw new AssertionError(message);}
    private static String hex(byte[] bytes){StringBuilder out=new StringBuilder();for(byte value:bytes)out.append(String.format(java.util.Locale.ROOT,"%02x",value&255));return out.toString();}
}

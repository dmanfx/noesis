package org.noesis.roomwalk;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.IBinder;
import android.os.PowerManager;
import android.os.SystemClock;
import android.util.AtomicFile;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.UUID;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import org.json.JSONObject;

/** A single explicitly requested, retained-archive upload. Never starts capture or resubmits automatically. */
public final class UploadService extends Service {
    static final String SCHEMA="noesis.android.upload_state.v1";
    static final String STATE_FILE="upload_state.json",LATEST_FILE="upload_latest.json";
    private static final String CHANNEL="roomwalk-upload",CANCEL="org.noesis.roomwalk.CANCEL_UPLOAD";
    private static final int NOTIFICATION=41;
    private static final long MAX_DURATION_MS=90*60*1000L,STALL_MS=180000;
    private static final Object LOCK=new Object();
    private static JSONObject cached;
    private static UploadService running;
    private final ThreadPoolExecutor worker=new ThreadPoolExecutor(0,1,30,TimeUnit.SECONDS,new SynchronousQueue<>(),
            task->new Thread(task,"roomwalk-upload"));
    private final ScheduledExecutorService watchdog=Executors.newSingleThreadScheduledExecutor(task->new Thread(task,"roomwalk-upload-watchdog"));
    private volatile Job job;
    private PowerManager.WakeLock wakeLock;

    public static void start(Context context,File archive,String server,boolean imu)throws Exception{
        Context app=context.getApplicationContext();File file=archive.getCanonicalFile();
        String origin=BundleTools.endpoint(server.trim(),"/").toString().replaceAll("/+$","");
        if(!file.isFile()||!file.getName().endsWith(".zip")||file.length()==0)throw new IOException("Choose a completed saved ZIP to upload");
        JSONObject state;
        synchronized(LOCK){
            initialize(app);if(cached.optBoolean("active"))throw new IOException("An upload is already active");
            if(running!=null&&running.job!=null)throw new IOException("The previous upload is still closing; retry shortly");
            long now=System.currentTimeMillis();
            state=new JSONObject().put("schema",SCHEMA).put("id",UUID.randomUUID().toString()).put("state","queued").put("active",true)
                    .put("archive_path",file.getPath()).put("capture_path",file.getParent()).put("archive_name",file.getName()).put("server_origin",origin).put("imu",imu)
                    .put("archive_modified_ms",file.lastModified()).put("sent_bytes",0).put("total_bytes",file.length()).put("bytes_per_second",0).put("elapsed_ms",0)
                    .put("started_unix_ms",now).put("updated_unix_ms",now).put("error","").put("can_retry",false);
            cached=copy(state);
        }
        try{app.startForegroundService(new Intent(app,UploadService.class).putExtra("id",state.getString("id")));}
        catch(Exception error){synchronized(LOCK){terminal(state,"failed","Android could not start the upload: "+message(error));cached=copy(state);persist(app,state);}throw error;}
    }
    public static void retryLast(Context context)throws Exception{
        JSONObject previous=snapshot(context);
        if(!previous.optBoolean("can_retry"))throw new IOException("There is no interrupted upload to retry");
        File archive=new File(previous.getString("archive_path"));
        if(!archive.isFile()||archive.length()!=previous.getLong("total_bytes")||archive.lastModified()!=previous.getLong("archive_modified_ms"))
            throw new IOException("The saved ZIP changed or is missing. Select the intended capture before uploading it again.");
        start(context,archive,previous.getString("server_origin"),previous.getBoolean("imu"));
    }
    public static void cancel(Context context)throws Exception{
        Job active;
        synchronized(LOCK){initialize(context.getApplicationContext());if(!cached.optBoolean("active"))return;
            cached.put("state","cancelling").put("updated_unix_ms",System.currentTimeMillis());active=running==null?null:running.job;}
        if(active!=null)active.transfer.cancel("Upload cancelled. RoomWalk may already have received the archive; the phone copy is retained.");
    }
    public static JSONObject snapshot(Context context)throws Exception{synchronized(LOCK){initialize(context.getApplicationContext());return copy(cached);}}
    public static boolean isActive(Context context)throws Exception{return snapshot(context).optBoolean("active");}
    private static void initialize(Context app)throws Exception{
        if(cached!=null)return;
        AtomicFile latest=new AtomicFile(new File(app.getFilesDir(),LATEST_FILE));
        if(latest.getBaseFile().isFile()||new File(latest.getBaseFile()+".bak").isFile()){
            byte[] bytes=latest.readFully();if(bytes.length>512*1024)throw new IOException("Saved upload status exceeds its bound");
            cached=new JSONObject(new String(bytes,StandardCharsets.UTF_8));
            if(!SCHEMA.equals(cached.optString("schema")))throw new IOException("Saved upload status has an unsupported schema");
            if(cached.optBoolean("active")){terminal(cached,"interrupted","Android stopped the previous upload before a receipt was saved. Retry the retained ZIP when ready.");persist(app,cached);}
        }else cached=new JSONObject().put("schema",SCHEMA).put("state","idle").put("active",false).put("can_retry",false);
    }
    private static JSONObject copy(JSONObject object)throws Exception{return new JSONObject(object.toString());}
    private static void atomic(File file,JSONObject value)throws Exception{
        byte[] bytes=value.toString(2).getBytes(StandardCharsets.UTF_8);if(bytes.length>512*1024)throw new IOException("Upload metadata exceeds its bound");
        AtomicFile atomic=new AtomicFile(file);FileOutputStream out=null;
        try{out=atomic.startWrite();out.write(bytes);atomic.finishWrite(out);}catch(Exception error){if(out!=null)atomic.failWrite(out);throw error;}
    }
    private static void persist(Context app,JSONObject state)throws Exception{
        atomic(new File(state.getString("capture_path"),STATE_FILE),state);
        // The latest pointer embeds its bounded snapshot: UI polling needs no
        // repeated filesystem reads, even when another capture is selected.
        JSONObject latest=copy(state).put("state_path",new File(state.getString("capture_path"),STATE_FILE).getPath());
        atomic(new File(app.getFilesDir(),LATEST_FILE),latest);
    }
    private static void terminal(JSONObject state,String phase,String error)throws Exception{
        state.put("state",phase).put("active",false).put("can_retry",!"complete".equals(phase)).put("updated_unix_ms",System.currentTimeMillis()).put("error",error);
        if(!"complete".equals(phase))state.remove("receipt");
    }
    private static String message(Throwable error){String text=error.getMessage();if(text==null||text.isEmpty())text=error.toString();return text.substring(0,Math.min(text.length(),1600));}

    @Override public void onCreate(){super.onCreate();synchronized(LOCK){running=this;}
        NotificationManager notifications=getSystemService(NotificationManager.class);
        notifications.createNotificationChannel(new NotificationChannel(CHANNEL,"Recording uploads",NotificationManager.IMPORTANCE_LOW));
    }
    @Override public int onStartCommand(Intent intent,int flags,int startId){
        try{
            if(intent!=null&&CANCEL.equals(intent.getAction())){if(job==null)stopSelf();else if(job.state.getString("id").equals(intent.getStringExtra("id")))cancel(this);return START_NOT_STICKY;}
            JSONObject state=snapshot(this);
            if(intent==null||!state.optBoolean("active")||!state.getString("id").equals(intent.getStringExtra("id"))){if(job==null)stopSelf();return START_NOT_STICKY;}
            if(job!=null)return START_NOT_STICKY;
            Job next=new Job(state,startId);job=next;
            startForeground(NOTIFICATION,notification(state),ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
            wakeLock=getSystemService(PowerManager.class).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK,"RoomWalk:upload");
            wakeLock.setReferenceCounted(false);wakeLock.acquire(MAX_DURATION_MS+10000);
            watchdog.scheduleWithFixedDelay(()->watch(next),1,1,TimeUnit.SECONDS);
            worker.execute(()->transfer(next));
        }catch(Exception error){Job failed=job;if(failed!=null){failed.transfer.cancel(message(error));finish(failed,"failed",message(error),null);}stopSelf();}
        return START_NOT_STICKY;
    }
    private void watch(Job current){
        if(job!=current)return;long now=SystemClock.elapsedRealtime();
        if(now-current.started>=MAX_DURATION_MS)current.transfer.cancel("Upload exceeded its 90-minute limit. The retained ZIP can be retried.");
        else if(now-current.transfer.lastProgress>=STALL_MS)current.transfer.cancel("Upload made no progress for three minutes. Check Wi-Fi and retry the retained ZIP.");
    }
    private void transfer(Job current){
        current.transfer.ownThread();
        try{
            synchronized(LOCK){if("cancelling".equals(cached.optString("state")))current.transfer.cancel("Upload cancelled. The phone copy is retained.");}
            current.transfer.check();publish(current,true);phase(current,"connecting");
            File archive=new File(current.state.getString("archive_path"));
            if(archive.length()!=current.state.getLong("total_bytes")||archive.lastModified()!=current.state.getLong("archive_modified_ms"))throw new IOException("The retained ZIP changed before upload");
            String server=current.state.getString("server_origin");
            JSONObject paired=current.state.getBoolean("imu")?null:BundleTools.pairing(archive);
            if(paired!=null){
                if(!BundleTools.endpoint(server,"/").toURI().equals(BundleTools.endpoint(paired.getString("server_origin"),"/").toURI()))throw new IOException("Upload this paired recording to its original RoomWalk server: "+paired.getString("server_origin"));
                phase(current,"finalizing");JSONObject saved=PairedCapture.finishSaved(getApplicationContext(),archive.getParentFile());
                for(String key:new String[]{"session_id","camera_id","phone_capture_id"})if(!paired.getString(key).equals(saved.getString(key)))throw new IOException("Saved paired identity differs from the retained ZIP");
                current.transfer.check();phase(current,"connecting");
            }
            JSONObject health=BundleTools.health(getApplicationContext(),server);
            if(!"ok".equals(health.optString("status")))throw new IOException("RoomWalk is not ready for upload");
            long free=health.optLong("storage_free_bytes",-1);
            if(free>=0&&free<archive.length()*2+512L*1024*1024)throw new IOException("RoomWalk needs more free storage for this archive and its extracted files. Free server space before retrying.");
            current.transfer.check();
            BundleTools.Progress progress=(sent,total)->{
                try{current.state.put("sent_bytes",sent).put("total_bytes",total);publish(current,false);}
                catch(Exception error){throw new IllegalStateException("Could not preserve upload progress",error);}
            };
            JSONObject receipt=current.state.getBoolean("imu")?BundleTools.uploadImu(getApplicationContext(),server,archive,progress,current.transfer):
                    BundleTools.upload(getApplicationContext(),server,archive,progress,"Android "+archive.getParentFile().getName(),current.transfer);
            current.transfer.check();finish(current,"complete","",receipt);
        }catch(Exception error){String reason=current.transfer.reason();
            String phase=current.interrupted?"interrupted":reason!=null&&reason.startsWith("Upload cancelled")?"cancelled":"failed";
            finish(current,phase,reason==null?message(error):reason,null);
        }finally{
            Thread.interrupted();
            new android.os.Handler(getMainLooper()).post(()->{if(job==current){job=null;releaseWakeLock();stopForeground("cancelled".equals(current.state.optString("state"))?STOP_FOREGROUND_REMOVE:STOP_FOREGROUND_DETACH);stopSelf();}});
        }
    }
    private void phase(Job current,String phase){
        try{current.transfer.check();current.state.put("state",phase);publish(current,true);}
        catch(Exception error){throw new IllegalStateException(error);}
    }
    private void publish(Job current,boolean force)throws Exception{
        long elapsed=SystemClock.elapsedRealtime()-current.started;
        current.state.put("elapsed_ms",elapsed).put("updated_unix_ms",System.currentTimeMillis());
        if(current.state.optLong("sent_bytes")>0)current.state.put("bytes_per_second",current.state.optLong("sent_bytes")*1000.0/Math.max(1,elapsed));
        synchronized(LOCK){
            if(current.transfer.cancelled()&&current.state.optBoolean("active"))current.state.put("state","cancelling");
        }
        if(force||elapsed-current.lastSave>=2000){persist(getApplicationContext(),current.state);current.lastSave=elapsed;}
        synchronized(LOCK){cached=copy(current.state);}
        if(force||elapsed-current.lastNotification>=1000){getSystemService(NotificationManager.class).notify(NOTIFICATION,notification(current.state));current.lastNotification=elapsed;}
    }
    private void finish(Job current,String phase,String error,JSONObject receipt){
        try{
            if(receipt!=null)atomic(new File(current.state.getString("capture_path"),current.state.getBoolean("imu")?"imu_upload_receipt.json":"upload_receipt.json"),receipt);
            terminal(current.state,phase,error);if(receipt!=null)current.state.put("receipt",receipt);
            publish(current,true);
        }catch(Exception failure){try{terminal(current.state,"failed","Could not preserve the upload receipt/status: "+message(failure));synchronized(LOCK){cached=copy(current.state);}persist(getApplicationContext(),current.state);}catch(Exception ignored){android.util.Log.e("RoomWalk","Could not save upload failure",ignored);}}
    }
    private Notification notification(JSONObject state){
        String phase=state.optString("state"),text,title="Uploading recording";boolean active=state.optBoolean("active");
        long total=state.optLong("total_bytes"),sent=state.optLong("sent_bytes");
        if("complete".equals(phase)){JSONObject receipt=state.optJSONObject("receipt");boolean pending=receipt!=null&&"pending".equals(receipt.optString("validation_status"));title=pending?"Transferred to RoomWalk":"Upload complete";text=pending?"Validation continues on RoomWalk":"The saved recording is on RoomWalk";}
        else if(!active){title="cancelled".equals(phase)?"Upload cancelled":"Upload needs attention";text=state.optString("error");}
        else if("uploading".equals(phase))text=String.format(Locale.US,"%d%% • %.0f Mbps",(long)(100.0*sent/Math.max(1,total)),state.optDouble("bytes_per_second")*8/1000000);
        else if("validating".equals(phase))text="Transfer sent; RoomWalk is checking the archive";
        else if("cancelling".equals(phase))text="Cancelling upload";
        else if("finalizing".equals(phase))text="Finalizing paired recording";
        else text="Connecting to RoomWalk";
        PendingIntent open=PendingIntent.getActivity(this,0,new Intent(this,MainActivity.class).addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP|Intent.FLAG_ACTIVITY_SINGLE_TOP),PendingIntent.FLAG_UPDATE_CURRENT|PendingIntent.FLAG_IMMUTABLE);
        PendingIntent cancel=PendingIntent.getService(this,1,new Intent(this,UploadService.class).setAction(CANCEL).putExtra("id",state.optString("id")),PendingIntent.FLAG_UPDATE_CURRENT|PendingIntent.FLAG_IMMUTABLE);
        int icon=active?android.R.drawable.stat_sys_upload:"complete".equals(phase)?android.R.drawable.stat_sys_upload_done:android.R.drawable.stat_notify_error;
        Notification.Builder builder=new Notification.Builder(this,CHANNEL).setSmallIcon(icon).setContentTitle(title)
                .setContentText(text).setSubText(state.optString("archive_name")).setContentIntent(open).setOnlyAlertOnce(true).setOngoing(active).setAutoCancel(!active);
        if(active)builder.setProgress(1000,(int)(1000.0*sent/Math.max(1,total)),!"uploading".equals(phase));
        if(active&&!"cancelling".equals(phase))builder.addAction(new Notification.Action.Builder(null,"Cancel",cancel).build());
        return builder.build();
    }
    private synchronized void releaseWakeLock(){if(wakeLock!=null&&wakeLock.isHeld())wakeLock.release();}
    @Override public void onTimeout(int startId,int foregroundServiceType){
        Job current=job;if(current!=null){current.interrupted=true;current.transfer.cancel("Android's background upload time limit was reached. Retry the retained ZIP when ready.");}
        releaseWakeLock();stopForeground(STOP_FOREGROUND_REMOVE);stopSelf();
    }
    @Override public void onDestroy(){
        Job current=job;if(current!=null){current.interrupted=true;current.transfer.cancel("Android stopped the upload service. Retry the retained ZIP when ready.");}
        watchdog.shutdownNow();worker.shutdown();releaseWakeLock();synchronized(LOCK){if(running==this)running=null;}super.onDestroy();
    }
    @Override public IBinder onBind(Intent intent){return null;}
    private final class Job{
        final JSONObject state;final int startId;final long started=SystemClock.elapsedRealtime();final UploadTransfer transfer;
        long lastSave,lastNotification;volatile boolean interrupted;
        Job(JSONObject state,int startId){this.state=state;this.startId=startId;transfer=new UploadTransfer(phase->phase(this,phase));}
    }
}

package org.noesis.roomwalk;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.File;
import java.io.IOException;
import java.util.UUID;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

/** Owns a bounded companion-service lease, independently of native frame callbacks. */
public final class PairedCapture {
    public static final String FILE="companion_capture.json",SCHEMA="noesis.phone_capture.companion_ref.v1";
    public interface Listener {
        void onPairReady(File directory,JSONObject reference);
        void onPairState(String state,JSONObject reference);
        void onPairStopped(File directory,JSONObject reference);
        void onPairError(String reason,JSONObject reference);
    }
    public interface Transport { Object request(String server,String path,JSONObject body,int timeoutMs) throws Exception; }
    private static final int MAX_PROBES=96,MAX_FAILURES=32;
    private final Transport transport;
    private final Listener listener;
    private final Handler main=new Handler(Looper.getMainLooper());
    private final ScheduledExecutorService worker=Executors.newSingleThreadScheduledExecutor();
    private volatile Session current;
    private volatile boolean closing;
    public PairedCapture(Context context,Listener listener){this((server,path,body,timeout)->path.equals("/api/health")?BundleTools.health(context,server):BundleTools.requestJsonValue(context,server,path,body,timeout),listener);}
    /** Explicit transport dependency also permits isolated Android instrumentation without camera gate changes. */
    public PairedCapture(Transport transport,Listener listener){this.transport=transport;this.listener=listener;}
    public boolean isActive(){return current!=null;}
    public static String phoneStopReason(JSONObject result){return result.optBoolean("partial")?result.optString("stop_reason","phone_interrupted"):"user";}
    public synchronized void start(File directory,String server,String cameraId){
        if(closing||current!=null)throw new IllegalStateException("A paired capture is already active");
        Session session=new Session(directory,server,cameraId);current=session;
        worker.execute(()->startSession(session));
    }
    public synchronized void stop(String reason){
        Session session=current;if(session==null)return;
        if(!session.stopRequested)session.stopReason=bounded(reason);
        session.stopRequested=true;
        worker.execute(()->finishSession(session));
    }
    public void close(){closing=true;Session session=current;if(session==null)worker.shutdown();else stop("app_closed");}
    private void startSession(Session s){
        try{
            BundleTools.endpoint(s.server,"/");
            if(!s.directory.isDirectory())throw new IOException("Phone capture directory is unavailable");
            if(new File(s.directory,FILE).exists())throw new IOException("Paired capture metadata already exists");
            save(s);state(s,"starting");
            if(!s.stopRequested&&!"ok".equals(object(transport.request(s.server,"/api/health",null,BundleTools.SETUP_TIMEOUT_MS)).optString("status")))throw new IOException("RoomWalk is not ready for paired capture");
            for(int i=0;i<3&&!s.stopRequested;i++)clockProbe(s,transport,"start",3000);
            if(s.stopRequested){finishSession(s);return;}
            JSONObject body=startBody(s);
            Exception first=null;
            for(int attempt=0;attempt<2;attempt++){
                try{adopt(s,object(transport.request(s.server,"/api/companion-captures",body,20000)));first=null;break;}
                catch(Exception error){first=error;if(error instanceof BundleTools.HttpFailure)break;}
            }
            if(first!=null)throw first;
            save(s);
            long deadline=SystemClock.elapsedRealtime()+15000;
            while("starting".equals(s.serverStatus)&&!s.stopRequested&&SystemClock.elapsedRealtime()<deadline){
                Thread.sleep(500);adopt(s,object(transport.request(s.server,path(s),null,5000)));save(s);
            }
            if(s.stopRequested){finishSession(s);return;}
            if(!healthy(s))throw new IOException("The room camera and tracking stream did not become ready: "+s.serverStatus);
            s.status="recording";save(s);
            s.heartbeat=worker.scheduleWithFixedDelay(()->heartbeat(s),10,10,TimeUnit.SECONDS);
            JSONObject reference=snapshot(s);main.post(()->listener.onPairReady(s.directory,reference));
        }catch(Exception error){
            s.error=bounded(error.getMessage());s.stopReason="phone_start_failed";s.stopRequested=true;
            error(s,"Paired capture could not start: "+s.error);finishSession(s);
        }
    }
    private void heartbeat(Session s){
        if(current!=s||s.stopRequested)return;
        try{
            JSONArray latest=new JSONArray();
            try{latest.put(clockProbe(s,transport,"heartbeat",3000));}
            catch(Exception error){rememberFailure(s,"clock",error);}
            JSONObject body=new JSONObject().put("phone_capture_id",s.captureId).put("client_request_id",s.requestId+"-heartbeat-"+(++s.heartbeats)).put("clock_probes",latest);
            adopt(s,object(transport.request(s.server,path(s)+"/heartbeat",body,5000)));
            if(!healthy(s))throw new IOException("The paired room camera or tracking stream stopped");
            save(s);state(s,"recording");
        }catch(Exception error){s.error=bounded(error.getMessage());s.stopReason="companion_stream_lost";s.stopRequested=true;error(s,s.error);finishSession(s);}
    }
    private void finishSession(Session s){
        if(current!=s||s.finished)return;
        if(s.heartbeat!=null)s.heartbeat.cancel(false);
        s.status="finalizing";state(s,"finalizing");
        try{finishRemote(s,transport);}
        catch(Exception error){s.needsRecovery=true;s.status="unconfirmed";s.error=bounded(error.getMessage());rememberFailure(s,"stop",error);}
        try{save(s);}catch(Exception failure){s.error="Could not save paired reference: "+bounded(failure.getMessage());error(s,s.error);}
        s.finished=true;current=null;JSONObject reference=snapshot(s);
        main.post(()->listener.onPairStopped(s.directory,reference));
        if(closing)worker.shutdown();
    }
    /** Retry only the exact retained session, without opening a new server recording. */
    public static JSONObject finishSaved(Context context,File directory) throws Exception {
        return finishSaved((server,path,body,timeout)->BundleTools.requestJsonValue(context,server,path,body,timeout),directory);
    }
    public static JSONObject finishSaved(Transport transport,File directory) throws Exception {
        Session s=readSession(directory);
        try{finishRemote(s,transport);save(s);return snapshot(s);}
        catch(Exception error){s.needsRecovery=true;s.status="unconfirmed";s.error=bounded(error.getMessage());save(s);throw error;}
    }
    private static void finishRemote(Session s,Transport transport) throws Exception {
        if(s.sessionId==null&&!s.startSubmitted){s.status=s.error==null?"cancelled":"failed";s.needsRecovery=false;return;}
        if(s.sessionId==null){
            Object value=transport.request(s.server,"/api/companion-captures",null,5000);
            if(!(value instanceof JSONArray))throw new IOException("Invalid companion session inventory");
            JSONArray rows=(JSONArray)value;JSONObject match=null;
            for(int i=0;i<rows.length();i++){
                JSONObject row=rows.optJSONObject(i);if(row==null)continue;
                if(s.captureId.equals(row.optString("phone_capture_id"))&&s.cameraId.equals(row.optString("camera_id"))){if(match!=null)throw new IOException("Multiple static sessions claim this phone capture");match=row;}
            }
            if(match==null){
                // A cancelled start before the POST owns no server resources.
                if(!s.startSubmitted){s.status="cancelled";s.needsRecovery=false;return;}
                throw new IOException("Static start could not be confirmed. The server lease will retain and stop any started session; retry Finalize paired capture.");
            }
            adopt(s,match);
        }
        if(!terminal(s.serverStatus)){
            JSONArray probes=new JSONArray();
            try{probes.put(clockProbe(s,transport,"stop",3000));}catch(Exception error){rememberFailure(s,"clock",error);}
            JSONObject body=new JSONObject().put("phone_capture_id",s.captureId).put("reason",s.stopReason).put("clock_probes",probes);
            adopt(s,object(transport.request(s.server,path(s)+"/stop",body,10000)));
            long deadline=SystemClock.elapsedRealtime()+15000;
            while(!terminal(s.serverStatus)&&SystemClock.elapsedRealtime()<deadline){
                Thread.sleep(750);adopt(s,object(transport.request(s.server,path(s),null,Math.min(5000,(int)Math.max(1,deadline-SystemClock.elapsedRealtime())))));
            }
        }
        if(!terminal(s.serverStatus))throw new IOException("Static finalization is pending; retry Finalize paired capture. Phone files are retained.");
        s.needsRecovery=false;s.status=s.serverStatus;
    }
    private static JSONObject startBody(Session s)throws Exception{
        s.startSubmitted=true;save(s);
        return new JSONObject().put("camera_id",s.cameraId).put("phone_capture_id",s.captureId).put("client_request_id",s.requestId).put("clock_probes",s.probes);
    }
    private static JSONObject clockProbe(Session s,Transport transport,String name,int timeout)throws Exception{
        if(s.probes.length()>=MAX_PROBES)throw new IOException("Clock probe bound reached");
        long sent=SystemClock.elapsedRealtimeNanos(),epoch=System.currentTimeMillis();
        JSONObject reply=object(transport.request(s.server,"/api/companion-captures/clock",null,timeout));
        long received=SystemClock.elapsedRealtimeNanos(),receivedEpoch=System.currentTimeMillis();
        JSONObject probe=new JSONObject().put("name",name).put("index",s.probes.length()).put("request_id",s.requestId+"-clock-"+s.probes.length())
                .put("client_clock","android.elapsedRealtimeNanos").put("client_send_elapsed_realtime_ns",Long.toString(sent)).put("client_receive_elapsed_realtime_ns",Long.toString(received))
                .put("client_send_monotonic_ms",sent/1e6).put("client_receive_monotonic_ms",received/1e6).put("client_send_epoch_ms",epoch).put("client_receive_epoch_ms",receivedEpoch)
                .put("round_trip_ms",(received-sent)/1e6).put("synchronization_verified",false)
                .put("timestamp_provenance","HTTP send/receive observations only; no phone/static acquisition synchronization asserted");
        for(String key:new String[]{"server_received_unix_ns","server_received_monotonic_ns","server_sent_unix_ns","server_sent_monotonic_ns"}){
            String raw=reply.getString(key);if(!raw.matches("[0-9]{1,19}")||Long.parseLong(raw)<=0)throw new IOException("Invalid server clock timestamp");probe.put(key,raw);
        }
        s.probes.put(probe);return probe;
    }
    private static void adopt(Session s,JSONObject value)throws Exception{
        String id=value.optString("session_id"),camera=value.optString("camera_id"),phone=value.optString("phone_capture_id");
        if(!id.matches("companion-[0-9]{8}-[0-9]{6}-[a-f0-9]{8}")||!s.cameraId.equals(camera)||!s.captureId.equals(phone)
                ||s.sessionId!=null&&!s.sessionId.equals(id))throw new IOException("Companion response identity does not match this phone capture");
        s.sessionId=id;s.serverStatus=value.optString("status","unknown");s.videoStatus=value.optString("video_status","unknown");s.trackingStatus=value.optString("tracking_status","unknown");
        s.provenance=value.optJSONObject("provenance");s.artifacts=value.optJSONObject("artifact_urls");
        if(value.has("error")&&!value.isNull("error"))s.error=bounded(value.optString("error"));
    }
    private static boolean healthy(Session s){return "recording".equals(s.serverStatus)&&healthyComponent(s.videoStatus)&&healthyComponent(s.trackingStatus);}
    private static boolean healthyComponent(String status){return "recording".equals(status)||"ready".equals(status)||"active".equals(status)||"ok".equals(status)||"healthy".equals(status);}
    private static boolean terminal(String status){return "stopped".equals(status)||"failed".equals(status)||"cancelled".equals(status)||"expired".equals(status)||"interrupted".equals(status)||"partial".equals(status);}
    private static String path(Session s)throws IOException{if(s.sessionId==null)throw new IOException("Static session identity is unavailable");return "/api/companion-captures/"+s.sessionId;}
    private static JSONObject object(Object value)throws IOException{if(!(value instanceof JSONObject))throw new IOException("Expected companion JSON object");return (JSONObject)value;}
    private static void rememberFailure(Session s,String phase,Exception error){try{if(s.failures.length()<MAX_FAILURES)s.failures.put(new JSONObject().put("phase",phase).put("reason",bounded(error.getMessage())).put("client_elapsed_realtime_ns",Long.toString(SystemClock.elapsedRealtimeNanos())));else s.failuresDropped++;}catch(Exception ignored){}}
    private static JSONObject snapshot(Session s){try{return new JSONObject().put("schema",SCHEMA).put("session_id",s.sessionId==null?JSONObject.NULL:s.sessionId).put("camera_id",s.cameraId).put("phone_capture_id",s.captureId)
                .put("client_request_id",s.requestId).put("server_origin",s.server).put("status",s.status).put("server_status",s.serverStatus).put("video_status",s.videoStatus).put("tracking_status",s.trackingStatus)
                .put("start_submitted",s.startSubmitted).put("needs_recovery",s.needsRecovery).put("stop_reason",s.stopReason).put("error",s.error==null?JSONObject.NULL:s.error)
                .put("clock_probes",new JSONArray(s.probes.toString())).put("clock_probe_failures",new JSONArray(s.failures.toString())).put("clock_probe_failure_dropped_count",s.failuresDropped)
                .put("provenance",s.provenance==null?new JSONObject():s.provenance).put("artifact_urls",s.artifacts==null?new JSONObject():s.artifacts)
                .put("synchronization_verified",false).put("timestamp_provenance","Native phone and server acquisition evidence remain in their original clocks; HTTP clock probes do not prove hardware synchronization");
        }catch(Exception error){throw new IllegalStateException(error);}}
    private static void save(Session s)throws Exception{
        JSONObject ref=snapshot(s);if(ref.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8).length>240000)throw new IOException("Paired capture metadata exceeds its bound");
        File pending=new File(s.directory,FILE+".partial");BundleTools.writeJson(pending,ref);
        if(!pending.renameTo(new File(s.directory,FILE)))throw new IOException("Could not preserve paired capture metadata");
    }
    private static Session readSession(File dir)throws Exception{
        JSONObject ref=BundleTools.readJson(new File(dir,FILE));if(!SCHEMA.equals(ref.optString("schema"))||!dir.getName().equals(ref.optString("phone_capture_id")))throw new IOException("Invalid saved paired capture identity");
        Session s=new Session(dir,ref.getString("server_origin"),ref.getString("camera_id"));s.requestId=ref.getString("client_request_id");s.sessionId=ref.isNull("session_id")?null:ref.optString("session_id",null);
        if(s.sessionId!=null&&!s.sessionId.matches("companion-[0-9]{8}-[0-9]{6}-[a-f0-9]{8}"))throw new IOException("Invalid saved static session ID");
        s.status=ref.optString("status");s.serverStatus=ref.optString("server_status");s.videoStatus=ref.optString("video_status");s.trackingStatus=ref.optString("tracking_status");
        s.stopReason=ref.optString("stop_reason","user");s.error=ref.isNull("error")?null:ref.optString("error",null);s.failuresDropped=ref.optInt("clock_probe_failure_dropped_count");
        s.startSubmitted=ref.optBoolean("start_submitted");s.probes=ref.optJSONArray("clock_probes");if(s.probes==null)s.probes=new JSONArray();s.failures=ref.optJSONArray("clock_probe_failures");if(s.failures==null)s.failures=new JSONArray();
        s.provenance=ref.optJSONObject("provenance");s.artifacts=ref.optJSONObject("artifact_urls");return s;
    }
    private void state(Session s,String state){JSONObject ref=snapshot(s);main.post(()->listener.onPairState(state,ref));}
    private void error(Session s,String reason){JSONObject ref=snapshot(s);main.post(()->listener.onPairError(bounded(reason),ref));}
    private static String bounded(String text){return text==null?"unknown":text.substring(0,Math.min(text.length(),400));}
    private static final class Session{
        final File directory;final String server,cameraId,captureId;String requestId=UUID.randomUUID().toString(),sessionId;
        String status="starting",serverStatus="unknown",videoStatus="unknown",trackingStatus="unknown",error;
        volatile String stopReason="user";volatile boolean stopRequested;boolean startSubmitted,needsRecovery,finished;int heartbeats,failuresDropped;
        JSONArray probes=new JSONArray(),failures=new JSONArray();JSONObject provenance,artifacts;ScheduledFuture<?> heartbeat;
        Session(File directory,String server,String cameraId){this.directory=directory;this.server=server;this.cameraId=cameraId;captureId=directory.getName();}
    }
}

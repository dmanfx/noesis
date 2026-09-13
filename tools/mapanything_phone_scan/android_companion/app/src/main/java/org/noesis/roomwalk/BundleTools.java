package org.noesis.roomwalk;

import android.content.Context;
import android.os.Build;
import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;
import android.os.SystemClock;
import java.util.zip.ZipFile;
import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.security.KeyStore;
import java.security.cert.CertificateFactory;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import javax.net.ssl.*;

/** Streaming local packaging and explicit HTTPS upload; never deletes recordings. */
public final class BundleTools {
    public static final int SETUP_TIMEOUT_MS=3000;
    private BundleTools() {}
    public interface Progress { void update(long current, long total); }
    public static JSONObject readJson(File f) throws Exception {
        if (f.length() > 512 * 1024) throw new IOException("Metadata too large");
        try (InputStream in = new FileInputStream(f)) { return new JSONObject(readText(in, 512 * 1024)); }
    }
    public static void writeJson(File f, JSONObject j) throws Exception {
        try (FileOutputStream out = new FileOutputStream(f)) { out.write(j.toString(2).getBytes(StandardCharsets.UTF_8)); out.getFD().sync(); }
    }
    public static void writeCaptureManifest(Context context, File dir, JSONObject result) throws Exception {
        if(new File(dir,PairedCapture.FILE).isFile()&&new File(dir,"roomwalk-"+dir.getName()+".zip").isFile())return;
        JSONObject c=result.getJSONObject("camera"), files=result.getJSONObject("files");
        android.content.SharedPreferences preferences=context.getSharedPreferences("native-device",0);
        String deviceId=preferences.getString("id",null);
        if(deviceId==null){deviceId="android-install:"+java.util.UUID.randomUUID();preferences.edit().putString("id",deviceId).apply();}
        JSONObject video=new JSONObject().put("path",files.getString("video")).put("timestamp_unit","ns");
        if(files.has("timestamps") && result.getJSONObject("timing").optBoolean("exact_frame_association_verified"))
            video.put("frame_timestamps_path",files.getString("timestamps"));
        JSONObject camera=new JSONObject().put("id",c.getString("id"))
            .put("intrinsics_source","missing").put("intrinsics",JSONObject.NULL)
            .put("distortion_model","unknown").put("distortion",new JSONArray())
            .put("resolution_px",new JSONArray().put(c.getInt("width")).put(c.getInt("height")))
            .put("orientation_deg",0).put("stabilization","unknown");
        JSONObject imu=new JSONObject().put("accel_path",files.getString("accelerometer")).put("gyro_path",files.getString("gyroscope"))
            .put("timestamp_unit","ns").put("axes","x,y,z").put("accel_unit","m/s^2").put("gyro_unit","rad/s")
            .put("noise",new JSONObject()).put("reference_frame","android_device_x_right_y_up_z_out_of_screen");
        JSONObject clocks=new JSONObject().put("camera_domain","REALTIME".equals(c.optString("timestamp_source"))?"android.elapsedRealtimeNanos":"camera_timestamp_source_unknown")
            .put("imu_domain","android.elapsedRealtimeNanos").put("timestamp_source","unverified").put("imu_to_camera_offset_ns",JSONObject.NULL);
        JSONObject manifest=new JSONObject().put("schema","noesis.phone_capture.v1").put("capture_id",dir.getName())
            .put("device",new JSONObject().put("id",deviceId).put("model",Build.MANUFACTURER+" "+Build.MODEL).put("android_api_level",Build.VERSION.SDK_INT))
            .put("video",video).put("camera",camera).put("imu",imu).put("clocks",clocks)
            .put("extrinsics",new JSONObject().put("T_imu_camera",JSONObject.NULL))
            .put("android_capture",new JSONObject().put("schema","noesis.phone_capture.android.v1")
                .put("capture_result_path","capture_result.json").put("encoder_pts_path",files.getString("encoder_pts"))
                .put("camera_results_path",files.getString("camera_results")))
            .put("admission",new JSONObject().put("metric_vio_allowed",false).put("reason","Camera/IMU calibration is unverified"));
        File paired=new File(dir,PairedCapture.FILE);
        if(paired.isFile()){
            JSONObject reference=readJson(paired);
            validatePairing(reference,dir.getName());manifest.put("companion_capture",reference);
        }
        writeJson(new File(dir,"capture_manifest.json"),manifest);
    }
    private static String readText(InputStream in, int maximum) throws Exception {
        if (in == null) return "";
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192]; int n;
        while ((n = in.read(buffer)) != -1) {
            if (out.size() + n > maximum) throw new IOException("Server response is too large");
            out.write(buffer,0,n);
        }
        return out.toString("UTF-8");
    }
    public static File zip(File dir) throws Exception {
        File retained=new File(dir,"roomwalk-"+dir.getName()+".zip");
        if(new File(dir,PairedCapture.FILE).isFile()&&retained.isFile())return retained;
        File manifest = new File(dir, "capture_manifest.json");
        File video = new File(dir, "camera.mp4");
        if (!manifest.isFile() || !video.isFile() || video.length() == 0) throw new IOException("No finished video bundle. Raw diagnostics remain on this phone.");
        String[] names = {"capture_manifest.json","camera.mp4","accel.csv","gyro.csv","timestamps.csv","encoder_pts.csv","camera_results.jsonl","capture_result.json","capabilities.json",PairedCapture.FILE};
        long total=0;
        for (String name:names) total+=new File(dir,name).length();
        if (total > 7L*1024*1024*1024) throw new IOException("Capture exceeds the upload bundle limit; raw files are retained.");
        if (dir.getUsableSpace() < total + 64L*1024*1024) throw new IOException("Not enough phone storage to package this recording. Raw files are retained; free space and retry.");
        File partial = new File(dir, "roomwalk-"+dir.getName()+".zip.partial");
        File archive = new File(dir, "roomwalk-"+dir.getName()+".zip");
        try (FileOutputStream file = new FileOutputStream(partial); ZipOutputStream zip = new ZipOutputStream(new BufferedOutputStream(file,262144))) {
            zip.setLevel(0); byte[] buffer = new byte[262144];
            for (String name:names) {
                File member = new File(dir,name); if (!member.isFile()) continue;
                zip.putNextEntry(new ZipEntry(name));
                try (InputStream in = new BufferedInputStream(new FileInputStream(member),262144)) {
                    int n; while((n=in.read(buffer))!=-1) zip.write(buffer,0,n);
                }
                zip.closeEntry();
            }
            zip.finish(); zip.flush(); file.getFD().sync();
        }
        if (!partial.renameTo(archive)) throw new IOException("Could not finalize archive; partial archive and raw data retained.");
        return archive;
    }
    public static File zipImu(File dir) throws Exception {
        String[] names={"imu_capture_manifest.json","accel.csv","gyro.csv"};
        JSONObject manifest=readJson(new File(dir,names[0]));
        if(!"noesis.phone_imu_calibration.v1".equals(manifest.optString("schema"))||"recording".equals(manifest.optString("status")))
            throw new IOException("IMU recording did not finalize. Raw files remain on this phone.");
        long total=0;
        for(String name:names){File f=new File(dir,name);if(!f.isFile()||f.length()==0)throw new IOException("Missing IMU file: "+name);total+=f.length();}
        if(total>ImuCalibrationWriter.MAX_BYTES)throw new IOException("IMU files exceed the 512 MiB bundle limit");
        if(dir.getUsableSpace()<total+64L*1024*1024)throw new IOException("Not enough space to package IMU data; raw files are retained");
        File partial=new File(dir,"roomwalk-"+dir.getName()+".zip.partial"),archive=new File(dir,"roomwalk-"+dir.getName()+".zip");
        try(FileOutputStream file=new FileOutputStream(partial);ZipOutputStream zip=new ZipOutputStream(new BufferedOutputStream(file,262144))){
            zip.setLevel(1);
            for(String name:names){zip.putNextEntry(new ZipEntry(name));copy(new File(dir,name),zip);zip.closeEntry();}
            zip.finish();zip.flush();file.getFD().sync();
        }
        if(!partial.renameTo(archive))throw new IOException("Could not finalize IMU archive; source data retained");
        return archive;
    }
    public static JSONObject uploadImu(Context context,String server,File archive,Progress progress) throws Exception {
        return uploadImu(context,server,archive,progress,null);
    }
    static JSONObject uploadImu(Context context,String server,File archive,Progress progress,UploadTransfer transfer) throws Exception {
        if(!archive.isFile()||archive.length()==0||archive.length()>ImuCalibrationWriter.MAX_BYTES)throw new IOException("IMU bundle is missing or too large");
        HttpsURLConnection conn=connection(context,endpoint(server,"/api/phone-calibration/imu-bundle"));
        try{
            if(transfer!=null)transfer.attach(conn);
            configureArchive(conn,archive);
            sendArchive(conn,archive,progress,transfer);
            int code=conn.getResponseCode();String body;
            try(InputStream in=code>=200&&code<300?conn.getInputStream():conn.getErrorStream()){body=readText(in,512*1024);}
            if(transfer!=null)transfer.check();
            if(code!=201)throw new IOException("IMU upload returned HTTP "+code+": "+body.substring(0,Math.min(body.length(),800))+". Local files are retained.");
            JSONObject receipt=new JSONObject(body);
            if(!"stored".equals(receipt.optString("status"))||!archive.getParentFile().getName().equals(receipt.optString("capture_id")))throw new IOException("Unexpected IMU upload receipt; local files are retained");
            return receipt;
        }finally{conn.disconnect();if(transfer!=null)transfer.detached();}
    }
    private static void configureArchive(HttpsURLConnection conn,File archive)throws Exception{
        conn.setRequestMethod("POST");conn.setDoOutput(true);conn.setRequestProperty("Content-Type","application/zip");
        conn.setRequestProperty("X-File-Name",archive.getName());conn.setFixedLengthStreamingMode(archive.length());
    }
    private static String sendArchive(HttpsURLConnection conn,File archive,Progress progress,UploadTransfer transfer)throws Exception{
        java.security.MessageDigest digest=java.security.MessageDigest.getInstance("SHA-256");
        if(transfer!=null)transfer.phase("uploading");
        try(InputStream in=new BufferedInputStream(new FileInputStream(archive),262144);OutputStream out=new BufferedOutputStream(conn.getOutputStream(),262144)){
            byte[] buffer=new byte[262144];long sent=0,last=0;int n;
            while((n=in.read(buffer))!=-1){
                if(transfer!=null)transfer.check();out.write(buffer,0,n);digest.update(buffer,0,n);sent+=n;
                if(transfer!=null)transfer.progress();long now=SystemClock.elapsedRealtime();
                if(now-last>=500){progress.update(sent,archive.length());last=now;}
            }
            out.flush();progress.update(sent,archive.length());
        }
        if(transfer!=null)transfer.phase("validating");
        StringBuilder hash=new StringBuilder();for(byte value:digest.digest())hash.append(String.format(java.util.Locale.ROOT,"%02x",value&255));return hash.toString();
    }
    public static URL endpoint(String server, String path) throws Exception {
        URI uri = new URI(server.trim());
        if (!"https".equalsIgnoreCase(uri.getScheme()) || uri.getHost()==null || uri.getUserInfo()!=null || uri.getQuery()!=null || uri.getFragment()!=null || !(uri.getPath().isEmpty() || "/".equals(uri.getPath())))
            throw new IOException("Enter an HTTPS RoomWalk origin, for example https://your-machine.local:8789");
        return new URI(server.replaceAll("/+$", "")+path).toURL();
    }
    private static HttpsURLConnection connection(Context context, URL url) throws Exception {
        HttpsURLConnection conn=connection(context,url,SystemClock.elapsedRealtime()+SETUP_TIMEOUT_MS);
        conn.setReadTimeout(180000);return conn;
    }
    private static HttpsURLConnection connection(Context context, URL url,long deadline) throws Exception {
        return LanIpv4Https.open(url,tlsFactory(context),deadline);
    }
    private static volatile javax.net.ssl.SSLSocketFactory verifiedTlsFactory;
    private static synchronized javax.net.ssl.SSLSocketFactory tlsFactory(Context context)throws Exception{
        if(verifiedTlsFactory!=null)return verifiedTlsFactory;
        // Trust this appliance's public CA,
        // never all certificates; optional standard trust for portable builds.
        try (InputStream ca=context.getAssets().open("roomwalk-ca.pem")) {
            KeyStore store=KeyStore.getInstance(KeyStore.getDefaultType()); store.load(null);
            store.setCertificateEntry("roomwalk", CertificateFactory.getInstance("X.509").generateCertificate(ca));
            TrustManagerFactory factory=TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm()); factory.init(store);
            SSLContext ssl=SSLContext.getInstance("TLS"); ssl.init(null,factory.getTrustManagers(),null);verifiedTlsFactory=ssl.getSocketFactory();
        } catch (FileNotFoundException ignored) {verifiedTlsFactory=HttpsURLConnection.getDefaultSSLSocketFactory();}
        return verifiedTlsFactory;
    }
    public static JSONObject health(Context context, String server) throws Exception {
        java.util.concurrent.Future<JSONObject> request;
        try{request=HEALTH_REQUESTS.submit(()->(JSONObject)requestJsonValue(context,server,"/api/health",null,SETUP_TIMEOUT_MS));}
        catch(java.util.concurrent.RejectedExecutionException error){throw new IOException("Earlier connection checks are still resolving. Wait briefly, then retry.",error);}
        try{return request.get(SETUP_TIMEOUT_MS,java.util.concurrent.TimeUnit.MILLISECONDS);}
        catch(java.util.concurrent.TimeoutException error){request.cancel(true);throw new IOException("Connection check timed out. Check the RoomWalk address and phone Wi-Fi, then retry.",error);}
        catch(java.util.concurrent.ExecutionException error){Throwable cause=error.getCause();if(cause instanceof Exception)throw (Exception)cause;throw new IOException("Connection check failed",cause);}
        catch(InterruptedException error){request.cancel(true);Thread.currentThread().interrupt();throw error;}
    }
    // Bound the complete LAN check, retain at most two in-flight read-only
    // checks, and never queue more work. DNS itself is cancellable and bounded.
    private static final java.util.concurrent.ThreadPoolExecutor HEALTH_REQUESTS=new java.util.concurrent.ThreadPoolExecutor(0,2,30,java.util.concurrent.TimeUnit.SECONDS,
            new java.util.concurrent.SynchronousQueue<>(),task->{Thread thread=new Thread(task,"roomwalk-health");thread.setDaemon(true);return thread;});
    public static JSONObject pairing(File archive) throws Exception {
        try(ZipFile zip=new ZipFile(archive)){
            ZipEntry entry=zip.getEntry("capture_manifest.json");if(entry==null)return null;
            if(entry.getSize()>512*1024)throw new IOException("Capture manifest exceeds its bound");
            JSONObject manifest;try(InputStream in=zip.getInputStream(entry)){manifest=new JSONObject(readText(in,512*1024));}
            JSONObject reference=manifest.optJSONObject("companion_capture");
            if(reference!=null)validatePairing(reference,manifest.getString("capture_id"));return reference;
        }
    }
    static void validatePairing(JSONObject reference,String captureId)throws Exception{
        if(!PairedCapture.SCHEMA.equals(reference.optString("schema"))||!captureId.equals(reference.optString("phone_capture_id"))
                ||!reference.optString("session_id").matches("companion-[0-9]{8}-[0-9]{6}-[a-f0-9]{8}")||reference.optString("camera_id").isEmpty())throw new IOException("Paired recording identity is incomplete; use Finalize paired capture before packaging");
        endpoint(reference.getString("server_origin"),"/");
    }
    public static final class HttpFailure extends IOException {
        public final int status;
        HttpFailure(int status,String message){super(message);this.status=status;}
    }
    public static Object requestJsonValue(Context context,String server,String path,JSONObject body,int timeoutMs)throws Exception{
        long deadline=SystemClock.elapsedRealtime()+timeoutMs;
        HttpsURLConnection conn=connection(context,endpoint(server,path),deadline);
        try{
            if(body!=null){byte[] bytes=body.toString().getBytes(StandardCharsets.UTF_8);if(bytes.length>256*1024)throw new IOException("Companion request exceeds its bound");conn.setRequestMethod("POST");conn.setDoOutput(true);conn.setRequestProperty("Content-Type","application/json");conn.setFixedLengthStreamingMode(bytes.length);try(OutputStream out=conn.getOutputStream()){out.write(bytes);}}
            int code=conn.getResponseCode();ByteArrayOutputStream response=new ByteArrayOutputStream();
            try(InputStream in=code>=200&&code<300?conn.getInputStream():conn.getErrorStream()){
                if(in!=null){byte[] buffer=new byte[8192];int n;while(true){long remaining=deadline-SystemClock.elapsedRealtime();if(remaining<=0)throw new IOException(requestLabel(path)+" timed out "+(response.size()==0?"while connecting to RoomWalk":"while reading the response")+". Check the RoomWalk address and phone Wi-Fi.");conn.setReadTimeout((int)Math.min(timeoutMs,remaining));n=in.read(buffer);if(n<0)break;if(response.size()+n>512*1024)throw new IOException("Companion response exceeds its bound");response.write(buffer,0,n);}}
            }
            String text=response.toString("UTF-8");if(code<200||code>=300)throw new HttpFailure(code,"RoomWalk returned HTTP "+code+": "+text.substring(0,Math.min(text.length(),500)));
            Object value=new JSONTokener(text).nextValue();if(!(value instanceof JSONObject)&&!(value instanceof JSONArray))throw new IOException("Invalid companion response");return value;
        }finally{conn.disconnect();}
    }
    private static String requestLabel(String path){return path.equals("/api/health")?"RoomWalk health check":path.endsWith("/cameras")?"Room camera list":path.endsWith("/clock")?"Clock check":path.endsWith("/heartbeat")?"Room capture heartbeat":path.endsWith("/stop")?"Room capture stop":"Room capture request";}
    public static JSONObject upload(Context context, String server, File archive, Progress progress) throws Exception {
        return upload(context,server,archive,progress,"Android "+archive.getParentFile().getName());
    }
    public static JSONObject upload(Context context, String server, File archive, Progress progress,String name) throws Exception {
        return upload(context,server,archive,progress,name,null);
    }
    static JSONObject upload(Context context,String server,File archive,Progress progress,String name,UploadTransfer transfer)throws Exception{
        if(!archive.isFile()||archive.length()==0||archive.length()>8L*1024*1024*1024)throw new IOException("Capture bundle is missing or exceeds the upload limit");
        JSONObject paired=pairing(archive);
        if(paired!=null&&!endpoint(server,"/").toURI().equals(endpoint(paired.getString("server_origin"),"/").toURI()))throw new IOException("Upload this paired recording to its original RoomWalk server: "+paired.getString("server_origin"));
        HttpsURLConnection conn=connection(context,endpoint(server,"/api/scans/sensor-bundle?name="+URLEncoder.encode(name,"UTF-8")));
        try {
            if(transfer!=null)transfer.attach(conn);
            configureArchive(conn,archive);
            if(paired!=null){conn.setRequestProperty("X-Companion-Session",paired.getString("session_id"));conn.setRequestProperty("X-Companion-Camera-ID",paired.getString("camera_id"));conn.setRequestProperty("X-Phone-Capture-ID",paired.getString("phone_capture_id"));}
            if(transfer!=null)conn.setRequestProperty("Prefer","respond-async");
            String digest=sendArchive(conn,archive,progress,transfer);
            int code=conn.getResponseCode();String text;
            try(InputStream in=code>=200&&code<300?conn.getInputStream():conn.getErrorStream()){text=readText(in,512*1024);}
            if(transfer!=null)transfer.check();
            if(code==202&&transfer!=null){
                JSONObject receipt=new JSONObject(text);validateStoredReceipt(receipt,archive,digest,paired);
                return receipt.put("validation_status","pending");
            }
            if(code!=201&&!(paired!=null&&code==200))throw new IOException("RoomWalk returned HTTP "+code+": "+text.substring(0,Math.min(text.length(),800))+". The bundle remains on this phone.");
            JSONObject receipt=new JSONObject(text);
            if(!receipt.optString("id").matches("[0-9]{8}-[0-9]{6}-[a-f0-9]{8}"))throw new IOException("Unexpected scan upload receipt; local files are retained");
            if(paired!=null){JSONObject linked=receipt.optJSONObject("companion_capture");JSONObject phone=linked==null?null:linked.optJSONObject("phone");String id=receipt.getString("id");
                if(linked==null||phone==null||!id.equals(phone.optString("scan_id"))
                        ||!paired.getString("session_id").equals(linked.optString("session_id"))||!paired.getString("camera_id").equals(linked.optString("camera_id"))
                        ||!paired.getString("phone_capture_id").equals(linked.optString("phone_capture_id")))throw new IOException("The upload receipt does not confirm the paired recording identities; retain this ZIP and retry");
            }
            return receipt;
        }finally{conn.disconnect();if(transfer!=null)transfer.detached();}
    }
    static void validateStoredReceipt(JSONObject receipt,File archive,String digest,JSONObject paired)throws Exception{
        JSONObject stored=receipt.optJSONObject("upload_receipt");String captureId;
        try(ZipFile zip=new ZipFile(archive)){
            ZipEntry entry=zip.getEntry("capture_manifest.json");if(entry==null)throw new IOException("Capture manifest is missing from the retained ZIP");
            try(InputStream in=zip.getInputStream(entry)){captureId=new JSONObject(readText(in,512*1024)).getString("capture_id");}
        }
        if(stored==null||!receipt.optString("id").matches("[0-9]{8}-[0-9]{6}-[a-f0-9]{8}")
                ||!"noesis.phone_capture.upload_receipt.v1".equals(stored.optString("schema"))||!"stored".equals(stored.optString("status"))
                ||archive.length()!=stored.optLong("size_bytes",-1)||!digest.equals(stored.optString("sha256"))||!captureId.equals(stored.optString("capture_id")))
            throw new IOException("RoomWalk did not confirm the exact stored ZIP bytes and capture identity. Retain this ZIP and retry.");
        if(paired!=null){
            if(!paired.getString("session_id").equals(stored.optString("companion_session_id"))||!paired.getString("camera_id").equals(stored.optString("companion_camera_id"))
                    ||!paired.getString("phone_capture_id").equals(stored.optString("capture_id")))throw new IOException("Stored upload receipt does not match the paired recording identities");
        }else if(!stored.isNull("companion_session_id")||!stored.isNull("companion_camera_id"))throw new IOException("An unpaired upload received an unexpected paired receipt");
    }
    public static JSONObject uploadPhoneReport(Context context, String server, File report) throws Exception {
        if (!report.isFile() || report.length() > 512 * 1024) throw new IOException("Phone report is missing or too large");
        HttpsURLConnection conn=connection(context,endpoint(server,"/api/phone-diagnostics"));
        try {
            conn.setRequestMethod("POST"); conn.setDoOutput(true);
            conn.setRequestProperty("Content-Type","application/json");
            conn.setFixedLengthStreamingMode(report.length());
            try(OutputStream out=conn.getOutputStream()){copy(report,out);}
            int code=conn.getResponseCode();
            String text;
            try(InputStream in=code>=200&&code<300?conn.getInputStream():conn.getErrorStream()){text=readText(in,65536);}
            if(code!=201)throw new IOException("Phone report upload returned HTTP "+code+": "+text.substring(0,Math.min(text.length(),400)));
            JSONObject receipt=new JSONObject(text);
            if(!"noesis.phone_capture.android_diagnostic_receipt.v1".equals(receipt.optString("schema")) || !receipt.optString("id").matches("[a-f0-9]{64}"))throw new IOException("Unexpected phone report response");
            return receipt;
        }finally{conn.disconnect();}
    }
    public static void copy(File file, OutputStream out) throws Exception {
        try(InputStream in=new BufferedInputStream(new FileInputStream(file))) { byte[] b=new byte[262144]; int n; while((n=in.read(b))!=-1) out.write(b,0,n); }
    }
}

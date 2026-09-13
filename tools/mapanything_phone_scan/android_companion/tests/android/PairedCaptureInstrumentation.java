package org.noesis.roomwalk;

import android.app.Instrumentation;
import android.os.Bundle;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/** Installed only as a separate test APK; production camera/encoder gates are untouched. */
public final class PairedCaptureInstrumentation extends Instrumentation {
    private int checks;
    private Bundle arguments;
    @Override public void onCreate(Bundle args){arguments=args;super.onCreate(args);start();}
    @Override public void onStart(){
        Bundle result=new Bundle();
        try{
            if(arguments.getString("uploadBenchServer")!=null)checks+=UploadInstrumentation.benchmark(this,arguments);
            else if(arguments.getString("uploadStoredServer")!=null)checks+=UploadInstrumentation.storedService(this,arguments);
            else if(arguments.getString("uploadServiceServer")!=null)checks+=UploadInstrumentation.service(this,arguments);
            else if(arguments.getString("uploadRecoveryServer")!=null||arguments.getString("uploadRecover")!=null)checks+=UploadInstrumentation.recovery(this,arguments);
            else if(arguments.getString("uploadReceipts")!=null)checks+=UploadInstrumentation.receipts(this);
            else if(arguments.getString("echoServer")!=null)ipv4Echo();
            else if(arguments.getString("ipv4DnsServer")!=null)ipv4Dns();
            else if(arguments.getString("pairedServer")!=null)livePair();
            else if(arguments.getString("dnsServer")!=null)dnsProbe();
            else if(arguments.getString("connectionServer")!=null)connectionProbe();
            else if(arguments.getString("healthTimeoutServer")!=null)healthTimeout();
            else if(arguments.getString("uploadSetupTimeoutServer")!=null)healthTimeout();
            else{lifecycleTests();if(arguments.getString("fixtureDirectory")!=null)fixtureUpload();}
            result.putString("stream","Paired capture instrumentation: "+checks+" checks passed\n");finish(-1,result);
        }catch(Throwable error){result.putString("stream","FAILED: "+error+"\n"+android.util.Log.getStackTraceString(error));finish(1,result);}
    }
    private void ipv4Echo()throws Exception{
        String server=arguments.getString("echoServer"),wrong=arguments.getString("wrongOrigin");java.net.URL origin=BundleTools.endpoint(server,"/");
        long started=android.os.SystemClock.elapsedRealtime();JSONObject response=BundleTools.health(getTargetContext(),server);
        JSONObject report=new JSONObject().put("echo",response).put("elapsed_ms",android.os.SystemClock.elapsedRealtime()-started);
        require("IPv4".equals(response.getString("peer_family")),"Server observes an IPv4 peer");
        require(origin.getAuthority().equalsIgnoreCase(response.getString("http_host")),"HTTP Host preserves the original origin authority");
        require(origin.getHost().equalsIgnoreCase(response.getString("tls_sni")),"TLS SNI preserves the original hostname");
        require("ok".equals(response.getString("status")),"Verified original-host request succeeds");
        boolean rejected=false;try{BundleTools.health(getTargetContext(),wrong);}catch(Exception error){
            report.put("wrong_origin_error",android.util.Log.getStackTraceString(error));
            for(Throwable cause=error;cause!=null;cause=cause.getCause())if(cause instanceof javax.net.ssl.SSLHandshakeException||cause instanceof javax.net.ssl.SSLPeerUnverifiedException||cause instanceof java.security.cert.CertificateException)rejected=true;
        }
        report.put("wrong_origin_rejected_by_tls",rejected);BundleTools.writeJson(new File(getTargetContext().getExternalFilesDir(null),"ipv4-echo.json"),report);
        require(rejected,"An IPv4 original origin without a matching certificate SAN is rejected by TLS");
        java.security.KeyStore empty=java.security.KeyStore.getInstance(java.security.KeyStore.getDefaultType());empty.load(null);
        javax.net.ssl.TrustManagerFactory trust=javax.net.ssl.TrustManagerFactory.getInstance(javax.net.ssl.TrustManagerFactory.getDefaultAlgorithm());trust.init(empty);
        javax.net.ssl.SSLContext untrusted=javax.net.ssl.SSLContext.getInstance("TLS");untrusted.init(null,trust.getTrustManagers(),null);
        java.lang.reflect.Method factory=BundleTools.class.getDeclaredMethod("connection",android.content.Context.class,java.net.URL.class);factory.setAccessible(true);
        javax.net.ssl.HttpsURLConnection good=(javax.net.ssl.HttpsURLConnection)factory.invoke(null,getTargetContext(),BundleTools.endpoint(server,"/api/health"));
        javax.net.ssl.HttpsURLConnection bad=LanIpv4Https.open(BundleTools.endpoint(server,"/api/health"),untrusted.getSocketFactory(),android.os.SystemClock.elapsedRealtime()+3000);
        require(good.getSSLSocketFactory()!=bad.getSSLSocketFactory(),"Changing the trusted factory cannot reuse a cached trust binding");good.disconnect();
        boolean caRejected=false;try{bad.getResponseCode();}catch(Exception error){report.put("untrusted_ca_error",android.util.Log.getStackTraceString(error));for(Throwable cause=error;cause!=null;cause=cause.getCause())if(cause instanceof javax.net.ssl.SSLHandshakeException||cause instanceof java.security.cert.CertificateException)caRejected=true;}finally{bad.disconnect();}
        report.put("untrusted_ca_rejected",caRejected);BundleTools.writeJson(new File(getTargetContext().getExternalFilesDir(null),"ipv4-echo.json"),report);
        require(caRejected,"The original hostname still requires a trusted certificate chain");System.out.println("IPv4 echo: "+report);
    }
    private void ipv4Dns()throws Exception{
        String server=arguments.getString("ipv4DnsServer"),host=BundleTools.endpoint(server,"/").getHost();JSONArray attempts=new JSONArray();
        for(int i=0;i<3;i++){
            long started=android.os.SystemClock.elapsedRealtime();JSONObject row=new JSONObject().put("host",host).put("query_type","static").put("configured_ipv4",LanIpv4Https.ROOMWALK_IPV4);
            try{java.net.Inet4Address address=LanIpv4Https.resolve(host,started+3000);row.put("ipv4",address.getHostAddress()).put("ok",true);}
            catch(Exception error){row.put("ok",false).put("error",error.toString());}
            row.put("elapsed_ms",android.os.SystemClock.elapsedRealtime()-started);attempts.put(row);
        }
        BundleTools.writeJson(new File(getTargetContext().getExternalFilesDir(null),"ipv4-dns.json"),new JSONObject().put("attempts",attempts));
        System.out.println("IPv4 DNS probe: "+attempts);
        for(int i=0;i<attempts.length();i++)require(attempts.getJSONObject(i).getBoolean("ok")&&LanIpv4Https.ROOMWALK_IPV4.equals(attempts.getJSONObject(i).getString("ipv4"))&&attempts.getJSONObject(i).getLong("elapsed_ms")<100,"Configured RoomWalk IPv4 route is used without DNS");
    }
    private void livePair()throws Exception{
        String server=arguments.getString("pairedServer"),camera=arguments.getString("pairedCamera","living-room");File dir=directory("live-connection");Events events=new Events();PairedCapture paired=new PairedCapture(getTargetContext(),events);
        long started=android.os.SystemClock.elapsedRealtime();
        try{
            paired.start(dir,server,camera);require(events.ready.await(40,TimeUnit.SECONDS),"Actual paired start confirms video and tracking readiness");
            require(events.heartbeat.await(18,TimeUnit.SECONDS),"Actual paired heartbeat succeeds after its ten-second idle interval");
            require(events.errors==0,"No paired transport errors during live lease");
        }finally{paired.stop("user");require(events.stopped.await(25,TimeUnit.SECONDS),"Actual paired stop finalizes the exact retained session");paired.close();}
        JSONObject ref=BundleTools.readJson(new File(dir,PairedCapture.FILE));require("stopped".equals(ref.optString("status"))&&!ref.optBoolean("needs_recovery"),"Live static capture finalized without recovery");
        require(ref.getJSONArray("clock_probes").length()>=5,"Start, heartbeat and stop clock observations are retained");
        JSONObject report=new JSONObject().put("test_fixture",true).put("phone_video_recorded",false).put("elapsed_ms",android.os.SystemClock.elapsedRealtime()-started).put("reference",ref);
        BundleTools.writeJson(new File(getTargetContext().getExternalFilesDir(null),"live-paired-connection.json"),report);System.out.println("Live paired session: "+ref.getString("session_id"));
    }
    private void dnsProbe()throws Exception{
        String server=arguments.getString("dnsServer");java.net.URL url=BundleTools.endpoint(server,"/api/health");JSONObject report=new JSONObject().put("server",server);JSONArray addresses=new JSONArray();
        long start=android.os.SystemClock.elapsedRealtime();java.net.InetAddress[] resolved=java.net.InetAddress.getAllByName(url.getHost());report.put("dns_ms",android.os.SystemClock.elapsedRealtime()-start);
        java.lang.reflect.Method method=BundleTools.class.getDeclaredMethod("connection",android.content.Context.class,java.net.URL.class);method.setAccessible(true);javax.net.ssl.HttpsURLConnection conn=(javax.net.ssl.HttpsURLConnection)method.invoke(null,getTargetContext(),url);
        for(java.net.InetAddress address:resolved){JSONObject row=new JSONObject().put("address",address.getHostAddress());long tcp=android.os.SystemClock.elapsedRealtime();try(java.net.Socket raw=new java.net.Socket()){
            raw.connect(new java.net.InetSocketAddress(address,url.getPort()),3000);row.put("tcp_ms",android.os.SystemClock.elapsedRealtime()-tcp);
            long tls=android.os.SystemClock.elapsedRealtime();try(javax.net.ssl.SSLSocket socket=(javax.net.ssl.SSLSocket)conn.getSSLSocketFactory().createSocket(raw,url.getHost(),url.getPort(),true)){
                javax.net.ssl.SSLParameters parameters=socket.getSSLParameters();parameters.setEndpointIdentificationAlgorithm("HTTPS");socket.setSSLParameters(parameters);socket.setSoTimeout(5000);socket.startHandshake();row.put("verified_tls_ms",android.os.SystemClock.elapsedRealtime()-tls);
                long headers=android.os.SystemClock.elapsedRealtime();socket.getOutputStream().write(("GET /api/health HTTP/1.1\r\nHost: "+url.getHost()+":"+url.getPort()+"\r\nConnection: close\r\n\r\n").getBytes(java.nio.charset.StandardCharsets.US_ASCII));
                java.io.BufferedReader reader=new java.io.BufferedReader(new java.io.InputStreamReader(socket.getInputStream(),java.nio.charset.StandardCharsets.UTF_8));row.put("status_line",reader.readLine());while(!"".equals(reader.readLine())){}row.put("headers_ms",android.os.SystemClock.elapsedRealtime()-headers);
            }
        }catch(Exception error){row.put("error",error.toString()).put("elapsed_ms",android.os.SystemClock.elapsedRealtime()-tcp);}addresses.put(row);report.put("addresses",addresses);System.out.println("DNS/TCP/TLS probe: "+report);BundleTools.writeJson(new File(getTargetContext().getExternalFilesDir(null),"dns-probe.json"),report);}
        conn.disconnect();
    }
    private void connectionProbe()throws Exception{
        String server=arguments.getString("connectionServer");JSONObject report=new JSONObject().put("server",server).put("schema","noesis.android.connection_probe.v1");JSONArray attempts=new JSONArray();
        java.lang.reflect.Method factory=BundleTools.class.getDeclaredMethod("connection",android.content.Context.class,java.net.URL.class);factory.setAccessible(true);
        for(int round=0;round<2;round++){
            if(round==1)Thread.sleep(11000);
            String[] paths=round==0?new String[]{"/api/health","/api/companion-captures/cameras","/api/companion-captures/clock"}:new String[]{"/api/companion-captures/clock","/api/health","/api/companion-captures/cameras"};
            for(String path:paths){
                long start=android.os.SystemClock.elapsedRealtime();JSONObject row=new JSONObject().put("round",round).put("path",path);
                try{Object value=path.equals("/api/health")?BundleTools.health(getTargetContext(),server):BundleTools.requestJsonValue(getTargetContext(),server,path,null,path.endsWith("/clock")?3000:5000);row.put("ok",true);if(value instanceof JSONObject){JSONObject response=(JSONObject)value;row.put("status",response.optString("status"));if(response.has("cameras"))row.put("camera_count",response.getJSONArray("cameras").length());}}
                catch(Exception error){row.put("ok",false).put("error",error.toString());}
                row.put("elapsed_ms",android.os.SystemClock.elapsedRealtime()-start);attempts.put(row);report.put("attempts",attempts);
                BundleTools.writeJson(new File(getTargetContext().getExternalFilesDir(null),"connection-probe.json"),report);System.out.println("Connection probe: "+row);
            }
        }
        javax.net.ssl.HttpsURLConnection a=(javax.net.ssl.HttpsURLConnection)factory.invoke(null,getTargetContext(),BundleTools.endpoint(server,"/api/health"));
        javax.net.ssl.HttpsURLConnection b=(javax.net.ssl.HttpsURLConnection)factory.invoke(null,getTargetContext(),BundleTools.endpoint(server,"/api/health"));
        report.put("tls_factory_reused",a.getSSLSocketFactory()==b.getSSLSocketFactory()).put("route",a.getURL().toString());a.disconnect();b.disconnect();
        report.put("default_connect_timeout_ms",a.getConnectTimeout()).put("default_upload_read_timeout_ms",a.getReadTimeout());
        BundleTools.writeJson(new File(getTargetContext().getExternalFilesDir(null),"connection-probe.json"),report);
        if(arguments.getString("requireConnectionSuccess")!=null){for(int i=0;i<attempts.length();i++)require(attempts.getJSONObject(i).getBoolean("ok"),"Actual health, room cameras and clock succeed on cold and warm requests");require(report.getBoolean("tls_factory_reused"),"One verified TLS factory permits pooled route reuse");require(a.getConnectTimeout()==750&&a.getReadTimeout()==180000,"Short IPv4 TCP setup preserves upload reads");require(attempts.getJSONObject(0).getLong("elapsed_ms")<3000,"Cold health succeeds within the three-second LAN budget");}
    }
    private void healthTimeout()throws Exception{
        boolean upload=arguments.getString("uploadSetupTimeoutServer")!=null;String server=arguments.getString(upload?"uploadSetupTimeoutServer":"healthTimeoutServer");java.net.ServerSocket stall=null;Thread fixture=null;CountDownLatch release=new CountDownLatch(1);
        if(server.equals("loopback")){stall=new java.net.ServerSocket(0,1,java.net.InetAddress.getByAddress(new byte[]{127,0,0,1}));final java.net.ServerSocket listener=stall;server="https://127.0.0.1:"+stall.getLocalPort();fixture=new Thread(()->{try(java.net.Socket socket=listener.accept()){release.await(20,TimeUnit.SECONDS);}catch(Exception ignored){}});fixture.setDaemon(true);fixture.start();}
        long started=android.os.SystemClock.elapsedRealtime();boolean failed=false;
        try{
            if(upload){java.lang.reflect.Method factory=BundleTools.class.getDeclaredMethod("connection",android.content.Context.class,java.net.URL.class);factory.setAccessible(true);javax.net.ssl.HttpsURLConnection conn=(javax.net.ssl.HttpsURLConnection)factory.invoke(null,getTargetContext(),BundleTools.endpoint(server,"/api/health"));require(conn.getReadTimeout()==180000,"Upload body read allowance remains180seconds");try{conn.getResponseCode();}finally{conn.disconnect();}}
            else BundleTools.health(getTargetContext(),server);
        }
        catch(IOException error){failed=true;System.out.println("Health timeout probe: "+error);require(error.getMessage().toLowerCase().contains("timed out"),"Stalled connection gives a timeout: "+error);}
        finally{release.countDown();if(stall!=null)stall.close();if(fixture!=null)fixture.join(1000);}
        require(failed,"Stalled connection cannot report healthy");
        long elapsed=android.os.SystemClock.elapsedRealtime()-started;require(elapsed<BundleTools.SETUP_TIMEOUT_MS+500,"Setup caller remains bounded during a stalled connection");
        BundleTools.writeJson(new File(getTargetContext().getExternalFilesDir(null),upload?"upload-setup-timeout.json":"health-timeout.json"),new JSONObject().put("fixture",server).put("elapsed_ms",elapsed).put("failed_as_expected",failed));
    }
    private void lifecycleTests()throws Exception{
        require("user".equals(PairedCapture.phoneStopReason(new JSONObject().put("stop_reason","user_stop").put("partial",false))),"Native user_stop maps to a clean static user stop");
        require("user".equals(PairedCapture.phoneStopReason(new JSONObject().put("stop_reason","short_test_duration_limit_10_seconds").put("partial",false))),"Completed native short test maps to a clean static stop");
        require("sensor_stalled".equals(PairedCapture.phoneStopReason(new JSONObject().put("stop_reason","sensor_stalled").put("partial",true))),"Interrupted native capture retains the actual failure reason");
        FakeServer server=new FakeServer();Events events=new Events();File dir=directory("normal");PairedCapture paired=new PairedCapture(server,events);paired.start(dir,"https://fixture.invalid","living_room");
        require(events.ready.await(5,TimeUnit.SECONDS),"Ready after server components confirm recording");
        require(server.created==1&&server.posts==1&&server.clocks==3&&server.healthChecks==1,"One bounded health warmup, one start and three initial clock probes");
        require(server.heartbeat.await(13,TimeUnit.SECONDS),"Bounded periodic heartbeat reaches the server");
        paired.stop("user");paired.stop("app_closed");require(events.stopped.await(5,TimeUnit.SECONDS),"Stop finalizes paired reference");paired.close();
        JSONObject ref=BundleTools.readJson(new File(dir,PairedCapture.FILE));require("stopped".equals(ref.getString("status")),"Server final state retained");
        require(dir.getName().equals(server.stopPhoneId),"Stop identifies exact phone take");require(!ref.getBoolean("synchronization_verified"),"HTTP timing does not claim acquisition sync");
        require("user".equals(server.stopReason),"Normal stop preserves the server's clean completion reason through a later close");
        JSONObject probe=ref.getJSONArray("clock_probes").getJSONObject(0);require("1750000000000000123".equals(probe.getString("server_received_unix_ns")),"Server nanoseconds above 2^53 remain exact strings");
        require(Long.parseLong(probe.getString("client_receive_elapsed_realtime_ns"))>=Long.parseLong(probe.getString("client_send_elapsed_realtime_ns")),"Native elapsed clock observations remain ordered");

        server=new FakeServer();server.loseFirstStart=true;events=new Events();dir=directory("retry");paired=new PairedCapture(server,events);paired.start(dir,"https://fixture.invalid","living_room");
        require(events.ready.await(5,TimeUnit.SECONDS),"Lost start response can be retried");require(server.created==1&&server.posts==2&&server.requestIds.get(0).equals(server.requestIds.get(1)),"Retry preserves request ID and creates one session");paired.stop("user");require(events.stopped.await(5,TimeUnit.SECONDS),"Retried session stops");paired.close();

        server=new FakeServer();server.wrongIdentity=true;events=new Events();dir=directory("wrong-identity");paired=new PairedCapture(server,events);paired.start(dir,"https://fixture.invalid","living_room");
        require(events.stopped.await(5,TimeUnit.SECONDS),"Mismatched start response terminates");require(events.ready.getCount()==1&&events.errors>0,"Mismatched identity never starts the phone");paired.close();

        server=new FakeServer();server.blockStart=true;events=new Events();dir=directory("cancel");paired=new PairedCapture(server,events);paired.start(dir,"https://fixture.invalid","living_room");
        require(server.startEntered.await(5,TimeUnit.SECONDS),"Start request entered");paired.stop("app_backgrounded");server.startRelease.countDown();require(events.stopped.await(5,TimeUnit.SECONDS),"Late start response is stopped after background cancellation");require(events.ready.getCount()==1&&server.stops==1,"Cancelled start never releases phone start callback");paired.close();

        server=new FakeServer();server.stopFails=true;events=new Events();dir=directory("stop-retry");paired=new PairedCapture(server,events);paired.start(dir,"https://fixture.invalid","living_room");require(events.ready.await(5,TimeUnit.SECONDS),"Recovery fixture starts");paired.stop("user");require(events.stopped.await(5,TimeUnit.SECONDS),"Unconfirmed stop retains reference");paired.close();
        ref=BundleTools.readJson(new File(dir,PairedCapture.FILE));require(ref.getBoolean("needs_recovery"),"Lost stop explicitly requires recovery");server.stopFails=false;ref=PairedCapture.finishSaved(server,dir);require("stopped".equals(ref.getString("status"))&&!ref.getBoolean("needs_recovery")&&server.created==1,"Saved retry finalizes same session without a new recording");
        require("user".equals(server.stopReason),"Saved retry preserves the original clean stop reason");
        System.out.println("Lifecycle checks: "+checks);
    }
    private void fixtureUpload()throws Exception{
        File directory=new File(arguments.getString("fixtureDirectory"));String destination=arguments.getString("serverUrl");
        JSONObject nativeResult=BundleTools.readJson(new File(directory,"capture_result.json"));
        JSONObject sourceManifest=BundleTools.readJson(new File(directory,"capture_manifest.json"));
        BundleTools.writeCaptureManifest(getTargetContext(),directory,nativeResult);
        JSONObject manifest=BundleTools.readJson(new File(directory,"capture_manifest.json"));
        manifest.put("device",sourceManifest.getJSONObject("device")).put("validation_fixture",new JSONObject().put("source_capture_id",sourceManifest.getString("capture_id")).put("fixture_reused_phone_recording",true).put("packaging_producer","Android instrumentation using installed RoomWalk BundleTools").put("simultaneous_acquisition_verified",false));
        BundleTools.writeJson(new File(directory,"capture_manifest.json"),manifest);File zip=BundleTools.zip(directory);
        JSONObject pairing=BundleTools.pairing(zip);require(pairing!=null&&pairing.optBoolean("fixture_reused_phone_recording"),"Fixture provenance explicitly states phone evidence was reused");
        String name="Validation fixture — reused phone + static";
        String originalHash=sha256(zip);JSONObject first=BundleTools.upload(getTargetContext(),destination,zip,(sent,total)->{},name);
        JSONObject second=BundleTools.upload(getTargetContext(),destination,zip,(sent,total)->{},name);
        require(first.getString("id").equals(second.getString("id")),"Exact repeated ZIP returns the same scan");
        BundleTools.writeCaptureManifest(getTargetContext(),directory,nativeResult);File repeated=BundleTools.zip(directory);
        require(originalHash.equals(sha256(repeated)),"Package again preserves immutable paired ZIP bytes");
        JSONObject output=new JSONObject().put("fixture_reused_phone_recording",true).put("simultaneous_acquisition_verified",false).put("first_receipt",first).put("duplicate_receipt",second).put("archive_sha256",originalHash).put("archive_bytes",zip.length());
        BundleTools.writeJson(new File(directory,"instrumented_upload_result.json"),output);
    }
    private File directory(String label)throws Exception{File root=new File(getTargetContext().getExternalFilesDir(null),"paired-validation");root.mkdirs();File out=new File(root,"fixture-"+label+"-"+UUID.randomUUID());if(!out.mkdirs())throw new IOException("Cannot create fixture directory");return out;}
    private void require(boolean condition,String message){checks++;if(!condition)throw new AssertionError(message);}
    private static String sha256(File file)throws Exception{MessageDigest digest=MessageDigest.getInstance("SHA-256");try(java.io.InputStream in=Files.newInputStream(file.toPath())){byte[] bytes=new byte[262144];int n;while((n=in.read(bytes))>=0)digest.update(bytes,0,n);}StringBuilder out=new StringBuilder();for(byte value:digest.digest())out.append(String.format("%02x",value&255));return out.toString();}
    private static final class Events implements PairedCapture.Listener{
        final CountDownLatch ready=new CountDownLatch(1),stopped=new CountDownLatch(1),heartbeat=new CountDownLatch(1);volatile int errors;
        public void onPairReady(File directory,JSONObject reference){ready.countDown();}
        public void onPairState(String state,JSONObject reference){if(state.equals("recording"))heartbeat.countDown();}
        public void onPairStopped(File directory,JSONObject reference){stopped.countDown();}
        public void onPairError(String reason,JSONObject reference){errors++;}
    }
    private static final class FakeServer implements PairedCapture.Transport{
        int created,posts,clocks,stops,healthChecks;boolean loseFirstStart,wrongIdentity,blockStart,stopFails;String phone,camera,request,status="recording",stopPhoneId,stopReason;
        final List<String> requestIds=new ArrayList<>();final CountDownLatch heartbeat=new CountDownLatch(1),startEntered=new CountDownLatch(1),startRelease=new CountDownLatch(1);
        public Object request(String server,String path,JSONObject body,int timeout)throws Exception{
            if(path.equals("/api/health")){healthChecks++;return new JSONObject().put("status","ok");}
            if(path.endsWith("/clock")){clocks++;return new JSONObject().put("server_received_unix_ns","1750000000000000123").put("server_sent_unix_ns","1750000000000000456").put("server_received_monotonic_ns","1234500000000123").put("server_sent_monotonic_ns","1234500000000456");}
            if(path.equals("/api/companion-captures")&&body==null)return created==0?new JSONArray():new JSONArray().put(response(false));
            if(path.equals("/api/companion-captures")){
                posts++;requestIds.add(body.getString("client_request_id"));if(created==0){created++;phone=body.getString("phone_capture_id");camera=body.getString("camera_id");request=body.getString("client_request_id");}
                else if(!request.equals(body.getString("client_request_id")))throw new IOException("Changed retry identity");
                startEntered.countDown();if(blockStart&&!startRelease.await(5,TimeUnit.SECONDS))throw new IOException("Fixture release timeout");
                if(loseFirstStart&&posts==1)throw new IOException("Simulated lost start response");return response(wrongIdentity);
            }
            if(path.endsWith("/heartbeat")){heartbeat.countDown();return response(false);}
            if(path.endsWith("/stop")){stops++;stopPhoneId=body.getString("phone_capture_id");stopReason=body.getString("reason");if(stopFails)throw new IOException("Simulated lost stop response");status="stopped";return response(false);}
            return response(false);
        }
        private JSONObject response(boolean bad)throws Exception{return new JSONObject().put("session_id","companion-20260909-000000-deadbeef").put("camera_id",bad?"wrong-room":camera).put("phone_capture_id",phone).put("status",status).put("video_status",status).put("tracking_status",status).put("artifact_urls",new JSONObject()).put("provenance",new JSONObject().put("test_fixture",true));}
    }
}

package org.noesis.roomwalk;

import android.app.*;
import android.content.*;
import android.net.Uri;
import android.os.Build;
import android.webkit.*;
import android.view.*;
import android.widget.*;
import org.json.JSONObject;
import java.io.*;
import java.net.URL;
import java.util.*;
import java.util.concurrent.*;
import javax.net.ssl.HttpsURLConnection;

/** Packaged web application, verified server APIs, and a main-frame-only native command channel. */
final class WebShell {
    interface Host { void command(String action, JSONObject args) throws Exception; void error(String message); }
    static final int PICK_FILES=51, SAVE_DOWNLOAD=52;
    private final Activity activity;
    private final Host host;
    final WebView view;
    private String origin;
    private String lastPublished;
    private ValueCallback<Uri[]> fileCallback;
    private Uri pendingDownload;
    private boolean downloading,closed;
    private final ExecutorService downloads=Executors.newSingleThreadExecutor();
    WebShell(Activity activity,Host host,String server)throws Exception {
        this.activity=activity;this.host=host;
        view=new WebView(activity);configure(view,true);
        view.getSettings().setUserAgentString(view.getSettings().getUserAgentString()+" RoomWalkAndroid/0.2");
        view.setWebViewClient(new WebViewClient(){
            @Override public void onPageStarted(WebView web,String url,android.graphics.Bitmap icon){lastPublished=null;}
            @Override public WebResourceResponse shouldInterceptRequest(WebView web,WebResourceRequest request){
                Uri uri=request.getUrl();
                if(!sameOrigin(uri))return blocked();
                if(!"GET".equals(request.getMethod()))return null;
                String path=uri.getPath();
                if("/".equals(path)||"/index.html".equals(path))path="/static/index.html";
                if(!(path.startsWith("/static/")||path.startsWith("/three/")))return null;
                if(path.contains("..")||path.contains("\\")||path.indexOf('\0')>=0)return blocked();
                try { return new WebResourceResponse(mime(path),"UTF-8",200,"OK",
                    Collections.singletonMap("Cache-Control","no-cache"),activity.getAssets().open("web"+path)); }
                catch(IOException missing){return null;}
            }
            @Override public boolean shouldOverrideUrlLoading(WebView web,WebResourceRequest request){
                Uri uri=request.getUrl();
                if("roomwalk-native".equals(uri.getScheme())){
                    if(request.isForMainFrame()&&isAppPage(view.getUrl()))dispatch(uri);
                    return true;
                }
                if(!request.isForMainFrame())return !sameOrigin(uri);
                if(!sameOrigin(uri)){host.error("External links cannot control RoomWalk. Open them separately if needed.");return true;}
                if(isAppPage(uri.toString()))return false;
                preview(uri);return true;
            }
            @Override public void onPageFinished(WebView web,String url){if(isAppPage(url))try{host.command("snapshot",new JSONObject());}catch(Exception error){host.error(error.getMessage());}}
        });
        view.setWebChromeClient(new WebChromeClient(){
            @Override public boolean onShowFileChooser(WebView web,ValueCallback<Uri[]> callback,FileChooserParams params){
                if(!isAppPage(view.getUrl()))return false;
                if(fileCallback!=null)fileCallback.onReceiveValue(null);fileCallback=callback;
                Intent pick=new Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*");
                String[] types=params.getAcceptTypes();if(types.length>0&&!types[0].isEmpty())pick.putExtra(Intent.EXTRA_MIME_TYPES,types);
                pick.putExtra(Intent.EXTRA_ALLOW_MULTIPLE,params.getMode()==FileChooserParams.MODE_OPEN_MULTIPLE);
                try{activity.startActivityForResult(pick,PICK_FILES);}catch(Exception error){fileCallback=null;callback.onReceiveValue(null);host.error("No file picker is available");}return true;
            }
            @Override public void onPermissionRequest(PermissionRequest request){request.deny();}
        });
        view.setDownloadListener((url,agent,disposition,type,length)->save(Uri.parse(url)));
        connect(server);
    }
    private static void configure(WebView web,boolean javascript){
        WebSettings settings=web.getSettings();settings.setJavaScriptEnabled(javascript);settings.setDomStorageEnabled(javascript);
        settings.setAllowFileAccess(false);settings.setAllowContentAccess(true);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        settings.setJavaScriptCanOpenWindowsAutomatically(false);settings.setSupportMultipleWindows(false);
        settings.setMediaPlaybackRequiresUserGesture(true);settings.setSafeBrowsingEnabled(true);
        web.setBackgroundColor(0xfff5f7f6);
    }
    void connect(String server)throws Exception {
        origin=BundleTools.endpoint(server,"/").toString().replaceAll("/+$","");
        view.loadUrl(origin+"/?android=1");
    }
    private boolean sameOrigin(Uri uri){
        if(origin==null||uri==null)return false;Uri expected=Uri.parse(origin);
        return "https".equals(uri.getScheme())&&expected.getHost()!=null&&uri.getHost()!=null&&expected.getHost().equalsIgnoreCase(uri.getHost())&&port(expected)==port(uri)&&uri.getUserInfo()==null;
    }
    private static int port(Uri uri){return uri.getPort()<0?443:uri.getPort();}
    private boolean isAppPage(String url){if(url==null)return false;Uri uri=Uri.parse(url);return sameOrigin(uri)&&("/".equals(uri.getPath())||"/index.html".equals(uri.getPath()));}
    private void dispatch(Uri uri){
        // A command needs an acknowledgement even when its resulting state is
        // unchanged. Deduplicate background publications, never this reply.
        lastPublished=null;
        try{
            String payload=uri.getQueryParameter("request");
            if(!"action".equals(uri.getHost())||payload==null||payload.length()>65536)throw new IOException("Invalid native request");
            JSONObject request=new JSONObject(payload);JSONObject args=request.optJSONObject("args");
            host.command(request.getString("action"),args==null?new JSONObject():args);
        }catch(Exception error){host.error(error.getMessage()==null?"Native action failed":error.getMessage());}
    }
    void publish(JSONObject state){
        if(closed||!isAppPage(view.getUrl()))return;
        String json=state.toString().replace("\u2028","\\u2028").replace("\u2029","\\u2029");
        if(json.equals(lastPublished))return;
        lastPublished=json;
        view.evaluateJavascript("window.RoomWalkNative&&window.RoomWalkNative.receive("+json+")",received->{if(!"true".equals(received))lastPublished=null;});
    }
    private void preview(Uri uri){
        if(!sameOrigin(uri))return;
        Dialog dialog=new Dialog(activity);LinearLayout root=new LinearLayout(activity);root.setOrientation(LinearLayout.VERTICAL);
        LinearLayout bar=new LinearLayout(activity);Button close=new Button(activity);close.setText("Back to RoomWalk");close.setOnClickListener(v->dialog.dismiss());bar.addView(close);
        Button download=new Button(activity);download.setText("Save file");download.setOnClickListener(v->save(uri));bar.addView(download);root.addView(bar);
        WebView asset=new WebView(activity);configure(asset,false);
        asset.setWebViewClient(new WebViewClient(){
            @Override public boolean shouldOverrideUrlLoading(WebView web,WebResourceRequest request){return !sameOrigin(request.getUrl());}
            @Override public WebResourceResponse shouldInterceptRequest(WebView web,WebResourceRequest request){return sameOrigin(request.getUrl())?null:blocked();}
        });
        asset.setDownloadListener((url,agent,disposition,type,length)->save(Uri.parse(url)));
        root.addView(asset,new LinearLayout.LayoutParams(-1,0,1));dialog.setContentView(root);dialog.setOnDismissListener(d->asset.destroy());dialog.show();dialog.getWindow().setLayout(-1,-1);asset.loadUrl(uri.toString());
    }
    private void save(Uri uri){
        if(!sameOrigin(uri)){host.error("Only files from the configured RoomWalk server can be saved");return;}
        if(downloading||pendingDownload!=null){host.error("Finish the current file save first");return;}
        pendingDownload=uri;String name=uri.getLastPathSegment();if(name==null||name.length()>180)name="roomwalk-download";
        Intent intent=new Intent(Intent.ACTION_CREATE_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("application/octet-stream");intent.putExtra(Intent.EXTRA_TITLE,name);
        try{activity.startActivityForResult(intent,SAVE_DOWNLOAD);}catch(Exception error){pendingDownload=null;host.error("No save destination is available");}
    }
    boolean result(int request,int result,Intent data){
        if(request==PICK_FILES){
            if(fileCallback!=null){Uri[] files=null;
                if(result==Activity.RESULT_OK&&data!=null){if(data.getClipData()!=null){int count=Math.min(data.getClipData().getItemCount(),64);files=new Uri[count];for(int i=0;i<count;i++)files[i]=data.getClipData().getItemAt(i).getUri();}else if(data.getData()!=null)files=new Uri[]{data.getData()};}
                fileCallback.onReceiveValue(files);fileCallback=null;}return true;
        }
        if(request!=SAVE_DOWNLOAD)return false;
        Uri source=pendingDownload;pendingDownload=null;
        if(result!=Activity.RESULT_OK||data==null||data.getData()==null||source==null)return true;
        final Uri destination=data.getData();downloading=true;
        downloads.execute(()->{try{
            HttpsURLConnection connection=BundleTools.downloadConnection(activity,new URL(source.toString()));
            try{
                String cookie=CookieManager.getInstance().getCookie(source.toString());if(cookie!=null)connection.setRequestProperty("Cookie",cookie);
                int code=connection.getResponseCode();if(code!=200)throw new IOException("Download returned HTTP "+code);
                try(InputStream in=connection.getInputStream();OutputStream out=activity.getContentResolver().openOutputStream(destination)){
                    if(out==null)throw new IOException("Save destination is unavailable");byte[] bytes=new byte[262144];long count=0;int n;
                    while((n=in.read(bytes))!=-1){count+=n;if(count>32L*1024*1024*1024)throw new IOException("Download exceeds the 32 GiB limit");out.write(bytes,0,n);}
                }
            }finally{connection.disconnect();}
            activity.runOnUiThread(()->Toast.makeText(activity,"File saved",Toast.LENGTH_SHORT).show());
        }catch(Exception error){activity.runOnUiThread(()->host.error("File save incomplete: "+error.getMessage()));}
        finally{activity.runOnUiThread(()->downloading=false);}});return true;
    }
    boolean back(){if(view.canGoBack()){view.goBack();return true;}return false;}
    void close(){closed=true;if(fileCallback!=null)fileCallback.onReceiveValue(null);fileCallback=null;view.stopLoading();view.destroy();downloads.shutdown();}
    private static WebResourceResponse blocked(){return new WebResourceResponse("text/plain","UTF-8",403,"Blocked",Collections.emptyMap(),new ByteArrayInputStream(new byte[0]));}
    private static String mime(String path){if(path.endsWith(".html"))return "text/html";if(path.endsWith(".js"))return "text/javascript";if(path.endsWith(".css"))return "text/css";if(path.endsWith(".json"))return "application/json";if(path.endsWith(".svg"))return "image/svg+xml";if(path.endsWith(".png"))return "image/png";return "application/octet-stream";}
}

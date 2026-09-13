package org.noesis.roomwalk;

import android.Manifest;
import android.app.*;
import android.content.*;
import android.content.pm.PackageManager;
import android.content.pm.ActivityInfo;
import android.content.res.Configuration;
import android.graphics.*;
import android.graphics.drawable.GradientDrawable;
import android.hardware.display.DisplayManager;
import android.net.Uri;
import android.os.*;
import android.view.*;
import android.widget.*;
import org.json.*;
import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.*;

/** Foreground-only native capture. Every finished take remains available locally. */
public final class MainActivity extends Activity implements CaptureEngine.Listener, ImuCalibrationRecorder.Listener, PairedCapture.Listener {
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private CaptureEngine engine;
    private ImuCalibrationRecorder imuRecorder;
    private PairedCapture paired;
    private Spinner roomCamera;
    private JSONArray roomCameras=new JSONArray();
    private String roomCamerasOrigin="";
    private TextView roomCameraStatus;
    private Button finalizePairButton;
    private JSONObject pendingNativeResult,pendingCamera;
    private boolean pendingShortTest,nativeStarted,pairFinished,packagingStarted;
    private long pendingUseCase;
    private String pairedRoomLabel="room camera";
    private EditText imuMinutes;
    private TextView imuProgress;
    private Button imuStartButton,imuStopButton,focusButton;
    private boolean imuActive,focusBusy;
    private CameraPreview previewController;
    private Dialog captureDialog;
    private TextureView preview;
    private FrameLayout previewBox;
    private DisplayManager displays;
    private final DisplayManager.DisplayListener displayListener = new DisplayManager.DisplayListener() {
        public void onDisplayAdded(int id) {}
        public void onDisplayRemoved(int id) {}
        public void onDisplayChanged(int id) {
            if (preview != null && preview.getDisplay() != null && preview.getDisplay().getDisplayId() == id) fitPreview();
        }
    };
    private Surface previewSurface;
    private TextView status, detail, saved, captureStatus, captureStats;
    private EditText server;
    private Spinner camera;
    private Button captureButton, captureBackButton, probeButton, testButton, recordButton, stopButton, uploadButton, exportButton, shareButton, historyButton, checkButton, packageButton, reportButton;
    private JSONArray cameras = new JSONArray();
    private JSONObject capabilityReport;
    private File captureDir, artifact, pendingExport, phoneReport;
    private boolean active=false, busy=false, destroyed=false;
    private boolean previewReady=false, viewerStarting=false, viewerClosing=false;
    private boolean pendingCaptureViewer,transferActive;
    private LinearLayout transferPanel;
    private TextView transferStatus,transferDetail;
    private ProgressBar transferProgress;
    private Button cancelUploadButton,retryUploadButton;
    private final Runnable transferPoll=new Runnable(){public void run(){if(destroyed)return;refreshTransfer();ui.postDelayed(this,750);}};
    private String lastScanId=null;
    private static final int CAMERA_PERMISSION=41, EXPORT_DOCUMENT=42, UPLOAD_NOTIFICATION_PERMISSION=43;
    private static final int INK=Color.rgb(24,42,42), GREEN=Color.rgb(23,107,100), MUTED=Color.rgb(85,103,101);

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        engine = new CaptureEngine(this, this);
        imuRecorder = new ImuCalibrationRecorder(this,this);
        paired = new PairedCapture(this,this);
        previewController = new CameraPreview(this);
        displays = getSystemService(DisplayManager.class);
        displays.registerDisplayListener(displayListener, ui);
        LinearLayout root=column(); root.setBackgroundColor(Color.rgb(245,247,246));
        root.setPadding(dp(20),dp(10),dp(20),dp(10));
        root.setOnApplyWindowInsetsListener((v,insets)->{
            if(Build.VERSION.SDK_INT>=30){
                android.graphics.Insets bars=insets.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
                v.setPadding(bars.left+dp(20),bars.top+dp(10),bars.right+dp(20),bars.bottom+dp(10));
            }else{v.setPadding(insets.getSystemWindowInsetLeft()+dp(20),insets.getSystemWindowInsetTop()+dp(10),insets.getSystemWindowInsetRight()+dp(20),insets.getSystemWindowInsetBottom()+dp(10));}
            return insets;
        });
        TextView title=text("RoomWalk",25,INK); title.setTypeface(null,Typeface.BOLD); root.addView(title);
        root.addView(text("NATIVE CAMERA + MOTION  •  8K / 30 FPS",11,GREEN));
        ScrollView scroll=new ScrollView(this); LinearLayout body=column(); body.setPadding(0,dp(10),0,dp(8)); scroll.addView(body);
        root.addView(scroll,new LinearLayout.LayoutParams(-1,0,1));
        status=text("Check the phone before recording",18,INK); status.setTypeface(null,Typeface.BOLD); body.addView(status);
        detail=text("Choose the phone and room cameras. Record saves native phone video and motion alongside the matching room-camera recording and tracking evidence.",13,MUTED); body.addView(detail);
        probeButton=button("Check phone",()->checkPhone());body.addView(probeButton);
        camera=new Spinner(this);body.addView(camera,new LinearLayout.LayoutParams(-1,dp(50)));
        reportButton=button("Upload phone report",()->uploadPhoneReport());body.addView(reportButton);
        camera.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener(){
            public void onNothingSelected(android.widget.AdapterView<?> parent) { updateButtons(); }
            public void onItemSelected(android.widget.AdapterView<?> p,View v,int position,long id) {
                JSONObject c=selectedCamera();
                if(c!=null) detail.setText(cameraDescription(c));
                fitPreview();updateButtons();
            }
        });
        body.addView(text("Capture opens the full-screen camera. Compose your shot, then record video and motion together. Leaving the app stops and saves the take. Maximum: 10 minutes or 6 GiB of video.",12,MUTED));
        TextView destination=text("ROOMWALK SERVER",11,GREEN); destination.setPadding(0,dp(16),0,0); body.addView(destination);
        server=new EditText(this); server.setSingleLine(true); server.setTextSize(14); server.setInputType(android.text.InputType.TYPE_CLASS_TEXT|android.text.InputType.TYPE_TEXT_VARIATION_URI);
        server.setText(getPreferences(0).getString("server",defaultServer())); body.addView(server);
        LinearLayout destinationButtons=row(); checkButton=button("Check connection",()->checkConnection()); addRowButton(destinationButtons,checkButton);
        addRowButton(destinationButtons,button("Open RoomWalk",()->openRoomWalk())); body.addView(destinationButtons);
        body.addView(text("PAIRED ROOM CAMERA",13,GREEN));
        roomCamera=new Spinner(this);body.addView(roomCamera,new LinearLayout.LayoutParams(-1,dp(50)));
        roomCameraStatus=text("Check connection to load available room cameras. A room camera is required for a paired walk.",12,MUTED);body.addView(roomCameraStatus);
        roomCamera.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener(){
            public void onNothingSelected(AdapterView<?> parent){updateButtons();}
            public void onItemSelected(AdapterView<?> parent,View view,int index,long id){JSONObject row=selectedRoomCamera();if(row!=null){getPreferences(0).edit().putString("room_camera_id",row.optString("camera_id")).apply();roomCameraStatus.setText("Record will capture this room camera and tracking alongside phone video + IMU.");}updateButtons();}
        });
        LinearLayout advanced=column();advanced.setVisibility(View.GONE);
        body.addView(button("Advanced diagnostics",()->advanced.setVisibility(advanced.getVisibility()==View.VISIBLE?View.GONE:View.VISIBLE)));
        TextView calibrationTitle=text("OPTIONAL IMU DIAGNOSTIC",15,GREEN);calibrationTitle.setPadding(0,dp(16),0,0);advanced.addView(calibrationTitle);
        advanced.addView(text("Normal walks already record IMU data. This optional 1–5 minute sensor-only check is for troubleshooting; it is not required before a walk. Keep the app open.",12,MUTED));
        LinearLayout imuRow=row();imuRow.addView(text("Minutes",13,INK));
        imuMinutes=new EditText(this);imuMinutes.setInputType(android.text.InputType.TYPE_CLASS_NUMBER);imuMinutes.setSingleLine(true);imuMinutes.setText(getPreferences(0).getString("imu_diagnostic_minutes","1"));imuRow.addView(imuMinutes,new LinearLayout.LayoutParams(dp(80),dp(50)));
        advanced.addView(imuRow);imuStartButton=button("Start IMU diagnostic",()->startImu());imuStopButton=button("Stop & save IMU",()->stopImu());advanced.addView(imuStartButton);advanced.addView(imuStopButton);
        imuProgress=text("No IMU recording is active. Saved takes appear under Saved captures.",12,MUTED);advanced.addView(imuProgress);
        body.addView(advanced);
        saved=text("No capture selected. Your recordings stay on this phone after upload.",13,MUTED); saved.setPadding(0,dp(12),0,0); body.addView(saved);
        uploadButton=button("Upload",()->upload()); exportButton=button("Export",()->export()); shareButton=button("Share",()->share()); historyButton=button("Saved captures",()->chooseSaved());
        packageButton=button("Package again",()->repackage());
        body.addView(uploadButton,new LinearLayout.LayoutParams(-1,dp(50)));
        transferPanel=column();transferPanel.setPadding(dp(12),dp(8),dp(12),dp(8));transferPanel.setBackgroundColor(Color.rgb(231,240,237));
        transferStatus=text("",15,INK);transferDetail=text("",12,MUTED);transferProgress=new ProgressBar(this,null,android.R.attr.progressBarStyleHorizontal);transferProgress.setMax(1000);
        transferPanel.addView(transferStatus);transferPanel.addView(transferProgress,new LinearLayout.LayoutParams(-1,dp(8)));transferPanel.addView(transferDetail);
        cancelUploadButton=button("Cancel upload",()->cancelUpload());retryUploadButton=button("Retry this upload",()->retryUpload());transferPanel.addView(cancelUploadButton);transferPanel.addView(retryUploadButton);transferPanel.setVisibility(View.GONE);body.addView(transferPanel);
        LinearLayout exportRow=row();addRowButton(exportRow,exportButton);addRowButton(exportRow,shareButton);body.addView(exportRow);
        body.addView(historyButton,new LinearLayout.LayoutParams(-1,dp(50)));body.addView(packageButton);
        finalizePairButton=button("Finalize paired capture",()->finalizePair());body.addView(finalizePairButton);
        captureButton=button("Capture",()->openCaptureViewer());captureButton.setTextSize(17);
        root.addView(captureButton,new LinearLayout.LayoutParams(-1,dp(56)));
        setContentView(root); root.requestApplyInsets(); loadLatest(); updateButtons();
    }
    private int dp(int value){return Math.round(value*getResources().getDisplayMetrics().density);}
    private LinearLayout column(){LinearLayout l=new LinearLayout(this);l.setOrientation(LinearLayout.VERTICAL);return l;}
    private LinearLayout row(){LinearLayout l=new LinearLayout(this);l.setOrientation(LinearLayout.HORIZONTAL);l.setGravity(Gravity.CENTER_VERTICAL);return l;}
    private TextView text(String value,int size,int color){TextView t=new TextView(this);t.setText(value);t.setTextSize(size);t.setTextColor(color);t.setPadding(0,dp(4),0,dp(4));return t;}
    private Button button(String label,Runnable action){Button b=new Button(this);b.setText(label);b.setTextSize(12);b.setAllCaps(false);b.setOnClickListener(v->action.run());return b;}
    private void addRowButton(LinearLayout row,Button button){LinearLayout.LayoutParams p=new LinearLayout.LayoutParams(0,-2,1);p.setMargins(dp(2),0,dp(2),0);row.addView(button,p);}
    private String defaultServer(){try(InputStream in=getAssets().open("server.json")){ByteArrayOutputStream b=new ByteArrayOutputStream();byte[] buf=new byte[1024];int n;while((n=in.read(buf))!=-1)b.write(buf,0,n);return new JSONObject(b.toString("UTF-8")).getString("url");}catch(Exception e){return "https://your-machine.local:8789";}}
    private File captureRoot(){File root=new File(getExternalFilesDir(null),"captures");root.mkdirs();return root;}
    private JSONObject selectedCamera(){return cameras.optJSONObject(camera.getSelectedItemPosition());}
    private long standardTestUseCase(JSONObject c){return SessionProbe.standardTestUseCase(capabilityReport==null?null:capabilityReport.optJSONObject("session_queries"),c);}
    private boolean fullWalkAvailable(JSONObject c){return c!=null&&(c.optBoolean("supported8k")||CapturePreflight.qualifiesWalk(c,standardTestUseCase(c)));}
    private String cameraDescription(JSONObject c){
        if(c.optBoolean("supported8k"))return "Rear camera "+c.optString("id")+" reports an 8K hardware recording path. Run the timing test to check actual recording.";
        if(fullWalkAvailable(c))return "This camera passed a saved 8K/30 timing test with the same recording configuration. Record walk is ready. Phone video and raw motion are retained together.";
        if(standardTestUseCase(c)>=0)return "The driver accepts this camera's standard 8K/30 configuration. Run the 10-second test, then upload the saved capture for video and timing checks. Full walks remain disabled for this newly found mode.";
        return c.optString("reason","8K recording is unavailable for this camera.");
    }
    private void updateButtons(){
        JSONObject c=selectedCamera();boolean homeReady=!active&&!imuActive&&!busy&&captureDialog==null&&!pendingCaptureViewer;
        boolean previewAvailable=preview!=null&&preview.isAvailable()&&previewReady;
        boolean canStart=!active&&!imuActive&&!focusBusy&&!busy&&!viewerStarting&&!viewerClosing&&previewAvailable&&selectedRoomCamera()!=null;
        captureButton.setEnabled(homeReady&&c!=null&&c.optInt("preview_width")>0&&c.optInt("preview_height")>0);
        if(testButton!=null){
            testButton.setEnabled(canStart&&(fullWalkAvailable(c)||standardTestUseCase(c)>=0));
            recordButton.setEnabled(canStart&&fullWalkAvailable(c));
            recordButton.setVisibility(active?View.GONE:View.VISIBLE);testButton.setVisibility(active?View.GONE:View.VISIBLE);
            stopButton.setVisibility(active?View.VISIBLE:View.GONE);stopButton.setEnabled(active&&!viewerClosing);
            captureBackButton.setEnabled(!viewerClosing);captureBackButton.setText(active?"Save & close":"Close");
            for(Button b:new Button[]{recordButton,testButton,stopButton,captureBackButton})b.setAlpha(b.isEnabled()?1f:0.38f);
        }
        if(focusButton!=null){boolean locked=hasFocusLock();focusButton.setText(locked?"Unlock focus":"Lock focus");focusButton.setEnabled(!active&&!viewerStarting&&!viewerClosing&&!focusBusy&&(locked||previewAvailable));focusButton.setAlpha(focusButton.isEnabled()?1f:0.38f);}
        imuStartButton.setEnabled(homeReady);imuMinutes.setEnabled(homeReady);imuStopButton.setEnabled(imuActive&&!busy);
        imuStopButton.setVisibility(imuActive?View.VISIBLE:View.GONE);
        if(roomCamera!=null)roomCamera.setEnabled(homeReady);
        if(finalizePairButton!=null){boolean present=captureDir!=null&&new File(captureDir,PairedCapture.FILE).isFile();finalizePairButton.setVisibility(present?View.VISIBLE:View.GONE);finalizePairButton.setEnabled(homeReady&&present);}
        probeButton.setEnabled(homeReady);camera.setEnabled(homeReady);checkButton.setEnabled(homeReady);server.setEnabled(homeReady);historyButton.setEnabled(homeReady);
        reportButton.setEnabled(homeReady&&phoneReport!=null&&phoneReport.isFile());
        uploadButton.setEnabled(homeReady&&!transferActive&&artifact!=null&&artifact.getName().endsWith(".zip"));exportButton.setEnabled(homeReady&&artifact!=null);shareButton.setEnabled(homeReady&&artifact!=null);
        packageButton.setEnabled(homeReady&&!transferActive&&captureDir!=null&&(new File(captureDir,"capture_result.json").isFile()||new File(captureDir,"imu_capture_manifest.json").isFile()));
    }
    private void fitPreview(){
        if(preview==null||preview.getWidth()==0||preview.getHeight()==0)return;
        JSONObject c=selectedCamera();if(c==null)return;
        if(c.optInt("preview_width")<=0||c.optInt("preview_height")<=0)return;
        int rotation=preview.getDisplay()==null?0:preview.getDisplay().getRotation()*90;
        Matrix m=new Matrix();m.setValues(PreviewTransform.matrix(preview.getWidth(),preview.getHeight(),
                c.optInt("preview_width",1920),c.optInt("preview_height",1080),
                c.optInt("sensor_orientation_degrees",90),rotation));preview.setTransform(m);
    }
    private Button viewerButton(String label,int color,Runnable action){
        Button b=button(label,action);b.setTextColor(Color.WHITE);b.setTextSize(13);
        GradientDrawable background=new GradientDrawable();background.setColor(color);background.setCornerRadius(dp(10));b.setBackground(background);
        LinearLayout.LayoutParams params=new LinearLayout.LayoutParams(-1,dp(48));params.topMargin=dp(8);b.setLayoutParams(params);return b;
    }
    private void openCaptureViewer(){
        JSONObject c=selectedCamera();if(destroyed||active||imuActive||busy||captureDialog!=null||c==null||c.optInt("preview_width")<=0||c.optInt("preview_height")<=0)return;
        if(checkSelfPermission(Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED){checkPhone();return;}
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE);
        if(getResources().getConfiguration().orientation!=Configuration.ORIENTATION_LANDSCAPE){
            if(pendingCaptureViewer)return;
            pendingCaptureViewer=true;updateButtons();
            ui.postDelayed(()->{if(pendingCaptureViewer&&!destroyed){pendingCaptureViewer=false;setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);status.setText("Camera screen could not rotate");detail.setText("Return to the full-screen app, then open Capture again.");updateButtons();}},5000);
            return;
        }
        pendingCaptureViewer=false;
        final Dialog dialog=new Dialog(this,android.R.style.Theme_Material_NoActionBar){
            @Override public void onBackPressed(){requestCloseCaptureViewer();}
        };
        captureDialog=dialog;previewReady=false;viewerStarting=false;viewerClosing=false;
        dialog.setCanceledOnTouchOutside(false);
        LinearLayout viewer=row();viewer.setBackgroundColor(Color.BLACK);
        previewBox=new FrameLayout(this);previewBox.setBackgroundColor(Color.BLACK);
        preview=new TextureView(this);previewBox.addView(preview,new FrameLayout.LayoutParams(-1,-1));
        viewer.addView(previewBox,new LinearLayout.LayoutParams(0,-1,1));
        LinearLayout rail=column();rail.setPadding(dp(12),dp(12),dp(12),dp(12));rail.setBackgroundColor(Color.rgb(17,27,27));
        ScrollView railScroll=new ScrollView(this);railScroll.setFillViewport(true);railScroll.addView(rail);viewer.addView(railScroll,new LinearLayout.LayoutParams(dp(144),-1));
        TextView mode=text("8K · 30 FPS",13,Color.WHITE);mode.setGravity(Gravity.CENTER);rail.addView(mode);
        captureStatus=text("Opening camera…",13,Color.WHITE);captureStatus.setGravity(Gravity.CENTER);rail.addView(captureStatus);
        captureStats=text("Video + IMU",11,Color.LTGRAY);captureStats.setGravity(Gravity.CENTER);rail.addView(captureStats);
        rail.addView(new View(this),new LinearLayout.LayoutParams(1,0,1));
        focusButton=viewerButton(hasFocusLock()?"Unlock focus":"Lock focus",Color.rgb(49,68,68),()->changeFocus());rail.addView(focusButton);
        recordButton=viewerButton("Record",GREEN,()->startCapture(false));rail.addView(recordButton);
        testButton=viewerButton("10-second test",Color.rgb(49,68,68),()->startCapture(true));rail.addView(testButton);
        stopButton=viewerButton("Stop & save",Color.rgb(169,49,49),()->requestCloseCaptureViewer());rail.addView(stopButton);
        rail.addView(new View(this),new LinearLayout.LayoutParams(1,0,1));
        captureBackButton=viewerButton("Close",Color.rgb(49,68,68),()->requestCloseCaptureViewer());rail.addView(captureBackButton);
        viewer.setOnApplyWindowInsetsListener((v,insets)->{
            if(Build.VERSION.SDK_INT>=30){android.graphics.Insets cutout=insets.getInsets(WindowInsets.Type.displayCutout());v.setPadding(cutout.left,cutout.top,cutout.right,cutout.bottom);}
            return insets;
        });
        preview.setSurfaceTextureListener(new TextureView.SurfaceTextureListener(){
            public void onSurfaceTextureAvailable(SurfaceTexture texture,int w,int h){
                if(captureDialog!=dialog||viewerClosing||viewerStarting||active)return;
                texture.setDefaultBufferSize(c.optInt("preview_width"),c.optInt("preview_height"));
                previewSurface=new Surface(texture);fitPreview();
                previewController.open(c.optString("id"),previewSurface,new CameraPreview.Listener(){
                    public void onReady(){if(destroyed||captureDialog!=dialog||viewerClosing||viewerStarting)return;previewReady=true;captureStatus.setText("Ready");captureStats.setText(selectedRoomCamera()==null?"Select a room camera on the main screen":fullWalkAvailable(c)?"Phone + room camera + IMU":standardTestUseCase(c)>=0?"Run a short test":"8K recording unavailable");updateButtons();}
                    public void onError(String message){if(destroyed||captureDialog!=dialog||viewerClosing||viewerStarting)return;previewReady=false;captureStatus.setText("Preview unavailable");captureStats.setText(message);updateButtons();}
                });updateButtons();
            }
            public void onSurfaceTextureSizeChanged(SurfaceTexture texture,int w,int h){if(captureDialog==dialog)fitPreview();}
            public boolean onSurfaceTextureDestroyed(SurfaceTexture texture){if(captureDialog==dialog)requestCloseCaptureViewer();return true;}
            public void onSurfaceTextureUpdated(SurfaceTexture texture){}
        });
        dialog.setContentView(viewer);dialog.show();
        Window window=dialog.getWindow();window.setLayout(-1,-1);window.setBackgroundDrawableResource(android.R.color.black);
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        if(Build.VERSION.SDK_INT>=30){
            window.setDecorFitsSystemWindows(false);WindowInsetsController insets=window.getInsetsController();
            if(insets!=null){insets.setSystemBarsBehavior(WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);insets.hide(WindowInsets.Type.systemBars());}
        }else window.getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_FULLSCREEN|View.SYSTEM_UI_FLAG_HIDE_NAVIGATION|View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY|View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN|View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION|View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
        if(Build.VERSION.SDK_INT>=33)dialog.getOnBackInvokedDispatcher().registerOnBackInvokedCallback(android.window.OnBackInvokedDispatcher.PRIORITY_DEFAULT,()->requestCloseCaptureViewer());
        viewer.requestApplyInsets();updateButtons();
    }
    private void requestCloseCaptureViewer(){
        if(pendingCaptureViewer){pendingCaptureViewer=false;setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);if(!destroyed)updateButtons();}
        final Dialog dialog=captureDialog;if(dialog==null||viewerClosing)return;
        viewerClosing=true;previewReady=false;captureStatus.setText(active?"Saving…":"Closing camera…");updateButtons();
        paired.stop("user");
        if(active){engine.stop();return;}
        if(viewerStarting)return; // The pending preview-close callback cancels this start.
        previewController.close((confirmed,reason)->{
            if(captureDialog!=dialog)return;
            if(!confirmed&&!destroyed){status.setText("Camera preview needs attention");detail.setText(reason);}
            finishCaptureViewer(dialog);
        });
    }
    private void finishCaptureViewer(Dialog dialog){
        if(dialog==null||captureDialog!=dialog)return;
        captureDialog=null;previewReady=false;viewerStarting=false;viewerClosing=false;
        dialog.dismiss();if(previewSurface!=null){previewSurface.release();previewSurface=null;}
        preview=null;previewBox=null;focusButton=null;focusBusy=false;testButton=null;recordButton=null;stopButton=null;captureBackButton=null;captureStatus=null;captureStats=null;
        if(!destroyed){setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);updateButtons();}
    }
    @Override public void onConfigurationChanged(Configuration configuration){
        super.onConfigurationChanged(configuration);
        if(pendingCaptureViewer&&configuration.orientation==Configuration.ORIENTATION_LANDSCAPE)ui.post(()->{if(pendingCaptureViewer&&!destroyed)openCaptureViewer();});
        if(preview!=null)preview.post(()->fitPreview());
    }
    private void checkPhone(){
        if(checkSelfPermission(Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED){requestPermissions(new String[]{Manifest.permission.CAMERA},CAMERA_PERMISSION);return;}
        busy=true;status.setText("Checking native cameras and encoders…");updateButtons();
        worker.execute(()->{try{
            JSONObject report=engine.probe();
            ui.post(()->{if(!destroyed)status.setText("Checking exact 8K sessions with the camera driver…");});
            JSONObject sessionQueries=SessionProbe.run(this,report);report.put("session_queries",sessionQueries);
            if(sessionQueries.optInt("query_count")>0)report.put("advertised_capabilities_only",false);
            File dir=new File(captureRoot(),"phone-check-"+System.currentTimeMillis());dir.mkdirs();File f=new File(dir,"capabilities.json");BundleTools.writeJson(f,report);
            ui.post(()->{if(destroyed)return;capabilityReport=report;cameras=report.optJSONArray("cameras");if(cameras==null)cameras=new JSONArray();
                ArrayList<String> labels=new ArrayList<>();int select=0;boolean found=false;
                for(int i=0;i<cameras.length();i++){JSONObject c=cameras.optJSONObject(i);boolean testReady=standardTestUseCase(c)>=0;labels.add(c.optString("label","Camera "+c.optString("id"))+ (fullWalkAvailable(c)?" — 8K ready":testReady?" — 8K test ready":" — mode needs diagnosis"));if(!found&&(fullWalkAvailable(c)||testReady)){select=i;found=true;}}
                ArrayAdapter<String> adapter=new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,labels);camera.setAdapter(adapter);if(!labels.isEmpty())camera.setSelection(select);
                phoneReport=f;artifact=f;captureDir=dir;saved.setText("Latest phone report selected. Upload it for camera-mode diagnosis. Recordings remain in Saved captures.");
                status.setText(fullWalkAvailable(selectedCamera())?"Phone check complete — ready to record":found?"Phone check complete — run the short test":"8K camera-mode check complete");
                JSONObject selected=selectedCamera();detail.setText(selected==null?report.optString("error","No camera entries returned."):cameraDescription(selected));
                busy=false;fitPreview();updateButtons();checkConnection();
            });
        }catch(Exception e){fail("Phone check failed",e);}});
    }
    @Override public void onRequestPermissionsResult(int request,String[] permissions,int[] grants){super.onRequestPermissionsResult(request,permissions,grants);if(request==CAMERA_PERMISSION){if(grants.length>0&&grants[0]==PackageManager.PERMISSION_GRANTED)checkPhone();else{status.setText("Camera permission is required");detail.setText("Allow camera access in Android app settings, then check the phone again.");}}}
    private void startCapture(boolean shortTest){
        JSONObject c=selectedCamera();if(c==null||selectedRoomCamera()==null||preview==null||!preview.isAvailable()||!previewReady||active||busy||viewerStarting||viewerClosing)return;
        long useCase=standardTestUseCase(c);
        if(!c.optBoolean("supported8k")&&(useCase<0||(!shortTest&&!fullWalkAvailable(c))))return;
        final Dialog dialog=captureDialog;viewerStarting=true;previewReady=false;
        captureStatus.setText("Preparing recording…");captureStats.setText("Keep the phone still");updateButtons();
        previewController.close((confirmed,reason)->{
            if(captureDialog!=dialog)return;
            if(destroyed||viewerClosing){finishCaptureViewer(dialog);return;}
            viewerStarting=false;
            if(!confirmed){finishCaptureViewer(dialog);status.setText("Camera could not switch to recording");detail.setText(reason);return;}
            beginCapture(shortTest,c,useCase,dialog);
        });
    }
    private void beginCapture(boolean shortTest,JSONObject c,long useCase,Dialog dialog){
        try{
            if(previewSurface==null||!previewSurface.isValid()||captureDialog!=dialog)throw new IOException("The camera view closed before recording started");
            JSONObject room=selectedRoomCamera();if(room==null)throw new IOException("Choose an available room camera before recording");
            String id=new SimpleDateFormat("yyyyMMdd-HHmmss",Locale.US).format(new Date())+"-"+UUID.randomUUID().toString().substring(0,8);
            captureDir=new File(captureRoot(),id);if(!captureDir.mkdirs())throw new IOException("Cannot create capture folder");
            pendingCamera=c;pendingShortTest=shortTest;pendingUseCase=useCase;pendingNativeResult=null;nativeStarted=false;pairFinished=false;packagingStarted=false;
            pairedRoomLabel=room.optString("label",room.optString("camera_id"));artifact=null;lastScanId=null;viewerStarting=true;
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);status.setText("Starting paired room capture…");captureStatus.setText("Starting room camera…");captureStats.setText(pairedRoomLabel);updateButtons();
            paired.start(captureDir,server.getText().toString().trim(),room.getString("camera_id"));
        }catch(Exception error){active=false;getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);finishCaptureViewer(dialog);fail("Could not start paired recording",error);}
    }
    @Override public void onPairReady(File directory,JSONObject reference){
        if(destroyed||viewerClosing||captureDialog==null){paired.stop("phone_start_cancelled");return;}
        try{
            viewerStarting=false;active=true;nativeStarted=true;fitPreview();status.setText("Starting phone video + IMU…");captureStatus.setText("Starting phone…");captureStats.setText(pairedRoomLabel+" recording");updateButtons();
            engine.start(directory,previewSurface,pendingCamera.getString("id"),pendingShortTest,pendingUseCase);
        }catch(Exception error){active=false;nativeStarted=false;paired.stop("native_phone_start_failed");fail("Phone recording could not start",error);}
    }
    @Override public void onPairState(String state,JSONObject reference){
        if(destroyed)return;
        if("recording".equals(state)&&active&&captureStats!=null)captureStats.setText("Phone + "+pairedRoomLabel+" + IMU");
        if("finalizing".equals(state)&&!active){status.setText("Finalizing paired recordings…");if(captureStatus!=null)captureStatus.setText("Saving both…");}
    }
    @Override public void onPairError(String reason,JSONObject reference){
        if(active)engine.stop();
        if(destroyed)return;status.setText("Paired capture needs attention");detail.setText(reason+" Original phone and room-camera files are retained.");
        if(captureStats!=null)captureStats.setText(reason);
    }
    @Override public void onPairStopped(File directory,JSONObject reference){
        pairFinished=true;
        if(destroyed)return;
        if(!nativeStarted){busy=false;getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);finishCaptureViewer(captureDialog);selectSaved(directory);status.setText("Paired capture did not start");detail.setText(reference.optString("error","Start was cancelled")+". The saved reference can be finalized again if needed.");updateButtons();return;}
        if(pendingNativeResult!=null&&!packagingStarted){packagingStarted=true;packageCapture(pendingNativeResult);}
    }
    @Override public void onState(String state,JSONObject details){
        if(destroyed)return;
        if("recording".equals(state)){
            status.setText("Recording 8K + motion • "+details.optLong("elapsed_seconds")+"s");
            detail.setText("Frames: "+details.optLong("encoded_frames")+"   Accel: "+details.optLong("accel_samples")+"   Gyro: "+details.optLong("gyro_samples")+". Walk slowly and keep this app open.");
            if(captureStatus!=null){long seconds=details.optLong("elapsed_seconds");captureStatus.setText(String.format(Locale.US,"● REC\n%02d:%02d",seconds/60,seconds%60));captureStats.setText("Phone + "+pairedRoomLabel+" + IMU\n"+details.optLong("encoded_frames")+" frames");}
        }
        else if("stopping".equals(state)||"finalizing".equals(state)){viewerClosing=captureDialog!=null;status.setText("Saving video and timing evidence…");if(captureStatus!=null)captureStatus.setText("Saving…");updateButtons();}
        else{status.setText(state);if(details.has("message"))detail.setText(details.optString("message"));if(captureStatus!=null)captureStatus.setText("Starting…");}
    }
    @Override public void onError(String message,JSONObject details){if(!destroyed){status.setText("Capture needs attention");detail.setText(message+" Raw files are retained on this phone.");if(captureStatus!=null)captureStatus.setText("Capture needs attention");}}
    @Override public void onStopped(JSONObject result){
        active=false;pendingNativeResult=result;paired.stop(PairedCapture.phoneStopReason(result));getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);finishCaptureViewer(captureDialog);
        if(destroyed)return;busy=true;status.setText("Saving phone data and finalizing the room recording…");updateButtons();
        if(pairFinished&&!packagingStarted){packagingStarted=true;packageCapture(result);}
    }
    private void packageCapture(JSONObject result){
        busy=false;
        File latestReport=new File(captureDir,"capabilities.json");if(latestReport.isFile())phoneReport=latestReport;
        if(!result.optBoolean("export_ready")){
            artifact=new File(captureDir,"capture_result.json");if(!artifact.isFile())artifact=null;
            status.setText("Capture could not finish");detail.setText(result.optString("stop_reason","Recording failed")+". Raw files are retained. Tap Upload phone report to send this attempt's diagnostic results.");
            saved.setText("Capture diagnostics: "+captureDir.getName());updateButtons();return;
        }
        busy=true;status.setText("Packaging the saved capture…");updateButtons();final File dir=captureDir;
        worker.execute(()->{try{
            BundleTools.writeCaptureManifest(this,dir,result);File archive=BundleTools.zip(dir);JSONObject reference=BundleTools.pairing(archive);
            ui.post(()->{if(destroyed)return;artifact=archive;busy=false;JSONObject timing=result.optJSONObject("timing");boolean match=timing!=null&&timing.optBoolean("exact_frame_association_verified");
                for(int i=0;i<cameras.length();i++){JSONObject c=cameras.optJSONObject(i);if(CapturePreflight.matches(result,c,Build.FINGERPRINT))try{c.put("recorded_preflight",CapturePreflight.reference(dir.getName(),result));}catch(JSONException ignored){}}
                boolean roomComplete=reference!=null&&"stopped".equals(reference.optString("server_status"))&&!reference.optBoolean("needs_recovery");
                status.setText(reference==null?(result.optBoolean("partial")?"Partial phone capture saved":"Phone capture saved"):!roomComplete?"Phone saved; room capture needs attention":result.optBoolean("partial")?"Partial paired capture saved":"Paired capture saved");
                detail.setText((match?"Encoded frames match camera sensor timestamps. RoomWalk will validate the uploaded evidence.":"Frame timing was not verified. Video and raw motion are retained for review.")+(reference==null?" Phone video and motion are retained for upload.":roomComplete?" Phone video, motion and the matching room capture are linked for upload.":" Room status: "+reference.optString("server_status","unknown")+". "+reference.optString("error","Use Finalize paired capture to retry.")+" All saved evidence is retained."));
                saved.setText(archive.getName()+"  •  "+android.text.format.Formatter.formatFileSize(this,archive.length())+"\nUpload or export; this local copy is retained.");updateButtons();
            });
        }catch(Exception e){ui.post(()->{if(destroyed)return;File f=new File(dir,"capabilities.json");if(f.isFile())artifact=f;});fail("Recording retained; packaging needs attention",e);}});
    }
    private JSONObject selectedRoomCamera(){
        if(roomCamera==null||server==null||!roomCamerasOrigin.equals(server.getText().toString().trim().replaceAll("/+$","")))return null;
        JSONObject row=roomCameras.optJSONObject(roomCamera.getSelectedItemPosition());return row!=null&&row.optBoolean("available")?row:null;
    }
    private void checkConnection(){
        final String destination=server.getText().toString().trim();getPreferences(0).edit().putString("server",destination).apply();busy=true;status.setText("Connecting and loading room cameras…");updateButtons();
        worker.execute(()->{try{
            JSONObject health=BundleTools.health(this,destination);if(!"ok".equals(health.optString("status")))throw new IOException("RoomWalk is not healthy");
            ui.post(()->{if(!destroyed)status.setText("Loading room cameras…");});
            Object value=BundleTools.requestJsonValue(this,destination,"/api/companion-captures/cameras",null,5000);if(!(value instanceof JSONObject))throw new IOException("Invalid room-camera inventory");
            JSONObject inventory=(JSONObject)value;JSONArray available=inventory.optJSONArray("cameras");final JSONArray rows=available==null?new JSONArray():available;
            ui.post(()->{if(destroyed)return;roomCameras=rows;roomCamerasOrigin=destination.replaceAll("/+$","");ArrayList<String> labels=new ArrayList<>();int selected=0;String prior=getPreferences(0).getString("room_camera_id","");
                for(int i=0;i<rows.length();i++){JSONObject row=rows.optJSONObject(i);labels.add(row.optString("label",row.optString("camera_id")));if(prior.equals(row.optString("camera_id")))selected=i;}
                roomCamera.setAdapter(new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,labels));if(!labels.isEmpty())roomCamera.setSelection(selected);
                busy=false;status.setText(rows.length()>0?"RoomWalk is ready for paired capture":"Room camera unavailable");roomCameraStatus.setText(rows.length()>0?"Select the room camera matching this walk. Record starts phone video + IMU and the room recording.":inventory.optString("reason","No active room-camera source is available"));
                JSONObject c=selectedCamera();detail.setText(c==null?"Check phone, choose the cameras, then open Capture.":cameraDescription(c));updateButtons();});
        }catch(Exception error){ui.post(()->{if(destroyed)return;roomCameras=new JSONArray();roomCamerasOrigin="";roomCameraStatus.setText(error.getMessage());});fail("RoomWalk connection needs attention",error);}});
    }
    private void finalizePair(){
        if(captureDir==null)return;final File directory=captureDir;busy=true;status.setText("Finalizing the selected paired recording…");updateButtons();
        worker.execute(()->{try{JSONObject reference=PairedCapture.finishSaved(this,directory);ui.post(()->{if(destroyed)return;busy=false;status.setText("Static recording "+reference.optString("server_status"));detail.setText("The exact room-camera session is retained. Package or upload the phone capture when ready.");updateButtons();});}catch(Exception error){fail("Paired finalization remains unconfirmed",error);}});
    }
    private void uploadPhoneReport(){
        if(phoneReport==null)return;final File file=phoneReport;final String destination=server.getText().toString();
        getPreferences(0).edit().putString("server",destination).apply();busy=true;status.setText("Uploading camera capability report…");updateButtons();
        worker.execute(()->{try{
            JSONObject receipt=BundleTools.uploadPhoneReport(this,destination,file);
            BundleTools.writeJson(new File(file.getParentFile(),"diagnostic_upload_receipt.json"),receipt);
            ui.post(()->{if(destroyed)return;busy=false;status.setText("Phone report uploaded");detail.setText("Report "+receipt.optString("id").substring(0,12)+" is on your RoomWalk server for camera-mode diagnosis.");updateButtons();});
        }catch(Exception e){fail("Phone report upload failed",e);}});
    }
    private void repackage(){
        if(captureDir==null)return;final File dir=captureDir;busy=true;status.setText("Checking the saved capture…");updateButtons();
        if(new File(dir,"imu_capture_manifest.json").isFile()){packageImu(dir);return;}
        worker.execute(()->{try{JSONObject result=BundleTools.readJson(new File(dir,"capture_result.json"));ui.post(()->{if(destroyed)return;busy=false;packageCapture(result);});}catch(Exception e){fail("Cannot package this capture",e);}});
    }
    private void upload(){
        if(artifact==null||transferActive)return;
        try{
            String destination=server.getText().toString().trim();getPreferences(0).edit().putString("server",destination).apply();
            UploadService.start(this,artifact,destination,new File(artifact.getParentFile(),"imu_capture_manifest.json").isFile());
            refreshTransfer();requestUploadNotifications();
        }catch(Exception error){fail("Could not start upload",error);}
    }
    private void requestUploadNotifications(){
        if(Build.VERSION.SDK_INT>=33&&checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)!=PackageManager.PERMISSION_GRANTED&&!getPreferences(0).getBoolean("upload_notifications_asked",false)){
            getPreferences(0).edit().putBoolean("upload_notifications_asked",true).apply();
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS},UPLOAD_NOTIFICATION_PERMISSION);
        }
    }
    private void cancelUpload(){try{UploadService.cancel(this);refreshTransfer();}catch(Exception error){fail("Could not cancel upload",error);}}
    private void retryUpload(){try{UploadService.retryLast(this);refreshTransfer();requestUploadNotifications();}catch(Exception error){fail("Could not retry upload",error);}}
    private void refreshTransfer(){
        try{
            JSONObject transfer=UploadService.snapshot(this);String phase=transfer.optString("state","idle");
            transferActive=transfer.optBoolean("active");transferPanel.setVisibility("idle".equals(phase)?View.GONE:View.VISIBLE);
            long sent=transfer.optLong("sent_bytes"),total=transfer.optLong("total_bytes");
            int fraction=(int)Math.min(1000,1000.0*sent/Math.max(1,total));
            transferProgress.setIndeterminate(transferActive&&!"uploading".equals(phase));transferProgress.setProgress(fraction);
            transferProgress.setVisibility(transferActive?View.VISIBLE:View.GONE);
            cancelUploadButton.setVisibility(transferActive?View.VISIBLE:View.GONE);cancelUploadButton.setEnabled(!"cancelling".equals(phase));
            retryUploadButton.setVisibility(transfer.optBoolean("can_retry")?View.VISIBLE:View.GONE);retryUploadButton.setEnabled(!active&&!imuActive&&!busy&&captureDialog==null);
            String title,description=transfer.optString("archive_name")+"\n";
            if("uploading".equals(phase)){
                title="Uploading • "+(fraction/10)+"%";
                double speed=transfer.optDouble("bytes_per_second");
                description+=android.text.format.Formatter.formatFileSize(this,sent)+" / "+android.text.format.Formatter.formatFileSize(this,total);
                if(speed>0){description+=String.format(Locale.US," • %.0f Mbps",speed*8/1_000_000);long seconds=Math.max(0,Math.round((total-sent)/speed));description+=" • about "+(seconds>=60?(seconds+59)/60+" min":seconds+" sec")+" left";}
                description+="\nYou can leave the app or turn off the screen.";
            }else if("validating".equals(phase)){title="Transfer finished • checking archive";description+="RoomWalk is importing and checking the recording. You can turn off the screen.";}
            else if("complete".equals(phase)){
                JSONObject receipt=transfer.optJSONObject("receipt");boolean pending=receipt!=null&&"pending".equals(receipt.optString("validation_status"));
                title=pending?"Transferred to RoomWalk":"Uploaded to RoomWalk";
                description+=pending?"The server has saved the archive and will validate it in the background. Open RoomWalk to follow processing. The original recording remains on this phone.":"Open RoomWalk to view the result. The original recording remains on this phone.";
                if(receipt!=null&&!transfer.optBoolean("imu"))lastScanId=receipt.optString("id",null);
            }
            else if("failed".equals(phase)||"interrupted".equals(phase)||"cancelled".equals(phase)){title="cancelled".equals(phase)?"Upload cancelled": "Upload needs attention";description+=transfer.optString("error", "Transfer did not finish.")+"\nThe saved archive is retained. Retry sends the same file.";}
            else{title="cancelling".equals(phase)?"Cancelling upload…":"finalizing".equals(phase)?"Finalizing paired recording…":"Connecting for upload…";description+="You can leave the app or turn off the screen.";}
            transferStatus.setText(title);transferDetail.setText(description);updateButtons();
        }catch(Exception error){transferStatus.setText("Upload status needs attention");transferDetail.setText(error.getMessage());}
    }
    private void export(){if(artifact==null)return;pendingExport=artifact;Intent intent=new Intent(Intent.ACTION_CREATE_DOCUMENT);intent.addCategory(Intent.CATEGORY_OPENABLE);intent.setType(artifact.getName().endsWith(".json")?"application/json":"application/zip");intent.putExtra(Intent.EXTRA_TITLE,artifact.getName());startActivityForResult(intent,EXPORT_DOCUMENT);}
    @Override protected void onActivityResult(int request,int result,Intent data){super.onActivityResult(request,result,data);if(request==EXPORT_DOCUMENT&&result==RESULT_OK&&data!=null&&data.getData()!=null&&pendingExport!=null){final Uri uri=data.getData();final File file=pendingExport;busy=true;updateButtons();worker.execute(()->{try(OutputStream out=getContentResolver().openOutputStream(uri)){if(out==null)throw new IOException("Export destination unavailable");BundleTools.copy(file,out);ui.post(()->{if(destroyed)return;busy=false;status.setText("Export saved");updateButtons();});}catch(Exception e){fail("Export failed",e);}});}}
    private void share(){if(artifact==null)return;Uri uri=new Uri.Builder().scheme("content").authority("org.noesis.roomwalk.files").appendPath(artifact.getParentFile().getName()).appendPath(artifact.getName()).build();Intent send=new Intent(Intent.ACTION_SEND);send.setType(artifact.getName().endsWith(".json")?"application/json":"application/zip");send.putExtra(Intent.EXTRA_STREAM,uri);send.setClipData(ClipData.newRawUri("RoomWalk capture",uri));send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);startActivity(Intent.createChooser(send,"Share RoomWalk capture"));}
    private void openRoomWalk(){try{BundleTools.endpoint(server.getText().toString(),"/");Intent open=new Intent(Intent.ACTION_VIEW,Uri.parse(server.getText().toString()));startActivity(open);}catch(Exception e){fail("Cannot open RoomWalk",e);}}
    private File[] savedDirectories(){File[] dirs=captureRoot().listFiles(File::isDirectory);if(dirs==null)return new File[0];Arrays.sort(dirs,(a,b)->Long.compare(b.lastModified(),a.lastModified()));return dirs.length>200?Arrays.copyOf(dirs,200):dirs;}
    private void loadLatest(){File[] dirs=savedDirectories();if(dirs.length>0)selectSaved(dirs[0]);}
    private void chooseSaved(){File[] dirs=savedDirectories();if(dirs.length==0){Toast.makeText(this,"No captures yet",Toast.LENGTH_SHORT).show();return;}String[] labels=new String[dirs.length];for(int i=0;i<dirs.length;i++)labels[i]=dirs[i].getName();new AlertDialog.Builder(this).setTitle("Saved captures and phone checks").setItems(labels,(d,index)->selectSaved(dirs[index])).setNegativeButton("Close",null).show();}
    private void selectSaved(File dir){captureDir=dir;File archive=new File(dir,"roomwalk-"+dir.getName()+".zip");File diagnostic=new File(dir,"capabilities.json");File imu=new File(dir,"imu_capture_manifest.json"),pair=new File(dir,PairedCapture.FILE);artifact=archive.isFile()?archive:imu.isFile()?imu:diagnostic.isFile()?diagnostic:pair.isFile()?pair:null;saved.setText(artifact!=null?artifact.getName():"Raw capture retained: "+dir.getName());if(uploadButton!=null)updateButtons();}
    private boolean hasFocusLock(){JSONObject c=selectedCamera();return c!=null&&getSharedPreferences("focus-locks",0).contains(c.optString("id"));}
    private void changeFocus(){
        JSONObject c=selectedCamera();if(c==null||active||focusBusy)return;
        final Dialog dialog=captureDialog;focusBusy=true;updateButtons();
        CameraPreview.FocusListener listener=(confirmed,message)->{if(destroyed||captureDialog!=dialog)return;focusBusy=false;captureStats.setText(message);if(confirmed)try{c.put("focus_control",FocusSettings.metadata(FocusSettings.read(this,c.optString("id"))));c.remove("recorded_preflight");captureStats.setText(message+". Run a new timing test for this focus mode.");}catch(Exception failure){captureStats.setText(failure.getMessage());}updateButtons();};
        if(hasFocusLock())previewController.unlockFocus(c.optString("id"),listener);else previewController.lockFocus(listener);
    }
    private void startImu(){
        if(active||imuActive||busy||captureDialog!=null)return;
        try{
            long minutes=Long.parseLong(imuMinutes.getText().toString());if(minutes<1||minutes>5)throw new IOException("Choose 1 to 5 minutes; one minute is the diagnostic default");
            getPreferences(0).edit().putString("imu_diagnostic_minutes",Long.toString(minutes)).apply();
            File dir=new File(captureRoot(),"imu-"+new SimpleDateFormat("yyyyMMdd-HHmmss",Locale.US).format(new Date())+"-"+UUID.randomUUID().toString().substring(0,8));
            if(!dir.mkdirs())throw new IOException("Cannot create IMU capture folder");
            captureDir=dir;artifact=null;lastScanId=null;imuActive=true;getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            status.setText("Starting IMU diagnostic…");imuProgress.setText("Starting motion sensors…");saved.setText("Raw motion capture: "+dir.getName());updateButtons();imuRecorder.start(dir,minutes*60);
        }catch(Exception error){fail("Could not start IMU diagnostic",error);}
    }
    private void stopImu(){if(!imuActive)return;busy=true;status.setText("Saving IMU diagnostic…");updateButtons();imuRecorder.stop("user_stopped");}
    @Override public void onImuState(String state,JSONObject details){
        if(destroyed)return;
        if("recording".equals(state)){long seconds=details.optLong("elapsed_seconds");status.setText(String.format(Locale.US,"IMU diagnostic • %d:%02d:%02d",seconds/3600,(seconds/60)%60,seconds%60));detail.setText("Keep the phone still and this app open. Accel: "+details.optLong("accel_samples")+"   Gyro: "+details.optLong("gyro_samples")+"   "+android.text.format.Formatter.formatFileSize(this,details.optLong("bytes")));imuProgress.setText(status.getText()+"\n"+detail.getText());}
        else{busy=true;status.setText("Saving IMU diagnostic…");imuProgress.setText("Saving motion data…");updateButtons();}
    }
    @Override public void onImuStopped(File directory,JSONObject manifest){
        imuActive=false;getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);if(destroyed)return;
        captureDir=directory;artifact=new File(directory,"imu_capture_manifest.json");busy=true;updateButtons();packageImu(directory);
    }
    @Override public void onImuError(String reason){
        imuActive=false;busy=false;getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);if(destroyed)return;
        status.setText("IMU diagnostic needs attention");detail.setText(reason+" Raw files are retained.");imuProgress.setText(reason+" Raw files are retained.");if(captureDir!=null)selectSaved(captureDir);updateButtons();
    }
    private void packageImu(File dir){
        busy=true;status.setText("Packaging IMU diagnostic…");updateButtons();
        worker.execute(()->{try{
            JSONObject manifest=BundleTools.readJson(new File(dir,"imu_capture_manifest.json"));File archive=BundleTools.zipImu(dir);
            ui.post(()->{if(destroyed)return;artifact=archive;busy=false;status.setText("complete".equals(manifest.optString("status"))?"IMU recording complete":"Partial IMU recording saved");
                detail.setText("Stop reason: "+manifest.optString("stop_reason")+". Upload or export the raw motion data for analysis; noise calibration remains unverified.");imuProgress.setText(status.getText()+" • "+Math.round(manifest.optDouble("actual_duration_s"))+" seconds\n"+detail.getText());
                saved.setText(archive.getName()+" • "+android.text.format.Formatter.formatFileSize(this,archive.length()));updateButtons();});
        }catch(Exception error){fail("IMU files retained; packaging needs attention",error);}});
    }
    private void fail(String title,Exception error){ui.post(()->{if(destroyed)return;busy=false;status.setText(title);detail.setText(error.getMessage()==null?error.toString():error.getMessage());updateButtons();});}
    @Override protected void onStart(){super.onStart();ui.removeCallbacks(transferPoll);ui.post(transferPoll);}
    @Override protected void onStop(){ui.removeCallbacks(transferPoll);super.onStop();}
    @Override protected void onPause(){if(imuActive){busy=true;imuRecorder.stop("app_backgrounded");}if(paired.isActive())paired.stop("app_backgrounded");if(captureDialog!=null||pendingCaptureViewer)requestCloseCaptureViewer();else if(active)engine.stop();super.onPause();}
    @Override protected void onDestroy(){destroyed=true;pendingCaptureViewer=false;ui.removeCallbacks(transferPoll);if(displays!=null)displays.unregisterDisplayListener(displayListener);engine.close();imuRecorder.close();paired.close();previewController.shutdown();finishCaptureViewer(captureDialog);worker.shutdown();super.onDestroy();}
}

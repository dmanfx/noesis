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
    private final Selection roomCamera=new Selection();
    private JSONArray roomCameras=new JSONArray();
    private String roomCamerasOrigin="";
    private final TextState roomCameraStatus=new TextState();
    private final ActionState finalizePairButton=new ActionState();
    private JSONObject pendingNativeResult,pendingCamera;
    private boolean pendingShortTest,nativeStarted,pairFinished,packagingStarted;
    private long pendingUseCase;
    private String pairedRoomLabel="room camera";
    private final TextState imuMinutes=new TextState();
    private final TextState imuProgress=new TextState();
    private final ActionState imuStartButton=new ActionState(),imuStopButton=new ActionState();
    private Button focusButton;
    private boolean imuActive,imuPreparing,focusBusy;
    private JSONObject imuTelemetry;
    private long imuPreparationDeadline,imuPreparationMinutes;
    private final Runnable imuPreparationTick=()->advanceImuPreparation();
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
    private final TextState status=new TextState(),detail=new TextState(),saved=new TextState(),server=new TextState();
    private TextView captureStatus,captureStats,captureGuide,recordPrerequisite;
    private Button captureChecksButton,previewRetryButton;
    private boolean showCaptureChecks;
    private JSONObject captureGuidance,captureOutcome;
    private boolean guidedStopRequested;
    private final Selection camera=new Selection();
    private Button captureBackButton,testButton,recordButton,stopButton;
    private final ActionState captureButton=new ActionState(),probeButton=new ActionState(),uploadButton=new ActionState(),exportButton=new ActionState(),shareButton=new ActionState(),historyButton=new ActionState(),checkButton=new ActionState(),packageButton=new ActionState(),reportButton=new ActionState();
    private WebShell shell;
    private JSONObject transferState=new JSONObject(),calibrationRequest,walkIntent;
    private String captureMode="reconstruction";
    private String walkIntentStatus="none";
    private static final class TextState {
        private String value="";void setText(CharSequence text){value=text==null?"":text.toString();}String getText(){return value;}void setEnabled(boolean ignored){}
    }
    private static final class Selection {
        private int selected=-1,size;void setSize(int size){this.size=size;selected=size>0?0:-1;}void setSelection(int index){if(index < -1||index>=size)throw new IllegalArgumentException("Camera selection is out of range");selected=index;}int getSelectedItemPosition(){return selected;}void setEnabled(boolean ignored){}
    }
    private static final class ActionState {
        private boolean enabled;void setEnabled(boolean value){enabled=value;}boolean isEnabled(){return enabled;}void setVisibility(int ignored){}
    }
    private JSONArray cameras = new JSONArray();
    private JSONObject capabilityReport;
    private File captureDir, artifact, pendingExport, phoneReport;
    private boolean active=false, busy=false, destroyed=false;
    private boolean previewReady=false, viewerStarting=false, viewerClosing=false;
    private boolean pendingCaptureViewer,transferActive;
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
        server.setText(getPreferences(0).getString("server",defaultServer()));
        imuMinutes.setText(getPreferences(0).getString("imu_diagnostic_minutes","1"));
        status.setText("Welcome to RoomWalk");
        detail.setText("Native phone video and motion for reconstruction coverage or paired path refinement. Optional calibration remains available.");
        try{
            shell=new WebShell(this,new WebShell.Host(){
                public void command(String action,JSONObject args)throws Exception{nativeCommand(action,args);}
                public void error(String message){status.setText("RoomWalk needs attention");detail.setText(message);updateButtons();}
            },server.getText());
            FrameLayout root=new FrameLayout(this);root.setBackgroundColor(Color.rgb(245,247,246));
            root.addView(shell.view,new FrameLayout.LayoutParams(-1,-1));
            root.setOnApplyWindowInsetsListener((v,insets)->{
                if(Build.VERSION.SDK_INT>=30){android.graphics.Insets bars=insets.getInsets(WindowInsets.Type.systemBars()|WindowInsets.Type.displayCutout());v.setPadding(bars.left,bars.top,bars.right,bars.bottom);}
                else v.setPadding(insets.getSystemWindowInsetLeft(),insets.getSystemWindowInsetTop(),insets.getSystemWindowInsetRight(),insets.getSystemWindowInsetBottom());
                return insets;
            });
            setContentView(root);root.requestApplyInsets();
            try{
                String savedIntent=getPreferences(0).getString("walk_intent",null);
                if(savedIntent!=null){walkIntent=WalkIntent.validate(new JSONObject(savedIntent));captureMode=walkIntent.optString("mode",WalkIntent.RECONSTRUCTION);walkIntentStatus="selected";}
                else calibrationRequest=new JSONObject(getPreferences(0).getString("calibration_request","{}"));
            }catch(Exception ignored){walkIntent=null;captureMode=WalkIntent.RECONSTRUCTION;}
            loadLatest();refreshSavedList();updateButtons();
        }catch(Exception error){TextView failure=text("RoomWalk could not open: "+error.getMessage(),16,INK);setContentView(failure);}
    }


    /** State and action gates are shared by the web presentation and native recorder. */
    private JSONObject actions()throws Exception{
        return new JSONObject().put("checkPhone",probeButton.isEnabled()).put("configure",checkButton.isEnabled())
            .put("checkConnection",checkButton.isEnabled()).put("capture",captureButton.isEnabled())
            .put("upload",uploadButton.isEnabled()).put("export",exportButton.isEnabled()).put("share",shareButton.isEnabled())
            .put("selectSaved",historyButton.isEnabled()).put("repackage",packageButton.isEnabled())
            .put("finalizePair",finalizePairButton.isEnabled()).put("uploadPhoneReport",reportButton.isEnabled())
            .put("startImu",imuStartButton.isEnabled()).put("stopImu",imuStopButton.isEnabled())
            .put("cancelUpload",transferActive).put("retryUpload",!active&&!imuActive&&!busy&&captureDialog==null&&transferState.optBoolean("can_retry"));
    }
    private JSONArray savedCaptures=new JSONArray();
    private void refreshSavedList(){
        JSONArray rows=new JSONArray();
        try{for(File dir:savedDirectories()){
            File zip=new File(dir,"roomwalk-"+dir.getName()+".zip");
            JSONObject row=new JSONObject().put("id",dir.getName()).put("name",dir.getName()).put("bytes",zip.isFile()?zip.length():0)
                .put("kind",new File(dir,"calibration_request.json").isFile()?"calibration":new File(dir,"imu_capture_manifest.json").isFile()?"imu_diagnostic":new File(dir,"capture_result.json").isFile()?"walk":"phone_report")
                .put("upload_ready",zip.isFile());
            File intentFile=new File(dir,"walk_intent.json");
            if(intentFile.isFile())try{
                JSONObject intent=WalkIntent.validate(BundleTools.readJson(intentFile));
                row.put("walk_intent",intent).put("walk_mode",intent.optString("mode"))
                    .put("target_scan_id",intent.isNull("target_scan_id")?JSONObject.NULL:intent.optString("target_scan_id"))
                    .put("carry_protocol",intent.optString("carry_protocol"))
                    .put("accuracy_target_m",intent.optDouble("accuracy_target_m"));
            }catch(Exception error){row.put("metadata_error","Walk intent needs review");}
            File request=new File(dir,"calibration_request.json");
            if(request.isFile())try{if(request.length()>32768)throw new IOException("Calibration settings exceed their bound");JSONObject metadata=BundleTools.readJson(request);row.put("calibration_mode",metadata.optString("mode")).put("short_test",metadata.optBoolean("short_test"));}catch(Exception error){row.put("metadata_error","Calibration settings need review");}
            rows.put(row);
        }}catch(Exception ignored){}savedCaptures=rows;
    }
    private void publishNativeState(){
        if(shell==null||destroyed)return;
        try{
            JSONArray rows=new JSONArray();
            for(int i=0;i<cameras.length();i++){
                JSONObject source=cameras.getJSONObject(i);
                JSONObject row=new JSONObject().put("id",source.optString("id")).put("label",source.optString("label","Camera "+source.optString("id")))
                    .put("full_walk_available",fullWalkAvailable(source)).put("test_available",standardTestUseCase(source)>=0)
                    .put("focus_locked",getSharedPreferences("focus-locks",0).contains(source.optString("id")));
                try{row.put("focus_control",FocusSettings.metadata(FocusSettings.read(this,source.optString("id"))));}catch(Exception error){row.put("focus_control",new JSONObject().put("mode","invalid")).put("focus_error",error.getMessage());}
                rows.put(row);
            }
            JSONObject state=new JSONObject().put("schema","roomwalk.native_state.v1").put("server",server.getText())
                .put("status",status.getText()).put("detail",detail.getText()).put("cameras",rows)
                .put("camera_index",camera.getSelectedItemPosition()).put("room_cameras",roomCameras)
                .put("room_camera_index",roomCamera.getSelectedItemPosition()).put("room_camera_status",roomCameraStatus.getText())
                .put("active",active).put("busy",busy||imuPreparing||captureDialog!=null||pendingCaptureViewer).put("imu_active",imuActive).put("imu_preparing",imuPreparing)
                .put("imu_progress",imuProgress.getText()).put("imu_minutes",imuMinutes.getText())
                .put("supports_target_reference",true)
                .put("enabled",actions()).put("saved",savedCaptures).put("saved_description",saved.getText())
                .put("selected_capture",captureDir==null?JSONObject.NULL:captureDir.getName()).put("capture_mode",captureMode).put("capture_short_test",pendingShortTest)
                .put("walk_intent",walkIntent==null?JSONObject.NULL:walkIntent).put("walk_intent_status",walkIntentStatus)
                .put("artifact",artifact==null?JSONObject.NULL:new JSONObject().put("name",artifact.getName()).put("bytes",artifact.length()))
                .put("transfer",transferState).put("last_scan_id",lastScanId==null?JSONObject.NULL:lastScanId);
            if(calibrationRequest!=null)state.put("calibration_request",calibrationRequest);
            if(captureGuidance!=null)state.put("capture_guidance",captureGuidance);
            if(captureOutcome!=null)state.put("capture_outcome",captureOutcome);
            if(imuTelemetry!=null)state.put("imu_telemetry",imuTelemetry);
            shell.publish(state);
        }catch(Exception ignored){}
    }
    private void nativeCommand(String action,JSONObject args)throws Exception{
        if(destroyed)return;
        if("snapshot".equals(action)){refreshSavedList();publishNativeState();return;}
        if(!actions().optBoolean(action))throw new IOException("This action is unavailable while RoomWalk is busy or its prerequisites are missing");
        switch(action){
            case "checkPhone":checkPhone();break;
            case "checkConnection":checkConnection();break;
            case "configure":
                String destination=args.optString("server",server.getText()).trim().replaceAll("/+$","");
                BundleTools.endpoint(destination,"/");
                if(args.has("camera_index")){camera.setSelection(args.getInt("camera_index"));if(selectedCamera()!=null)detail.setText(cameraDescription(selectedCamera()));}
                if(args.has("room_camera_index")){roomCamera.setSelection(args.getInt("room_camera_index"));JSONObject room=selectedRoomCamera();getPreferences(0).edit().putString("room_camera_id",room==null?"":room.optString("camera_id")).apply();}
                if(args.has("imu_minutes")){int minutes=args.getInt("imu_minutes");if(minutes<1||minutes>180)throw new IOException("Choose 1–180 minutes for diagnostics or stationary noise");imuMinutes.setText(Integer.toString(minutes));}
                boolean changed=!destination.equals(server.getText());server.setText(destination);getPreferences(0).edit().putString("server",destination).apply();
                if(changed){roomCameras=new JSONArray();roomCamerasOrigin="";shell.connect(destination);}
                checkConnection();break;
            case "capture":
                String requestedMode=args.optString("mode",WalkIntent.RECONSTRUCTION);
                if(WalkIntent.isWalkMode(requestedMode)){
                    String normalized=WalkIntent.normalizeMode(requestedMode);
                    JSONObject targetReference=null;
                    if(args.has("target_reference")){
                        Object rawReference=args.opt("target_reference");
                        if(!(rawReference instanceof JSONObject))throw new IOException("The retained PCF selection must be an object");
                        targetReference=(JSONObject)rawReference;
                    }
                    JSONObject requestedIntent=WalkIntent.create(normalized,args.has("target_scan_id")?args.optString("target_scan_id",null):null,targetReference);
                    if(WalkIntent.requiresPair(normalized)&&selectedRoomCamera()==null)
                        throw new IOException("Path refinement requires an available paired room camera");
                    captureMode=normalized;walkIntent=requestedIntent;walkIntentStatus="selected";
                    getPreferences(0).edit().putString("walk_intent",walkIntent.toString()).apply();
                    calibrationRequest=null;
                }else if(Arrays.asList("camera","imu").contains(requestedMode)){
                    captureMode=requestedMode;walkIntent=null;walkIntentStatus="none";calibrationRequest=calibrationRequest(requestedMode,args);
                    getPreferences(0).edit().remove("walk_intent").putString("calibration_request",calibrationRequest.toString()).apply();
                }else throw new IOException("Choose reconstruction, path refinement, camera calibration or camera–IMU calibration");
                imuTelemetry=null;
                openCaptureViewer();break;
            case "upload":upload();break;
            case "export":export();break;
            case "share":share();break;
            case "selectSaved":
                String id=args.getString("id");File selected=null;for(File dir:savedDirectories())if(dir.getName().equals(id)){selected=dir;break;}
                if(selected==null)throw new IOException("Saved capture was not found");selectSaved(selected);break;
            case "repackage":repackage();break;
            case "finalizePair":finalizePair();break;
            case "uploadPhoneReport":uploadPhoneReport();break;
            case "startImu":
                int minutes=args.optInt("minutes",1);if(minutes<1||minutes>180)throw new IOException("Choose 1–180 minutes");imuMinutes.setText(Integer.toString(minutes));startImu();break;
            case "stopImu":stopImu();break;
            case "cancelUpload":cancelUpload();break;
            case "retryUpload":retryUpload();break;
            default:throw new IOException("Unknown RoomWalk action");
        }
        updateButtons();
    }
    private JSONObject calibrationRequest(String mode,JSONObject args)throws Exception{
        JSONObject board=args.optJSONObject("board");
        if(board==null){board=new JSONObject().put("squares_x",10).put("squares_y",14).put("square_length_m",0.018).put("marker_length_m",0.0132).put("dictionary","DICT_4X4_1000").put("legacy_pattern",false);
            JSONArray ids=new JSONArray();for(int id=300;id<370;id++)ids.put(id);board.put("marker_ids",ids);}
        int x=board.getInt("squares_x"),y=board.getInt("squares_y");double square=board.getDouble("square_length_m"),marker=board.getDouble("marker_length_m");
        if(x<3||y<3||x>30||y>30||!Double.isFinite(square)||!Double.isFinite(marker)||square<=0||square>0.2||square*Math.max(x,y)>3||marker<0.001||marker>=square)throw new IOException("Board dimensions or square/marker sizes are invalid");
        String dictionary=board.getString("dictionary");if(!dictionary.matches("DICT_[4567]X[4567]_(50|100|250|1000)"))throw new IOException("Choose a supported ArUco dictionary");
        int capacity=Integer.parseInt(dictionary.substring(dictionary.lastIndexOf('_')+1));
        JSONArray ids=board.getJSONArray("marker_ids");if(ids.length()!=x*y/2)throw new IOException("The board needs "+(x*y/2)+" marker IDs");
        HashSet<Integer> unique=new HashSet<>();for(int i=0;i<ids.length();i++){int id=ids.getInt(i);if(id<0||id>=capacity||!unique.add(id))throw new IOException("Marker IDs must be unique and fit the selected dictionary");}
        JSONObject request=new JSONObject().put("schema","roomwalk.calibration_request.v1").put("mode",mode).put("board",board)
            .put("board_geometry_confirmed",args.optBoolean("board_geometry_confirmed",false));
        if(args.has("camera_calibration_id"))request.put("camera_calibration_id",args.getString("camera_calibration_id"));
        if(args.has("noise_calibration_id"))request.put("noise_calibration_id",args.getString("noise_calibration_id"));
        if("imu".equals(mode)){
            if(request.optString("camera_calibration_id").trim().isEmpty())throw new IOException("Choose a qualified camera calibration before motion capture");
            if(!request.optBoolean("board_geometry_confirmed"))throw new IOException("Confirm the measured board dimensions before motion capture");
            JSONObject guard=args.optJSONObject("calibration_focus_guard");
            if(guard==null||!"manual_locked".equals(guard.optString("mode")))throw new IOException("Keep the calibrated focus locked, then refresh phone status before motion capture");
            request.put("calibration_focus_guard",guard);
        }
        return request;
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
    private boolean fullWalkAvailable(JSONObject c){return c!=null&&(c.optBoolean("supported8k")||CapturePreflight.qualifiesWalk(c,standardTestUseCase(c),calibrationCapture()));}
    private boolean calibrationCapture(){return "camera".equals(captureMode)||"imu".equals(captureMode);}
    private String cameraDescription(JSONObject c){
        if(c.optBoolean("supported8k"))return "Rear camera "+c.optString("id")+" reports an 8K hardware recording path. Run the timing test to check actual recording.";
        if(fullWalkAvailable(c))return "This camera passed a saved 8K/30 timing test with the same recording configuration. Record walk is ready. Phone video and raw motion are retained together.";
        if(standardTestUseCase(c)>=0)return "The driver accepts this camera's standard 8K/30 configuration. Run the 10-second test, then upload the saved capture for video and timing checks. Full walks remain disabled for this newly found mode.";
        return c.optString("reason","8K recording is unavailable for this camera.");
    }
    private void updateButtons(){
        JSONObject c=selectedCamera();boolean homeReady=!active&&!imuActive&&!imuPreparing&&!busy&&captureDialog==null&&!pendingCaptureViewer;
        boolean previewAvailable=preview!=null&&preview.isAvailable()&&previewReady;
        boolean canStart=!active&&!imuActive&&!focusBusy&&!busy&&!viewerStarting&&!viewerClosing&&previewAvailable&&(!WalkIntent.requiresPair(captureMode)||selectedRoomCamera()!=null);
        captureButton.setEnabled(homeReady&&c!=null&&c.optInt("preview_width")>0&&c.optInt("preview_height")>0);
        if(testButton!=null){
            testButton.setEnabled(canStart&&(fullWalkAvailable(c)||standardTestUseCase(c)>=0));
            recordButton.setEnabled(canStart&&fullWalkAvailable(c)&&(!calibrationCapture()||hasFocusLock()));
            recordButton.setVisibility(active?View.GONE:View.VISIBLE);testButton.setVisibility(active||(fullWalkAvailable(c)&&!showCaptureChecks)?View.GONE:View.VISIBLE);
            if(recordPrerequisite!=null){String reason=calibrationCapture()&&!hasFocusLock()?"Lock focus on the board first.":!fullWalkAvailable(c)&&standardTestUseCase(c)>=0?(calibrationCapture()?"Run the 10-second test with this focus setting. After it passes, reopen this step to record.":"Run the 10-second recording test for this camera mode, then record your walk."):WalkIntent.requiresPair(captureMode)&&selectedRoomCamera()==null?"Choose the paired room camera before recording path refinement.":"";recordPrerequisite.setText(reason);recordPrerequisite.setVisibility(!active&&!reason.isEmpty()?View.VISIBLE:View.GONE);}
            if(captureChecksButton!=null){captureChecksButton.setVisibility(!active&&fullWalkAvailable(c)?View.VISIBLE:View.GONE);captureChecksButton.setText(showCaptureChecks?"Hide extra check":"Extra check");}
            stopButton.setVisibility(active?View.VISIBLE:View.GONE);stopButton.setEnabled(active&&!viewerClosing);
            captureBackButton.setEnabled(!viewerClosing);captureBackButton.setText(active?"Save & close":"Close");
            for(Button b:new Button[]{recordButton,testButton,stopButton,captureBackButton})b.setAlpha(b.isEnabled()?1f:0.38f);
        }
        if(focusButton!=null){boolean locked=hasFocusLock();focusButton.setVisibility(calibrationCapture()?View.VISIBLE:View.GONE);focusButton.setText(locked?"Unlock focus":"Lock focus");focusButton.setEnabled(calibrationCapture()&&!active&&!viewerStarting&&!viewerClosing&&!focusBusy&&(locked||previewAvailable));focusButton.setAlpha(focusButton.isEnabled()?1f:0.38f);}
        if(previewRetryButton!=null){previewRetryButton.setVisibility(!active&&!previewReady&&!viewerStarting?View.VISIBLE:View.GONE);previewRetryButton.setEnabled(!active&&!busy&&!viewerStarting&&!viewerClosing&&!focusBusy);}
        imuStartButton.setEnabled(homeReady);imuMinutes.setEnabled(homeReady);imuStopButton.setEnabled((imuActive||imuPreparing)&&!busy);
        imuStopButton.setVisibility(imuActive||imuPreparing?View.VISIBLE:View.GONE);
        if(roomCamera!=null)roomCamera.setEnabled(homeReady);
        if(finalizePairButton!=null){boolean present=captureDir!=null&&new File(captureDir,PairedCapture.FILE).isFile();finalizePairButton.setVisibility(present?View.VISIBLE:View.GONE);finalizePairButton.setEnabled(homeReady&&present);}
        probeButton.setEnabled(homeReady);camera.setEnabled(homeReady);checkButton.setEnabled(homeReady);server.setEnabled(homeReady);historyButton.setEnabled(homeReady);
        reportButton.setEnabled(homeReady&&phoneReport!=null&&phoneReport.isFile());
        uploadButton.setEnabled(homeReady&&!transferActive&&artifact!=null&&artifact.getName().endsWith(".zip"));exportButton.setEnabled(homeReady&&artifact!=null);shareButton.setEnabled(homeReady&&artifact!=null);
        packageButton.setEnabled(homeReady&&!transferActive&&captureDir!=null&&(new File(captureDir,"capture_result.json").isFile()||new File(captureDir,"imu_capture_manifest.json").isFile()));
        publishNativeState();
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
        String targetScanId=walkIntent==null||walkIntent.isNull("target_scan_id")?null:walkIntent.optString("target_scan_id",null);
        CaptureGuide.Step initial=CaptureGuide.at(captureMode,0,targetScanId);
        captureGuide=text(initial.title+"\n"+initial.instruction,16,Color.WHITE);
        captureGuide.setPadding(dp(14),dp(8),dp(14),dp(10));captureGuide.setBackgroundColor(0xc0182a2a);
        FrameLayout.LayoutParams guidanceLayout=new FrameLayout.LayoutParams(-1,-2,Gravity.BOTTOM);
        previewBox.addView(captureGuide,guidanceLayout);
        viewer.addView(previewBox,new LinearLayout.LayoutParams(0,-1,1));
        LinearLayout rail=column();rail.setPadding(dp(12),dp(12),dp(12),dp(12));rail.setBackgroundColor(Color.rgb(17,27,27));
        ScrollView railScroll=new ScrollView(this);railScroll.setFillViewport(true);railScroll.addView(rail);viewer.addView(railScroll,new LinearLayout.LayoutParams(dp(144),-1));
        TextView mode=text(WalkIntent.RECONSTRUCTION.equals(captureMode)?"RECONSTRUCTION":"path_refinement".equals(captureMode)?"PATH REFINEMENT":"camera".equals(captureMode)?"CAMERA CALIBRATION":"CAMERA + IMU CALIBRATION",13,Color.WHITE);mode.setGravity(Gravity.CENTER);rail.addView(mode);
        captureStatus=text("Opening camera…",13,Color.WHITE);captureStatus.setGravity(Gravity.CENTER);rail.addView(captureStatus);
        captureStats=text(calibrationCapture()?"Board + video + IMU":WalkIntent.requiresPair(captureMode)?"Paired room camera + video + IMU":"Video + IMU",11,Color.LTGRAY);captureStats.setGravity(Gravity.CENTER);rail.addView(captureStats);
        rail.addView(new View(this),new LinearLayout.LayoutParams(1,0,1));
        focusButton=viewerButton(hasFocusLock()?"Unlock focus":"Lock focus",Color.rgb(49,68,68),()->changeFocus());focusButton.setVisibility(calibrationCapture()?View.VISIBLE:View.GONE);rail.addView(focusButton);
        previewRetryButton=viewerButton("Restart preview",Color.rgb(49,68,68),()->restartPreview());rail.addView(previewRetryButton);
        int guideTarget=CaptureGuide.targetSeconds(captureMode);
        recordButton=viewerButton(guideTarget>0?"Record · "+guideTarget+" sec":"Record",GREEN,()->startCapture(false));rail.addView(recordButton);
        recordPrerequisite=text("",11,Color.rgb(255,215,150));rail.addView(recordPrerequisite);
        showCaptureChecks=false;captureChecksButton=viewerButton("Extra check",Color.rgb(49,68,68),()->{showCaptureChecks=!showCaptureChecks;updateButtons();});rail.addView(captureChecksButton);
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
                openCurrentPreview(dialog,c);updateButtons();
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
    private void openCurrentPreview(Dialog dialog,JSONObject c){
        previewController.open(c.optString("id"),previewSurface,new CameraPreview.Listener(){
            public void onReady(){if(destroyed||captureDialog!=dialog||viewerClosing||viewerStarting)return;previewReady=true;captureStatus.setText(calibrationCapture()&&hasFocusLock()?"Lens + focus locked":"Ready");captureStats.setText(calibrationCapture()?"Keep the board visible. At fixed focus, nearer/farther objects may blur; the lens should not switch.":WalkIntent.requiresPair(captureMode)?selectedRoomCamera()==null?"Choose the paired room camera":"Phone + paired room camera + IMU":fullWalkAvailable(c)?"Phone video + IMU":standardTestUseCase(c)>=0?"Run a short test":"8K recording unavailable");updateButtons();}
            public void onError(String message){if(destroyed||captureDialog!=dialog||viewerClosing||viewerStarting)return;previewReady=false;captureStatus.setText("Preview stopped");captureStats.setText(message+" Restart preview here; no app reload is needed.");updateButtons();}
        },calibrationCapture());
    }
    private void restartPreview(){
        final Dialog dialog=captureDialog;JSONObject c=selectedCamera();
        if(dialog==null||c==null||active||busy||focusBusy||viewerStarting||viewerClosing||previewSurface==null||!previewSurface.isValid())return;
        viewerStarting=true;previewReady=false;captureStatus.setText("Restarting preview…");updateButtons();
        previewController.close((confirmed,reason)->{
            if(destroyed||captureDialog!=dialog)return;
            viewerStarting=false;
            if(viewerClosing){finishCaptureViewer(dialog);return;}
            if(confirmed)openCurrentPreview(dialog,c);
            else{captureStatus.setText("Preview release unconfirmed");captureStats.setText(reason);}
            updateButtons();
        });
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
        preview=null;previewBox=null;captureGuide=null;captureGuidance=null;focusButton=null;focusBusy=false;testButton=null;recordButton=null;recordPrerequisite=null;captureChecksButton=null;previewRetryButton=null;stopButton=null;captureBackButton=null;captureStatus=null;captureStats=null;
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
                camera.setSize(labels.size());if(!labels.isEmpty())camera.setSelection(select);
                phoneReport=f;artifact=f;captureDir=dir;saved.setText("Latest phone report selected. Upload it for camera-mode diagnosis. Recordings remain in Saved captures.");
                status.setText(fullWalkAvailable(selectedCamera())?"Phone check complete — ready to record":found?"Phone check complete — run the short test":"8K camera-mode check complete");
                JSONObject selected=selectedCamera();detail.setText(selected==null?report.optString("error","No camera entries returned."):cameraDescription(selected));
                busy=false;fitPreview();updateButtons();checkConnection();
            });
        }catch(Exception e){fail("Phone check failed",e);}});
    }
    @Override public void onRequestPermissionsResult(int request,String[] permissions,int[] grants){super.onRequestPermissionsResult(request,permissions,grants);if(request==CAMERA_PERMISSION){if(grants.length>0&&grants[0]==PackageManager.PERMISSION_GRANTED)checkPhone();else{status.setText("Camera permission is required");detail.setText("Allow camera access in Android app settings, then check the phone again.");}}}
    private void startCapture(boolean shortTest){
        JSONObject c=selectedCamera();
        if(c==null||preview==null||!preview.isAvailable()||!previewReady||active||busy||viewerStarting||viewerClosing)return;
        if(WalkIntent.requiresPair(captureMode)){
            JSONObject room=selectedRoomCamera();
            if(room==null){captureStats.setText("Choose the paired room camera before recording path refinement");return;}
            if(!pathReferenceMatchesRoomCamera(room)){captureStats.setText("The selected PCF is bound to a different room camera. Choose that camera before recording.");return;}
        }
        if(!shortTest&&calibrationCapture()&&!hasFocusLock()){captureStats.setText("Lock focus on the board before recording calibration");return;}
        if("imu".equals(captureMode))try{
            if(!FocusSettings.matchesMotionGuard(calibrationRequest,FocusSettings.metadata(FocusSettings.read(this,c.optString("id"))))){captureStats.setText("Focus changed after opening motion capture. Close the viewer and choose a camera calibration matching the current lens/focus. No recording started.");return;}
        }catch(Exception error){captureStats.setText("Cannot verify the motion focus lock: "+error.getMessage());return;}
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
            JSONObject room=selectedRoomCamera();
            if(WalkIntent.requiresPair(captureMode)&&room==null)throw new IOException("Choose an available room camera before path refinement");
            if(WalkIntent.requiresPair(captureMode)&&!pathReferenceMatchesRoomCamera(room))throw new IOException("The selected PCF is bound to a different room camera");
            String id=new SimpleDateFormat("yyyyMMdd-HHmmss",Locale.US).format(new Date())+"-"+UUID.randomUUID().toString().substring(0,8);
            captureDir=new File(captureRoot(),id);if(!captureDir.mkdirs())throw new IOException("Cannot create capture folder");
            pendingCamera=c;pendingShortTest=shortTest;pendingUseCase=useCase;pendingNativeResult=null;captureOutcome=null;nativeStarted=false;pairFinished=false;packagingStarted=false;guidedStopRequested=false;captureGuidance=null;
            if(calibrationCapture()){
                if(calibrationRequest==null)throw new IOException("Calibration board configuration is missing");
                JSONObject request=new JSONObject(calibrationRequest.toString());request.put("capture_id",id).put("short_test",shortTest);
                request.put("capture_guidance",new JSONObject().put("protocol","roomwalk.fixed_board_short_capture.v1").put("board_stationary",true).put("moving_device","phone").put("target_seconds",shortTest?10:CaptureGuide.targetSeconds(captureMode)));
                BundleTools.writeJson(new File(captureDir,"calibration_request.json"),request);
                pairFinished=true;pairedRoomLabel="camera".equals(captureMode)?"Camera calibration":"Camera–IMU calibration";artifact=null;lastScanId=null;viewerStarting=true;
                getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                onPairReady(captureDir,new JSONObject());return;
            }
            if(walkIntent==null)throw new IOException("Walk intent is missing; choose reconstruction or path refinement again");
            WalkIntent.validate(walkIntent);
            BundleTools.writeJson(new File(captureDir,"walk_intent.json"),walkIntent);
            pairedRoomLabel=room==null?"unpaired reconstruction":room.optString("label",room.optString("camera_id"));artifact=null;lastScanId=null;viewerStarting=true;
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);status.setText(room==null?"Starting reconstruction capture…":"Starting paired room capture…");captureStatus.setText(room==null?"Starting phone…":"Starting room camera…");captureStats.setText(pairedRoomLabel);updateButtons();
            if(room==null){pairFinished=true;onPairReady(captureDir,new JSONObject().put("paired",false));}
            else paired.start(captureDir,server.getText().toString().trim(),room.getString("camera_id"));
        }catch(Exception error){active=false;getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);finishCaptureViewer(dialog);fail("Could not start recording",error);}
    }
    @Override public void onPairReady(File directory,JSONObject reference){
        if(destroyed||viewerClosing||captureDialog==null){paired.stop("phone_start_cancelled");return;}
        try{
            viewerStarting=false;active=true;nativeStarted=true;fitPreview();status.setText("Starting phone video + IMU…");captureStatus.setText("Starting phone…");captureStats.setText(pairedRoomLabel+" recording");updateButtons();
            engine.start(directory,previewSurface,pendingCamera.getString("id"),pendingShortTest,pendingUseCase,calibrationCapture());
        }catch(Exception error){active=false;nativeStarted=false;paired.stop("native_phone_start_failed");if(calibrationCapture()||!WalkIntent.requiresPair(captureMode)){getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);finishCaptureViewer(captureDialog);refreshSavedList();}fail("Phone recording could not start",error);}
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
            long seconds=details.optLong("elapsed_seconds");
            status.setText("Recording "+captureMode+" • "+seconds+"s");
            detail.setText("Frames: "+details.optLong("encoded_frames")+"   Accel: "+details.optLong("accel_samples")+"   Gyro: "+details.optLong("gyro_samples")+". Keep RoomWalk open and preserve the guided route.");
            if(captureStatus!=null){captureStatus.setText(String.format(Locale.US,"● REC\n%02d:%02d",seconds/60,seconds%60));captureStats.setText("Phone + "+pairedRoomLabel+" + IMU\n"+details.optLong("encoded_frames")+" frames");}
            updateCaptureGuide(seconds);
        }
        else if("stopping".equals(state)||"finalizing".equals(state)){viewerClosing=captureDialog!=null;status.setText("Saving video and timing evidence…");if(captureStatus!=null)captureStatus.setText("Saving…");updateButtons();}
        else{status.setText(state);if(details.has("message"))detail.setText(details.optString("message"));if(captureStatus!=null)captureStatus.setText("Starting…");}
    }
    private void updateCaptureGuide(long seconds){
        if(pendingShortTest){if(captureGuide!=null)captureGuide.setText("10-second timing test · Keep the phone steady. This checks recording and timestamps; it is not a board calibration or walk-accuracy result.");return;}
        String targetScanId=walkIntent==null||walkIntent.isNull("target_scan_id")?null:walkIntent.optString("target_scan_id",null);
        CaptureGuide.Step step=CaptureGuide.at(captureMode,seconds,targetScanId);int target=CaptureGuide.targetSeconds(captureMode);
        if(captureGuide!=null)captureGuide.setText(step.title+"\n"+step.instruction);
        try{captureGuidance=new JSONObject().put("mode",captureMode).put("phase",step.phase).put("title",step.title).put("instruction",step.instruction)
            .put("elapsed_seconds",seconds).put("target_seconds",target).put("stationary",step.stationary).put("completed",step.completed)
                .put("target_scan_id",targetScanId==null?JSONObject.NULL:targetScanId)
                .put("progress",target>0?Math.min(1.0,seconds/(double)target):0.0);
        }catch(JSONException ignored){}
        publishNativeState();
        if(step.completed&&active&&!guidedStopRequested){guidedStopRequested=true;engine.stop();}
    }
    @Override public void onError(String message,JSONObject details){if(!destroyed){status.setText("Capture needs attention");detail.setText(message+" Raw files are retained on this phone.");if(captureStatus!=null)captureStatus.setText("Capture needs attention");}}
    private JSONObject captureOutcome(JSONObject result){
        try{
            JSONObject timing=result.optJSONObject("timing");
            long first=timing==null?0:timing.optLong("first_matched_sensor_timestamp_ns"),last=timing==null?0:timing.optLong("last_matched_sensor_timestamp_ns");
            double seconds=first>0&&last>=first?(last-first)/1e9:0;
            String reason=result.optString("stop_reason","unknown"),explanation;
            switch(reason){
                case "active_physical_camera_changed":explanation="The phone changed lenses during recording.";break;
                case "manual_focus_not_confirmed":explanation="The phone stopped confirming the locked lens/focus.";break;
                case "locked_lens_metadata_missing":explanation="The phone did not return timing metadata for the locked lens.";break;
                case "camera_stream_stalled_for_5_seconds":explanation="The camera stopped delivering frames for five seconds.";break;
                case "encoder_stream_stalled_for_5_seconds":explanation="The video encoder stopped delivering frames for five seconds.";break;
                case "start_failed":explanation="The phone could not start this lens at the requested 8K setting. No alternate lens or resolution was used.";break;
                default:explanation="Stop reason: "+reason+".";
            }
            String targetScanId=walkIntent==null||walkIntent.isNull("target_scan_id")?null:walkIntent.optString("target_scan_id",null);
            return new JSONObject().put("capture_id",captureDir==null?JSONObject.NULL:captureDir.getName())
                .put("mode",captureMode).put("target_scan_id",targetScanId==null?JSONObject.NULL:targetScanId)
                .put("partial",result.optBoolean("partial")).put("stop_reason",reason).put("duration_seconds",seconds)
                .put("message",(seconds>0?String.format(Locale.US,"Recording stopped after %.1f seconds. ",seconds):"Recording did not finish. ")+explanation+" The original is retained in Library; review the selected mode and route before another take.");
        }catch(Exception invalid){return null;}
    }
    @Override public void onStopped(JSONObject result){
        captureOutcome=captureOutcome(result);
        active=false;busy=true;pendingNativeResult=result;paired.stop(PairedCapture.phoneStopReason(result));getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);finishCaptureViewer(captureDialog);
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
                for(int i=0;i<cameras.length();i++){JSONObject c=cameras.optJSONObject(i);if(CapturePreflight.matches(result,c,Build.FINGERPRINT,calibrationCapture()))try{c.put(calibrationCapture()?"recorded_preflight":"automatic_recorded_preflight",CapturePreflight.reference(dir.getName(),result));}catch(JSONException ignored){}}
                boolean roomComplete=reference!=null&&"stopped".equals(reference.optString("server_status"))&&!reference.optBoolean("needs_recovery");
                status.setText(reference==null?(result.optBoolean("partial")?"Partial phone capture saved":"Phone capture saved"):!roomComplete?"Phone saved; room capture needs attention":result.optBoolean("partial")?"Partial paired capture saved":"Paired capture saved");
                detail.setText((match?"Encoded frames match camera sensor timestamps. RoomWalk will validate the uploaded evidence.":"Frame timing was not verified. Video and raw motion are retained for review.")+(reference==null?" Phone video and motion are retained for upload.":roomComplete?" Phone video, motion and the matching room capture are linked for upload.":" Room status: "+reference.optString("server_status","unknown")+". "+reference.optString("error","Use Finalize paired capture to retry.")+" All saved evidence is retained."));
                if(result.optBoolean("partial")&&captureOutcome!=null){status.setText("Recording stopped early · original retained");detail.setText(captureOutcome.optString("message"));}
                saved.setText(archive.getName()+"  •  "+android.text.format.Formatter.formatFileSize(this,archive.length())+"\nUpload or export; this local copy is retained.");refreshSavedList();updateButtons();
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
            ui.post(()->{if(destroyed)return;roomCameras=rows;roomCamerasOrigin=destination.replaceAll("/+$","");ArrayList<String> labels=new ArrayList<>();int selected=-1;String prior=getPreferences(0).getString("room_camera_id","");
                for(int i=0;i<rows.length();i++){JSONObject row=rows.optJSONObject(i);labels.add(row.optString("label",row.optString("camera_id")));if(prior.equals(row.optString("camera_id")))selected=i;}
                roomCamera.setSize(labels.size());if(!labels.isEmpty())roomCamera.setSelection(selected);
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
            transferState=UploadService.snapshot(this);transferActive=transferState.optBoolean("active");
            JSONObject receipt=transferState.optJSONObject("receipt");
            if("complete".equals(transferState.optString("state"))&&receipt!=null&&!transferState.optBoolean("imu"))lastScanId=receipt.optString("id",null);
            updateButtons();
        }catch(Exception error){status.setText("Upload status needs attention");detail.setText(error.getMessage());publishNativeState();}
    }
    private void export(){if(artifact==null)return;pendingExport=artifact;Intent intent=new Intent(Intent.ACTION_CREATE_DOCUMENT);intent.addCategory(Intent.CATEGORY_OPENABLE);intent.setType(artifact.getName().endsWith(".json")?"application/json":"application/zip");intent.putExtra(Intent.EXTRA_TITLE,artifact.getName());startActivityForResult(intent,EXPORT_DOCUMENT);}
    @Override protected void onActivityResult(int request,int result,Intent data){super.onActivityResult(request,result,data);if(shell!=null&&shell.result(request,result,data))return;if(request==EXPORT_DOCUMENT&&result==RESULT_OK&&data!=null&&data.getData()!=null&&pendingExport!=null){final Uri uri=data.getData();final File file=pendingExport;busy=true;updateButtons();worker.execute(()->{try(OutputStream out=getContentResolver().openOutputStream(uri)){if(out==null)throw new IOException("Export destination unavailable");BundleTools.copy(file,out);ui.post(()->{if(destroyed)return;busy=false;status.setText("Export saved");updateButtons();});}catch(Exception e){fail("Export failed",e);}});}}
    private void share(){if(artifact==null)return;Uri uri=new Uri.Builder().scheme("content").authority("org.noesis.roomwalk.files").appendPath(artifact.getParentFile().getName()).appendPath(artifact.getName()).build();Intent send=new Intent(Intent.ACTION_SEND);send.setType(artifact.getName().endsWith(".json")?"application/json":"application/zip");send.putExtra(Intent.EXTRA_STREAM,uri);send.setClipData(ClipData.newRawUri("RoomWalk capture",uri));send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);startActivity(Intent.createChooser(send,"Share RoomWalk capture"));}
    private void openRoomWalk(){try{shell.connect(server.getText());}catch(Exception e){fail("Cannot open RoomWalk",e);}}
    private File[] savedDirectories(){File[] dirs=captureRoot().listFiles(File::isDirectory);if(dirs==null)return new File[0];Arrays.sort(dirs,(a,b)->Long.compare(b.lastModified(),a.lastModified()));return dirs.length>200?Arrays.copyOf(dirs,200):dirs;}
    private boolean pathReferenceMatchesRoomCamera(JSONObject room){
        if(!WalkIntent.requiresPair(captureMode)||walkIntent==null||!walkIntent.has("target_reference"))return true;
        JSONObject reference=walkIntent.optJSONObject("target_reference");
        return reference!=null&&room!=null&&reference.optString("camera_id","").equals(room.optString("camera_id",""));
    }
    private void loadLatest(){File[] dirs=savedDirectories();if(dirs.length>0)selectSaved(dirs[0]);}
    private void chooseSaved(){File[] dirs=savedDirectories();if(dirs.length==0){Toast.makeText(this,"No captures yet",Toast.LENGTH_SHORT).show();return;}String[] labels=new String[dirs.length];for(int i=0;i<dirs.length;i++)labels[i]=dirs[i].getName();new AlertDialog.Builder(this).setTitle("Saved captures and phone checks").setItems(labels,(d,index)->selectSaved(dirs[index])).setNegativeButton("Close",null).show();}
    private void selectSaved(File dir){
        captureDir=dir;calibrationRequest=null;walkIntent=null;walkIntentStatus="unknown";imuTelemetry=null;captureGuidance=null;captureOutcome=null;captureMode=WalkIntent.RECONSTRUCTION;pendingShortTest=false;lastScanId=null;
        File archive=new File(dir,"roomwalk-"+dir.getName()+".zip"),diagnostic=new File(dir,"capabilities.json");
        File imu=new File(dir,"imu_capture_manifest.json"),pair=new File(dir,PairedCapture.FILE),request=new File(dir,"calibration_request.json");
        artifact=archive.isFile()?archive:imu.isFile()?imu:diagnostic.isFile()?diagnostic:pair.isFile()?pair:null;
        File intentFile=new File(dir,"walk_intent.json");
        if(intentFile.isFile())try{
            walkIntent=WalkIntent.validate(BundleTools.readJson(intentFile));
            captureMode=walkIntent.optString("mode",WalkIntent.RECONSTRUCTION);
            walkIntentStatus="selected";
        }catch(Exception error){status.setText("Saved walk intent needs review");detail.setText(error.getMessage());}
        if(request.isFile())try{
            if(request.length()>32768)throw new IOException("Calibration settings exceed their bound");
            JSONObject metadata=BundleTools.readJson(request);String mode=metadata.getString("mode");
            if(!Arrays.asList("camera","imu").contains(mode)||!dir.getName().equals(metadata.optString("capture_id")))throw new IOException("Saved calibration identity differs from this take");
            calibrationRequest=metadata;captureMode=mode;pendingShortTest=metadata.optBoolean("short_test");
            walkIntentStatus="none";
        }catch(Exception error){status.setText("Saved calibration settings need review");detail.setText(error.getMessage());}
        File result=new File(dir,"capture_result.json");
        if(result.isFile()&&result.length()<=262144)try{captureOutcome=captureOutcome(BundleTools.readJson(result));}catch(Exception ignored){}
        saved.setText(artifact!=null?artifact.getName():"Raw capture retained: "+dir.getName());if(uploadButton!=null)updateButtons();
    }
    private boolean hasFocusLock(){JSONObject c=selectedCamera();return c!=null&&getSharedPreferences("focus-locks",0).contains(c.optString("id"));}
    private void changeFocus(){
        JSONObject c=selectedCamera();if(c==null||active||focusBusy)return;
        final Dialog dialog=captureDialog;focusBusy=true;updateButtons();
        CameraPreview.FocusListener listener=(confirmed,message)->{if(destroyed||captureDialog!=dialog)return;focusBusy=false;captureStats.setText(message);if(confirmed)try{c.put("focus_control",FocusSettings.metadata(FocusSettings.read(this,c.optString("id"))));c.remove("recorded_preflight");captureStats.setText(message+". Run a new timing test for this lens/focus mode.");}catch(Exception failure){captureStats.setText(failure.getMessage());}updateButtons();if(confirmed&&!previewReady)restartPreview();};
        if(hasFocusLock())previewController.unlockFocus(c.optString("id"),listener);else previewController.lockFocus(listener);
    }
    private void startImu(){
        if(active||imuActive||imuPreparing||busy||captureDialog!=null)return;
        try{
            long minutes=Long.parseLong(imuMinutes.getText().toString());if(minutes<1||minutes>180)throw new IOException("Choose 1–180 minutes; one minute is the diagnostic default, 180 minutes records stationary noise");
            getPreferences(0).edit().putString("imu_diagnostic_minutes",Long.toString(minutes)).apply();
            imuPreparationMinutes=minutes;imuPreparing=true;captureGuidance=null;
            imuPreparationDeadline=SystemClock.elapsedRealtime()+5000;
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            advanceImuPreparation();
        }catch(Exception error){fail("Could not prepare stationary recording",error);}
    }
    private void advanceImuPreparation(){
        if(!imuPreparing||destroyed)return;
        long remaining=imuPreparationDeadline-SystemClock.elapsedRealtime();
        if(remaining<=0){imuPreparing=false;beginImu();return;}
        try{imuTelemetry=new JSONObject().put("state","preparing").put("countdown_seconds",(remaining+999)/1000).put("elapsed_seconds",0).put("expected_duration_s",imuPreparationMinutes*60).put("accel_samples",0).put("gyro_samples",0);}catch(JSONException ignored){}
        status.setText("Set the phone down · starting in "+((remaining+999)/1000)+" seconds");imuProgress.setText("Not recording yet. Leave the phone untouched.");updateButtons();
        ui.postDelayed(imuPreparationTick,Math.min(1000,remaining));
    }
    private void cancelImuPreparation(){
        if(!imuPreparing)return;imuPreparing=false;ui.removeCallbacks(imuPreparationTick);imuTelemetry=null;
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        status.setText("Stationary recording cancelled");imuProgress.setText("No new sensor recording was started.");updateButtons();
    }
    private void beginImu(){
        try{
            File dir=new File(captureRoot(),"imu-"+new SimpleDateFormat("yyyyMMdd-HHmmss",Locale.US).format(new Date())+"-"+UUID.randomUUID().toString().substring(0,8));
            if(!dir.mkdirs())throw new IOException("Cannot create IMU capture folder");
            captureDir=dir;artifact=null;lastScanId=null;calibrationRequest=null;walkIntent=null;walkIntentStatus="none";captureOutcome=null;pendingShortTest=false;imuActive=true;getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            imuTelemetry=new JSONObject().put("state","starting").put("elapsed_seconds",0).put("expected_duration_s",imuPreparationMinutes*60).put("accel_samples",0).put("gyro_samples",0);
            status.setText("Starting stationary sensors…");imuProgress.setText("Starting motion sensors…");saved.setText("Raw motion capture: "+dir.getName());updateButtons();imuRecorder.start(dir,imuPreparationMinutes*60);
        }catch(Exception error){imuActive=false;imuTelemetry=null;getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);fail("Could not start IMU diagnostic",error);}
    }
    private void stopImu(){if(imuPreparing){cancelImuPreparation();return;}if(!imuActive)return;busy=true;status.setText("Saving IMU diagnostic…");updateButtons();imuRecorder.stop("user_stopped");}
    @Override public void onImuState(String state,JSONObject details){
        if(destroyed)return;
        try{imuTelemetry=new JSONObject(details.toString()).put("state",state);}catch(JSONException ignored){}
        if("recording".equals(state)){long seconds=details.optLong("elapsed_seconds");status.setText(String.format(Locale.US,"IMU diagnostic • %d:%02d:%02d",seconds/3600,(seconds/60)%60,seconds%60));detail.setText("Keep the phone still and this app open. Accel: "+details.optLong("accel_samples")+"   Gyro: "+details.optLong("gyro_samples")+"   "+android.text.format.Formatter.formatFileSize(this,details.optLong("bytes")));imuProgress.setText(status.getText()+"\n"+detail.getText());}
        else{busy=true;status.setText("Saving IMU diagnostic…");imuProgress.setText("Saving motion data…");updateButtons();}
        publishNativeState();
    }
    @Override public void onImuStopped(File directory,JSONObject manifest){
        imuActive=false;getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);if(destroyed)return;
        try{if(imuTelemetry!=null)imuTelemetry.put("state",manifest.optString("status")).put("elapsed_seconds",manifest.optDouble("actual_duration_s"));}catch(JSONException ignored){}
        captureDir=directory;artifact=new File(directory,"imu_capture_manifest.json");busy=true;updateButtons();packageImu(directory);
    }
    @Override public void onImuError(String reason){
        imuActive=false;busy=false;imuTelemetry=null;getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);if(destroyed)return;
        status.setText("IMU diagnostic needs attention");detail.setText(reason+" Raw files are retained.");imuProgress.setText(reason+" Raw files are retained.");if(captureDir!=null)selectSaved(captureDir);updateButtons();
    }
    private void packageImu(File dir){
        busy=true;status.setText("Packaging IMU diagnostic…");updateButtons();
        worker.execute(()->{try{
            JSONObject manifest=BundleTools.readJson(new File(dir,"imu_capture_manifest.json"));File archive=BundleTools.zipImu(dir);
            ui.post(()->{if(destroyed)return;artifact=archive;busy=false;status.setText("complete".equals(manifest.optString("status"))?"IMU recording complete":"Partial IMU recording saved");
                detail.setText("Stop reason: "+manifest.optString("stop_reason")+". Upload or export the raw motion data for analysis; noise calibration remains unverified.");imuProgress.setText(status.getText()+" • "+Math.round(manifest.optDouble("actual_duration_s"))+" seconds\n"+detail.getText());
                saved.setText(archive.getName()+" • "+android.text.format.Formatter.formatFileSize(this,archive.length()));refreshSavedList();updateButtons();});
        }catch(Exception error){fail("IMU files retained; packaging needs attention",error);}});
    }
    private void fail(String title,Exception error){ui.post(()->{if(destroyed)return;busy=false;status.setText(title);detail.setText(error.getMessage()==null?error.toString():error.getMessage());updateButtons();});}
    @Override public void onBackPressed(){if(captureDialog!=null||pendingCaptureViewer){requestCloseCaptureViewer();return;}if(shell==null||!shell.back())super.onBackPressed();}
    @Override protected void onStart(){super.onStart();ui.removeCallbacks(transferPoll);ui.post(transferPoll);}
    @Override protected void onStop(){ui.removeCallbacks(transferPoll);super.onStop();}
    @Override protected void onPause(){if(imuPreparing)cancelImuPreparation();if(imuActive){busy=true;imuRecorder.stop("app_backgrounded");}if(paired.isActive())paired.stop("app_backgrounded");if(captureDialog!=null||pendingCaptureViewer)requestCloseCaptureViewer();else if(active)engine.stop();super.onPause();}
    @Override protected void onDestroy(){destroyed=true;imuPreparing=false;pendingCaptureViewer=false;ui.removeCallbacks(imuPreparationTick);ui.removeCallbacks(transferPoll);if(displays!=null)displays.unregisterDisplayListener(displayListener);engine.close();imuRecorder.close();paired.close();previewController.shutdown();finishCaptureViewer(captureDialog);if(shell!=null)shell.close();worker.shutdown();super.onDestroy();}
}

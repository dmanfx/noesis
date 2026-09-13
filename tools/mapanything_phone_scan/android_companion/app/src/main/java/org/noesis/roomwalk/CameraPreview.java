package org.noesis.roomwalk;

import android.content.Context;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureFailure;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.OutputConfiguration;
import android.hardware.camera2.params.SessionConfiguration;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Range;
import android.view.Surface;

import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/** Live Camera2 viewfinder only. Owns no encoder, sensors, capture files, or caller-provided Surface. */
public final class CameraPreview {
    public interface Listener {
        /** At least one preview capture completed; called on the main thread. */
        void onReady();
        void onError(String reason);
    }

    public interface ClosedListener {
        /** Called once on the main thread. Recording may start only when confirmed is true. */
        void onClosed(boolean confirmed, String reason);
    }

    private static final long STARTUP_TIMEOUT_MS = 5000;
    private static final long CLOSE_TIMEOUT_MS = 5000;
    private static final int MAX_CLOSE_LISTENERS = 8;
    private final CameraManager cameras;
    private final Context context;
    private final Handler main = new Handler(Looper.getMainLooper());
    private final HandlerThread cameraThread = new HandlerThread("RoomWalkViewfinder");
    private final Handler control;
    private volatile PreviewSession current;
    private volatile boolean shutdownRequested;
    private volatile boolean threadStopped;

    public CameraPreview(Context context) {
        Context application = context.getApplicationContext();
        this.context = application == null ? context : application;
        cameras = (CameraManager) (application == null ? context : application).getSystemService(Context.CAMERA_SERVICE);
        cameraThread.start();
        control = new Handler(cameraThread.getLooper());
    }

    public void open(String cameraId, Surface surface, Listener listener) {
        if (listener == null) throw new IllegalArgumentException("A preview listener is required");
        if (shutdownRequested || threadStopped) {
            main.post(() -> listener.onError("The preview controller has shut down"));
            return;
        }
        if (!control.post(() -> beginOpen(cameraId, surface, listener)))
            main.post(() -> listener.onError("The preview camera worker is unavailable"));
    }

    /** Does not release the caller's Surface; wait for confirmed closure before reusing it. */
    public void close(ClosedListener listener) {
        CloseRequest request = listener == null ? null : new CloseRequest(listener);
        if (request != null) main.postDelayed(request.timeout, CLOSE_TIMEOUT_MS);
        PreviewSession session = current;
        if (session != null) {
            session.closing = true;
            session.startOutcome.compareAndSet(0, 2);
            main.removeCallbacks(session.startupTimeout);
        }
        if (threadStopped) {
            finish(request, current == null, current == null
                    ? "Preview is closed" : "Preview camera closure is unconfirmed");
        } else if (!control.post(() -> beginClose(current, request))) {
            finish(request, false, "The preview camera worker could not confirm closure");
        }
    }

    /** Permanently disables new previews; a pending open remains watched until actually closed. */
    public void shutdown() {
        shutdownRequested = true;
        close(null);
    }

    private void beginOpen(String cameraId, Surface surface, Listener listener) {
        if (shutdownRequested) {
            main.post(() -> listener.onError("The preview controller has shut down"));
            stopThreadIfClosed();
            return;
        }
        if (current != null) {
            String reason = current.closing ? "The previous preview camera has not confirmed closure" : "A preview is already open";
            main.post(() -> listener.onError(reason));
            return;
        }
        if (cameraId == null || cameraId.isEmpty() || surface == null || !surface.isValid()) {
            main.post(() -> listener.onError("A selected camera and a live preview surface are required"));
            return;
        }
        PreviewSession session = new PreviewSession(cameraId, surface, listener);
        current = session;
        main.postDelayed(session.startupTimeout, STARTUP_TIMEOUT_MS);
        try {
            if (cameras == null) throw new IllegalStateException("Camera service is unavailable");
            session.characteristics = cameras.getCameraCharacteristics(cameraId);
            session.focus = FocusSettings.read(context,cameraId);
            if (session.closing || shutdownRequested) {
                beginClose(session, null);
                return;
            }
            session.openPending = true;
            cameras.openCamera(cameraId, deviceCallback(session), control);
        } catch (Exception failure) {
            // A synchronous open failure did not hand this controller a camera device.
            session.openPending = false;
            fail(session, "Preview could not open: " + describe(failure));
        }
    }

    private CameraDevice.StateCallback deviceCallback(PreviewSession session) {
        return new CameraDevice.StateCallback() {
            @Override public void onOpened(CameraDevice device) {
                session.openPending = false;
                if (current != session) { closeLateDevice(session, device); return; }
                session.device = device;
                if (session.closing || shutdownRequested) { beginClose(session, null); return; }
                try { configure(session); }
                catch (Exception failure) { fail(session, "Preview configuration failed: " + describe(failure)); }
            }

            @Override public void onDisconnected(CameraDevice device) {
                session.openPending = false;
                if (current != session) { closeLateDevice(session, device); return; }
                session.device = device;
                if (session.closing) beginClose(session, null);
                else fail(session, "The preview camera disconnected");
            }

            @Override public void onError(CameraDevice device, int error) {
                session.openPending = false;
                if (current != session) { closeLateDevice(session, device); return; }
                session.device = device;
                if (session.closing) beginClose(session, null);
                else fail(session, "The preview camera reported error " + error);
            }

            @Override public void onClosed(CameraDevice device) {
                session.openPending = false;
                if (current != session) return;
                if (!session.closing && !shutdownRequested) notifyError(session, "The preview camera closed unexpectedly");
                session.device = null;
                session.capture = null;
                confirmClosed(session, "CameraDevice.onClosed confirmed preview release");
            }
        };
    }

    private void configure(PreviewSession session) throws Exception {
        if (!session.surface.isValid()) throw new IllegalStateException("The preview surface is no longer available");
        CaptureRequest.Builder request = session.device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
        // The recording-capable path uses the same explicit, available camera controls.
        // Older devices can still display a preview with their standard template defaults.
        if (Build.VERSION.SDK_INT >= 33) {
            Range<Integer> templateFps = request.get(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE);
            SessionProbe.applyRequestSettings(request, session.characteristics, new JSONObject());
            Range<Integer> fixed30 = new Range<>(CaptureEngine.FPS, CaptureEngine.FPS);
            Range<Integer>[] advertisedFps = session.characteristics.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
            boolean fixed30Available = false;
            if (advertisedFps != null) for (Range<Integer> range : advertisedFps)
                if (fixed30.equals(range)) { fixed30Available = true; break; }
            // Preview is useful even when this camera cannot meet recording's fixed-FPS gate.
            if (!fixed30Available) request.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, templateFps);
        }
        session.autoFocusMode=request.get(CaptureRequest.CONTROL_AF_MODE);
        FocusSettings.apply(request,session.characteristics,session.cameraId,session.focus);
        session.request=request;
        request.addTarget(session.surface);
        CaptureRequest repeating = request.build();
        OutputConfiguration output = new OutputConfiguration(session.surface);
        if (Build.VERSION.SDK_INT >= 33) {
            long useCase = CameraCharacteristics.SCALER_AVAILABLE_STREAM_USE_CASES_DEFAULT;
            long[] advertised = session.characteristics.get(CameraCharacteristics.SCALER_AVAILABLE_STREAM_USE_CASES);
            if (advertised != null) for (long value : advertised)
                if (value == CameraCharacteristics.SCALER_AVAILABLE_STREAM_USE_CASES_PREVIEW) {
                    useCase = value;
                    break;
                }
            output.setStreamUseCase(useCase);
            output.setTimestampBase(OutputConfiguration.TIMESTAMP_BASE_SENSOR);
        }
        if (Build.VERSION.SDK_INT >= 34) output.setReadoutTimestampEnabled(false);
        SessionConfiguration configuration = new SessionConfiguration(SessionConfiguration.SESSION_REGULAR,
                Collections.singletonList(output), command -> control.post(command), new CameraCaptureSession.StateCallback() {
            @Override public void onConfigured(CameraCaptureSession capture) {
                if (current != session || session.closing || shutdownRequested) {
                    closeCapture(session, capture);
                    if (current == session) beginClose(session, null);
                    return;
                }
                session.capture = capture;
                try {
                    session.callback = new CameraCaptureSession.CaptureCallback() {
                        @Override public void onCaptureCompleted(CameraCaptureSession capture, CaptureRequest request, TotalCaptureResult result) {
                            if (current != session || session.closing || shutdownRequested) return;
                            session.latest=result;session.receivedNs=SystemClock.elapsedRealtimeNanos();
                            FocusSettings.Lock expected=session.pendingFocus==null?session.focus:session.pendingFocus;
                            if(expected!=null&&!FocusSettings.matches(expected,result)) {
                                if(session.focusConfirmed)fail(session,"The camera no longer confirms the saved focus lock");
                                return;
                            }
                            if(session.pendingFocus!=null) {
                                try{FocusSettings.save(context,session.pendingFocus);session.focus=session.pendingFocus;session.pendingFocus=null;session.focusConfirmed=true;completeFocus(session,true,"Focus locked and saved");}
                                catch(Exception failure){completeFocus(session,false,describe(failure));fail(session,describe(failure));return;}
                            }
                            if(session.focus!=null)session.focusConfirmed=true;
                            if (session.startOutcome.compareAndSet(0, 1)) {
                                main.removeCallbacks(session.startupTimeout);
                                main.post(() -> {
                                    if (current == session && !session.closing && !shutdownRequested) session.listener.onReady();
                                });
                            }
                        }
                        @Override public void onCaptureFailed(CameraCaptureSession capture, CaptureRequest request, CaptureFailure failure) {
                            if (current == session && !session.closing)
                                fail(session, "Preview capture failed with reason " + failure.getReason());
                        }
                    };
                    capture.setRepeatingRequest(repeating,session.callback,control);
                } catch (Exception failure) { fail(session, "Preview could not start: " + describe(failure)); }
            }

            @Override public void onConfigureFailed(CameraCaptureSession capture) {
                closeCapture(session, capture);
                if (current == session && !session.closing) fail(session, "The camera rejected the preview session");
                else if (current == session) beginClose(session, null);
            }
        });
        configuration.setSessionParameters(repeating);
        session.device.createCaptureSession(configuration);
    }

    public interface FocusListener { void onFocus(boolean confirmed,String message); }
    /** Lock the measured, settled preview focus. Persist only after an actual result confirms it. */
    public void lockFocus(FocusListener listener) {
        control.post(()->{
            PreviewSession s=current;
            if(s==null||s.closing||s.capture==null||s.focusListener!=null){main.post(()->listener.onFocus(false,"The preview is not ready to lock focus"));return;}
            try{
                s.pendingFocus=FocusSettings.measured(s.characteristics,s.cameraId,s.latest,s.receivedNs);
                s.focusListener=listener;s.focusConfirmed=false;
                FocusSettings.apply(s.request,s.characteristics,s.cameraId,s.pendingFocus);
                s.capture.setRepeatingRequest(s.request.build(),s.callback,control);
                s.focusTimeout=()->{if(s.pendingFocus!=null){completeFocus(s,false,"The camera did not confirm manual focus within 3 seconds");fail(s,"Focus lock was not confirmed; reopen the preview to retry");}};
                control.postDelayed(s.focusTimeout,3000);
            }catch(Exception failure){s.pendingFocus=null;boolean submitted=s.focusListener!=null;completeFocus(s,false,describe(failure));if(!submitted)main.post(()->listener.onFocus(false,describe(failure)));else fail(s,"Focus request failed; reopen the preview to retry");}
        });
    }
    public void unlockFocus(String cameraId,FocusListener listener) {
        control.post(()->{
            try{
                FocusSettings.clear(context,cameraId);PreviewSession s=current;
                if(s!=null&&!s.closing&&s.cameraId.equals(cameraId)&&s.capture!=null){
                    completeFocus(s,false,"Focus lock cancelled");s.pendingFocus=null;s.focus=null;s.focusConfirmed=false;
                    s.request.set(CaptureRequest.CONTROL_AF_MODE,s.autoFocusMode);
                    s.request.set(CaptureRequest.LENS_FOCUS_DISTANCE,null);
                    s.capture.setRepeatingRequest(s.request.build(),s.callback,control);
                }
                main.post(()->listener.onFocus(true,"Automatic focus restored; saved lock removed"));
            }catch(Exception failure){main.post(()->listener.onFocus(false,describe(failure)));if(current!=null)fail(current,"Could not restore automatic focus");}
        });
    }
    private void completeFocus(PreviewSession session,boolean confirmed,String message) {
        if(session.focusTimeout!=null)control.removeCallbacks(session.focusTimeout);
        FocusListener listener=session.focusListener;session.focusListener=null;
        if(listener!=null)main.post(()->listener.onFocus(confirmed,message));
    }

    private void beginClose(PreviewSession session, CloseRequest request) {
        if (session == null) {
            finish(request, true, "No preview camera is open or pending");
            stopThreadIfClosed();
            return;
        }
        if (session != current) {
            finish(request, false, "The preview changed before closure could be confirmed");
            return;
        }
        completeFocus(session,false,"Preview closed before focus was confirmed");
        session.closing = true;
        session.startOutcome.compareAndSet(0, 2);
        main.removeCallbacks(session.startupTimeout);
        session.closeRequests.removeIf(value -> value.finished.get());
        if (request != null && !request.finished.get()) {
            if (session.closeRequests.size() >= MAX_CLOSE_LISTENERS)
                finish(request, false, "Too many pending preview close requests");
            else session.closeRequests.add(request);
        }
        if (!session.closeWatchStarted) {
            session.closeWatchStarted = true;
            main.postDelayed(session.closeTimeout, CLOSE_TIMEOUT_MS);
        }
        if (session.capture != null) {
            CameraCaptureSession capture = session.capture;
            session.capture = null;
            closeCapture(session, capture);
        }
        if (session.device != null) {
            if (!session.deviceCloseRequested) {
                session.deviceCloseRequested = true;
                try { session.device.close(); }
                catch (RuntimeException failure) {
                    session.deviceCloseRequested = false;
                    notifyError(session, "Preview camera close failed: " + describe(failure));
                    for (CloseRequest waiting : session.closeRequests)
                        finish(waiting, false, "The preview camera did not confirm release: " + describe(failure));
                    session.closeRequests.clear();
                }
            }
            // CameraDevice.close() returning is not the handoff signal. onClosed is.
        } else if (!session.openPending) {
            confirmClosed(session, "No camera device was opened by this preview attempt");
        }
    }

    private void confirmClosed(PreviewSession session, String reason) {
        session.closed = true;
        main.removeCallbacks(session.startupTimeout);
        main.removeCallbacks(session.closeTimeout);
        if (current == session) current = null;
        for (CloseRequest request : session.closeRequests) finish(request, true, reason);
        session.closeRequests.clear();
        stopThreadIfClosed();
    }

    private void closeCapture(PreviewSession session, CameraCaptureSession capture) {
        try { capture.close(); }
        catch (RuntimeException failure) { notifyError(session, "Preview session close failed: " + describe(failure)); }
    }

    private void closeLateDevice(PreviewSession session, CameraDevice device) {
        try { device.close(); }
        catch (RuntimeException failure) { notifyError(session, "A late preview camera could not close: " + describe(failure)); }
    }

    private void fail(PreviewSession session, String reason) {
        if (current != session) return;
        session.startOutcome.compareAndSet(0, 2);
        notifyError(session, reason);
        beginClose(session, null);
    }

    private void notifyError(PreviewSession session, String reason) {
        if (session.errorReported.compareAndSet(false, true))
            main.post(() -> session.listener.onError(bounded(reason)));
    }

    private void finish(CloseRequest request, boolean confirmed, String reason) {
        if (request == null || !request.finished.compareAndSet(false, true)) return;
        main.removeCallbacks(request.timeout);
        main.post(() -> request.listener.onClosed(confirmed, bounded(reason)));
    }

    private void stopThreadIfClosed() {
        if (shutdownRequested && current == null && !threadStopped) {
            threadStopped = true;
            cameraThread.quitSafely();
        }
    }

    private static String describe(Exception failure) {
        return bounded(failure.getClass().getSimpleName() + (failure.getMessage() == null ? "" : ": " + failure.getMessage()));
    }

    private static String bounded(String text) {
        return text.length() <= 256 ? text : text.substring(0, 253) + "...";
    }

    private final class CloseRequest {
        final ClosedListener listener;
        final AtomicBoolean finished = new AtomicBoolean();
        final Runnable timeout;

        CloseRequest(ClosedListener listener) {
            this.listener = listener;
            timeout = () -> finish(this, false,
                    "Preview camera closure was not confirmed within 5 seconds; recording must not start");
        }
    }

    private final class PreviewSession {
        final String cameraId;
        final Surface surface;
        final Listener listener;
        final AtomicInteger startOutcome = new AtomicInteger(); // 0 pending, 1 first frame, 2 failed/cancelled
        final AtomicBoolean errorReported = new AtomicBoolean();
        final List<CloseRequest> closeRequests = new ArrayList<>();
        final Runnable startupTimeout;
        final Runnable closeTimeout;
        volatile boolean closing, closed;
        boolean openPending, deviceCloseRequested, closeWatchStarted;
        CameraCharacteristics characteristics;
        CameraDevice device;
        CameraCaptureSession capture;
        CaptureRequest.Builder request;CameraCaptureSession.CaptureCallback callback;
        TotalCaptureResult latest;long receivedNs;Integer autoFocusMode;
        FocusSettings.Lock focus,pendingFocus;boolean focusConfirmed;FocusListener focusListener;Runnable focusTimeout;

        PreviewSession(String cameraId, Surface surface, Listener listener) {
            this.cameraId = cameraId;
            this.surface = surface;
            this.listener = listener;
            startupTimeout = () -> {
                if (current != this || closing || !startOutcome.compareAndSet(0, 2)) return;
                closing = true;
                notifyError(this, "No preview frame arrived within 5 seconds; camera release is pending");
                control.post(() -> beginClose(this, null));
            };
            closeTimeout = () -> {
                if (!closed) notifyError(this, "Preview camera closure is unconfirmed; do not start recording");
            };
        }
    }
}

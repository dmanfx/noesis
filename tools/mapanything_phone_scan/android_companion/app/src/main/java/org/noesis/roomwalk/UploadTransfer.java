package org.noesis.roomwalk;

import android.os.SystemClock;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.net.InetAddress;
import java.net.Socket;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import javax.net.ssl.HttpsURLConnection;
import javax.net.ssl.SSLSocketFactory;

/** One upload's cancellation and socket lifetime, independent of the activity. */
final class UploadTransfer {
    interface Phase { void changed(String state); }
    private static final ThreadPoolExecutor CLEANUP=new ThreadPoolExecutor(0,1,30,TimeUnit.SECONDS,
            new ArrayBlockingQueue<>(1),task->{Thread thread=new Thread(task,"roomwalk-upload-close");thread.setDaemon(true);return thread;});
    private final Phase phase;
    private volatile Socket socket;
    private volatile String cancellation;
    private volatile Thread owner;
    volatile long lastProgress=SystemClock.elapsedRealtime();
    UploadTransfer(Phase phase){this.phase=phase;}
    void ownThread(){owner=Thread.currentThread();}
    boolean cancelled(){return cancellation!=null;}
    String reason(){return cancellation;}
    void check()throws IOException{if(cancelled())throw new InterruptedIOException(cancellation);}
    void progress(){lastProgress=SystemClock.elapsedRealtime();}
    void phase(String state)throws IOException{check();progress();phase.changed(state);}
    synchronized void cancel(String reason){
        if(cancellation!=null)return;cancellation=reason;
        Thread thread=owner;if(thread!=null)thread.interrupt();
        // disconnect() can wait behind Android's active HTTP writer. Close only
        // its plain TCP socket here; the transfer worker disconnects in finally.
        try{CLEANUP.execute(()->{Socket raw=socket;if(raw!=null)try{raw.close();}catch(IOException ignored){}});}
        catch(java.util.concurrent.RejectedExecutionException ignored){/* Earlier close already owns the only transfer. */}
    }
    void attach(HttpsURLConnection conn)throws IOException{
        check();SSLSocketFactory delegate=conn.getSSLSocketFactory();
        conn.setSSLSocketFactory(new SSLSocketFactory(){
            @Override public String[] getDefaultCipherSuites(){return delegate.getDefaultCipherSuites();}
            @Override public String[] getSupportedCipherSuites(){return delegate.getSupportedCipherSuites();}
            @Override public Socket createSocket(Socket raw,String host,int port,boolean close)throws IOException{
                socket=raw;
                if(cancelled()){raw.close();check();}
                return delegate.createSocket(raw,host,port,close);
            }
            @Override public Socket createSocket()throws IOException{return delegate.createSocket();}
            @Override public Socket createSocket(String host,int port)throws IOException{return delegate.createSocket(host,port);}
            @Override public Socket createSocket(String host,int port,InetAddress local,int localPort)throws IOException{return delegate.createSocket(host,port,local,localPort);}
            @Override public Socket createSocket(InetAddress host,int port)throws IOException{return delegate.createSocket(host,port);}
            @Override public Socket createSocket(InetAddress host,int port,InetAddress local,int localPort)throws IOException{return delegate.createSocket(host,port,local,localPort);}
        });
    }
    void detached(){socket=null;}
}

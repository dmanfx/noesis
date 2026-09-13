package org.noesis.roomwalk;

import android.net.DnsResolver;
import android.os.CancellationSignal;
import android.os.SystemClock;
import java.io.IOException;
import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.Proxy;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.net.URI;
import java.net.URL;
import java.net.UnknownHostException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.HttpsURLConnection;
import javax.net.ssl.SNIHostName;
import javax.net.ssl.SSLParameters;
import javax.net.ssl.SSLSocket;
import javax.net.ssl.SSLSocketFactory;

/** Direct LAN IPv4 routing with the original HTTPS hostname as TLS authority. */
final class LanIpv4Https {
    static final String ROOMWALK_HOSTNAME = "TauntonMainframe.local";
    static final String ROOMWALK_IPV4 = "192.168.3.126";
    static final int DNS_TIMEOUT_MS=1000;
    static final int CONNECT_TIMEOUT_MS=750;
    private static final Map<String,HostTls> HOST_TLS=new LinkedHashMap<String,HostTls>(8,0.75f,true){
        @Override protected boolean removeEldestEntry(Map.Entry<String,HostTls> entry){return size()>8;}
    };
    private LanIpv4Https(){}

    static HttpsURLConnection open(URL origin,SSLSocketFactory trusted,long deadline)throws Exception{
        String hostname=origin.getHost();Inet4Address address=resolve(hostname,deadline);
        URL route=new URI("https://"+address.getHostAddress()+(origin.getPort()<0?"":":"+origin.getPort())+origin.getFile()).toURL();
        HttpsURLConnection connection=(HttpsURLConnection)route.openConnection(Proxy.NO_PROXY);
        HostTls binding=binding(hostname,trusted);
        connection.setSSLSocketFactory(binding.factory);
        connection.setHostnameVerifier(binding.verifier);
        connection.setRequestProperty("Host",origin.getAuthority());
        connection.setConnectTimeout(Math.min(CONNECT_TIMEOUT_MS,remaining(deadline)));
        connection.setReadTimeout(remaining(deadline));
        connection.setInstanceFollowRedirects(false);
        return connection;
    }

    static int remaining(long deadline)throws SocketTimeoutException{
        long remaining=deadline-SystemClock.elapsedRealtime();
        if(remaining<=0)throw new SocketTimeoutException("RoomWalk LAN request timed out");
        return (int)Math.min(Integer.MAX_VALUE,remaining);
    }

    private static Inet4Address literal(String host)throws UnknownHostException{
        if(!host.matches("[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+"))return null;
        String[] parts=host.split("\\.");byte[] bytes=new byte[4];
        for(int i=0;i<4;i++){
            if(parts[i].length()>3||(parts[i].length()>1&&parts[i].startsWith("0")))throw new UnknownHostException("Use a standard IPv4 address");
            int value=Integer.parseInt(parts[i]);if(value>255)throw new UnknownHostException("Invalid IPv4 address");bytes[i]=(byte)value;
        }
        return (Inet4Address)InetAddress.getByAddress(bytes);
    }

    static Inet4Address resolve(String host,long deadline)throws Exception{
        // The appliance address is intentionally fixed for this LAN. Keep the
        // logical hostname for TLS/SNI and HTTP Host, but never ask DNS for it.
        if(ROOMWALK_HOSTNAME.equalsIgnoreCase(host))return literal(ROOMWALK_IPV4);
        Inet4Address literal=literal(host);if(literal!=null)return literal;
        if(host.indexOf(':')>=0)throw new UnknownHostException("RoomWalk requires an IPv4 LAN address or a hostname with an IPv4 DNS record");
        CompletableFuture<Inet4Address> answer=new CompletableFuture<>();CancellationSignal cancellation=new CancellationSignal();
        try{
            // Query A records explicitly; filtering getAllByName results would
            // still wait for AAAA lookup and inherit its negative-cache delays.
            DnsResolver.getInstance().query(null,host,DnsResolver.TYPE_A,DnsResolver.FLAG_NO_RETRY,Runnable::run,cancellation,new DnsResolver.Callback<List<InetAddress>>(){
                @Override public void onAnswer(List<InetAddress> addresses,int rcode){
                    for(InetAddress address:addresses)if(rcode==0&&address instanceof Inet4Address){answer.complete((Inet4Address)address);return;}
                    answer.completeExceptionally(new UnknownHostException("No IPv4 LAN address found for "+host));
                }
                @Override public void onError(DnsResolver.DnsException error){
                    UnknownHostException failure=new UnknownHostException("Could not resolve the IPv4 LAN address for "+host);failure.initCause(error);answer.completeExceptionally(failure);
                }
            });
            return answer.get(Math.min(DNS_TIMEOUT_MS,remaining(deadline)),TimeUnit.MILLISECONDS);
        }catch(TimeoutException error){throw new SocketTimeoutException("IPv4 LAN address lookup timed out after "+DNS_TIMEOUT_MS+" ms. Check the RoomWalk address and phone Wi-Fi.");}
        catch(ExecutionException error){Throwable cause=error.getCause();if(cause instanceof Exception)throw (Exception)cause;throw new IOException("IPv4 lookup failed",cause);}
        catch(InterruptedException error){Thread.currentThread().interrupt();throw error;}
        finally{cancellation.cancel();}
    }

    private static synchronized HostTls binding(String hostname,SSLSocketFactory trusted){
        String key=hostname.toLowerCase(Locale.ROOT);HostTls value=HOST_TLS.get(key);
        if(value==null||value.trusted!=trusted){value=new HostTls(hostname,trusted);HOST_TLS.put(key,value);}return value;
    }
    private static final class HostTls{
        final SSLSocketFactory trusted,factory;final HostnameVerifier verifier;
        HostTls(String hostname,SSLSocketFactory trusted){
            this.trusted=trusted;
            factory=new BoundFactory(hostname,trusted);
            HostnameVerifier standard=HttpsURLConnection.getDefaultHostnameVerifier();
            verifier=(route,session)->standard.verify(hostname,session);
        }
    }
    private static final class BoundFactory extends SSLSocketFactory{
        private final String hostname;private final SSLSocketFactory trusted;
        BoundFactory(String hostname,SSLSocketFactory trusted){this.hostname=hostname;this.trusted=trusted;}
        @Override public String[] getDefaultCipherSuites(){return trusted.getDefaultCipherSuites();}
        @Override public String[] getSupportedCipherSuites(){return trusted.getSupportedCipherSuites();}
        @Override public Socket createSocket(Socket socket,String route,int port,boolean autoClose)throws IOException{
            if(!(socket.getInetAddress() instanceof Inet4Address))throw new IOException("RoomWalk HTTPS requires an established IPv4 connection");
            SSLSocket tls=(SSLSocket)trusted.createSocket(socket,hostname,port,autoClose);
            SSLParameters parameters=tls.getSSLParameters();parameters.setEndpointIdentificationAlgorithm("HTTPS");
            if(literal(hostname)==null)parameters.setServerNames(Collections.singletonList(new SNIHostName(hostname)));
            tls.setSSLParameters(parameters);int readTimeout=socket.getSoTimeout();
            tls.setSoTimeout(readTimeout>0?Math.min(readTimeout,3000):3000);
            // Complete verified TLS while the logical hostname is authoritative.
            // Android's HTTPS implementation later configures the URL's route
            // hostname; that must never change the identity used for this handshake.
            try{tls.startHandshake();tls.setSoTimeout(readTimeout);return tls;}
            catch(IOException error){try{tls.close();}catch(IOException closeError){error.addSuppressed(closeError);}throw error;}
        }
        // HttpsURLConnection supplies its already-connected route. Refuse any
        // unexpected overload that could silently perform dual-stack lookup.
        @Override public Socket createSocket()throws IOException{throw routeRequired();}
        @Override public Socket createSocket(String host,int port)throws IOException{throw routeRequired();}
        @Override public Socket createSocket(String host,int port,InetAddress local,int localPort)throws IOException{throw routeRequired();}
        @Override public Socket createSocket(InetAddress host,int port)throws IOException{throw routeRequired();}
        @Override public Socket createSocket(InetAddress host,int port,InetAddress local,int localPort)throws IOException{throw routeRequired();}
        private IOException routeRequired(){return new IOException("RoomWalk TLS requires an established IPv4 route");}
    }
}

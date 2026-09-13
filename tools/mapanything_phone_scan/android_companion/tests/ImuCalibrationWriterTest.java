package org.noesis.roomwalk;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.util.concurrent.atomic.AtomicInteger;

/** Exercises retained raw data, independent clocks, checksums and finite writer limits. */
public final class ImuCalibrationWriterTest {
    private static int checks;
    private static final class Events implements ImuCalibrationWriter.Listener {
        volatile String failure; final AtomicInteger failures=new AtomicInteger(),finished=new AtomicInteger();
        public void onFailure(String value){failure=value;failures.incrementAndGet();}
        public void onFinished(){finished.incrementAndGet();}
    }
    private static File directory() throws Exception{return Files.createTempDirectory("roomwalk-imu-writer-test-").toFile();}
    private static ImuCalibrationWriter.Sample sample(boolean accel,long stamp){return new ImuCalibrationWriter.Sample(accel,stamp,new float[]{1.25f,-2f,9.81f,0.1f,0.2f,0.3f},3,stamp+100);}
    private static void finish(ImuCalibrationWriter writer,Events events) throws Exception{writer.finish();require(writer.awaitFinished(3000),"Writer finalized");require(events.finished.get()==1,"One terminal callback");}
    public static void main(String[] args) throws Exception {
        File dir=directory();Events events=new Events();ImuCalibrationWriter writer=new ImuCalibrationWriter(dir,events,64,100,100000);
        float[] values={4,5,6};writer.offer(new ImuCalibrationWriter.Sample(true,1000,values,0,1400));values[0]=999;
        writer.offer(sample(false,1500));writer.offer(sample(true,2000));writer.offer(sample(false,2700));writer.start();finish(writer,events);
        String accel=new String(Files.readAllBytes(new File(dir,"accel.csv").toPath()),StandardCharsets.UTF_8);
        String gyro=new String(Files.readAllBytes(new File(dir,"gyro.csv").toPath()),StandardCharsets.UTF_8);
        require(accel.startsWith(ImuCalibrationWriter.HEADER),"Exact CSV header");require(accel.contains("1000,4.0,5.0,6.0,,,,0,1400\n"),"Cloned original values and absent bias retained");
        require(gyro.contains("1500,1.25,-2.0,9.81,0.1,0.2,0.3,3,1600\n"),"Unsubtracted raw values and sensor bias estimates retained");
        require(writer.accelerometer.firstTimestamp==1000&&writer.gyroscope.firstTimestamp==1500,"Stream clocks remain independent");
        require(writer.accelerometer.maximumInterval==1000&&writer.gyroscope.maximumInterval==1200,"Independent sample cadence");
        require(writer.accelerometer.unreliable==1,"Accuracy statistics retained");require(writer.droppedRecords.get()==0&&events.failure==null,"Valid data does not degrade");
        require(writer.bytes()==new File(dir,"accel.csv").length()+new File(dir,"gyro.csv").length(),"Byte bound counts headers and rows");
        require(hash(new File(dir,"accel.csv")).equals(writer.accelerometer.sha256)&&hash(new File(dir,"gyro.csv")).equals(writer.gyroscope.sha256),"Checksums cover exact durable CSV bytes");
        require(!writer.offer(sample(true,3000)),"Closed writer rejects new samples");
        boolean overwrite=false;try{new ImuCalibrationWriter(dir,new Events()).start();}catch(java.io.IOException expected){overwrite=true;}require(overwrite,"Existing evidence is not overwritten");

        events=new Events();writer=new ImuCalibrationWriter(directory(),events,1,10,10000);writer.offer(sample(true,1));require(!writer.offer(sample(false,2)),"Overflow does not block producer");writer.start();finish(writer,events);
        require(events.failures.get()==1&&"imu_writer_queue_overflow".equals(events.failure)&&writer.droppedRecords.get()==1,"Overflow explicitly invalidates evidence while retaining queued sample");require(writer.accelerometer.count==1,"Overflow retains queued data");

        events=new Events();writer=new ImuCalibrationWriter(directory(),events,8,2,10000);for(int i=1;i<=4;i++)writer.offer(sample(true,i));writer.start();finish(writer,events);
        require(writer.accelerometer.count==2&&writer.droppedRecords.get()==2&&"imu_stream_row_limit".equals(events.failure),"Per-stream row cap bounds storage and counts discarded tail");

        events=new Events();writer=new ImuCalibrationWriter(directory(),events,8,100,2*ImuCalibrationWriter.HEADER.length());writer.offer(sample(true,1));writer.start();finish(writer,events);
        require(writer.accelerometer.count==0&&writer.bytes()==2*ImuCalibrationWriter.HEADER.length()&&"imu_storage_limit".equals(events.failure),"Byte cap cannot be exceeded by a row");

        events=new Events();writer=new ImuCalibrationWriter(directory(),events,8,100,10000);writer.offer(sample(false,10));writer.offer(sample(false,9));writer.offer(sample(false,11));writer.start();finish(writer,events);
        require(writer.gyroscope.nonmonotonic==1&&writer.gyroscope.count==2&&writer.droppedRecords.get()==1&&"nonmonotonic_imu_timestamp".equals(events.failure),"Clock regression retained and flagged without interpolation");
        System.out.println("IMU writer: "+checks+" checks passed");
    }
    private static String hash(File file)throws Exception{byte[] bytes=MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file.toPath()));StringBuilder out=new StringBuilder();for(byte value:bytes)out.append(String.format("%02x",value&255));return out.toString();}
    private static void require(boolean condition,String message){checks++;if(!condition)throw new AssertionError(message);}
}

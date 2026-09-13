package org.noesis.roomwalk;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.DigestOutputStream;
import java.security.MessageDigest;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/** Bounded raw IMU persistence; acquisition only copies and offers a small sample. */
public final class ImuCalibrationWriter {
    public static final String HEADER = "timestamp_ns,x,y,z,bias_x,bias_y,bias_z,accuracy,received_elapsed_realtime_ns\n";
    public static final long MAX_ROWS_PER_STREAM = 4_000_000;
    public static final long MAX_BYTES = 512L * 1024 * 1024;
    public static final int QUEUE_CAPACITY = 8192;
    public interface Listener {
        void onFailure(String reason);
        void onFinished();
    }
    public static final class Sample {
        final boolean accel;
        final long timestamp, received;
        final int accuracy;
        final float[] values;
        public Sample(boolean accel, long timestamp, float[] values, int accuracy, long received) {
            this.accel = accel; this.timestamp = timestamp; this.received = received;
            this.accuracy = accuracy;
            this.values = values.clone();
        }
        String csv() throws IOException {
            if (timestamp <= 0 || received <= 0 || values.length < 3)
                throw new IOException("invalid_sensor_sample");
            StringBuilder row = new StringBuilder(144).append(timestamp);
            for (int i = 0; i < 6; i++) {
                row.append(',');
                if (i < values.length) {
                    if (!Float.isFinite(values[i])) throw new IOException("nonfinite_sensor_sample");
                    row.append(values[i]);
                }
            }
            return row.append(',').append(accuracy).append(',').append(received).append('\n').toString();
        }
    }
    public static final class Stats {
        public volatile long count, firstTimestamp, lastTimestamp, lastReceived, bytes;
        public volatile long nonmonotonic, unreliable, minimumInterval = Long.MAX_VALUE, maximumInterval;
        public volatile String sha256;
        void add(Sample sample, int size) {
            if (count == 0) firstTimestamp = sample.timestamp;
            else {
                long interval = sample.timestamp - lastTimestamp;
                if (interval <= 0) nonmonotonic++;
                else { minimumInterval = Math.min(minimumInterval, interval); maximumInterval = Math.max(maximumInterval, interval); }
            }
            if (sample.accuracy == 0) unreliable++;
            lastTimestamp = sample.timestamp; lastReceived = sample.received;
            bytes += size; count++;
        }
    }
    public final Stats accelerometer = new Stats(), gyroscope = new Stats();
    public final AtomicLong droppedRecords = new AtomicLong();
    private final ArrayBlockingQueue<Sample> queue;
    private final AtomicBoolean accepting = new AtomicBoolean(true), failureReported = new AtomicBoolean();
    private final File directory;
    private final long maximumRows, maximumBytes;
    private final Listener listener;
    private volatile boolean finishRequested, finalized;
    private Thread thread;

    public ImuCalibrationWriter(File directory, Listener listener) {
        this(directory, listener, QUEUE_CAPACITY, MAX_ROWS_PER_STREAM, MAX_BYTES - 65536);
    }
    ImuCalibrationWriter(File directory, Listener listener, int capacity, long maximumRows, long maximumBytes) {
        this.directory = directory; this.listener = listener;
        this.maximumRows = maximumRows; this.maximumBytes = maximumBytes;
        queue = new ArrayBlockingQueue<>(capacity);
    }
    public synchronized void start() throws IOException {
        if (thread != null) throw new IOException("IMU writer already started");
        if (!directory.isDirectory()) throw new IOException("IMU output directory is missing");
        if (new File(directory, "accel.csv").exists() || new File(directory, "gyro.csv").exists())
            throw new IOException("IMU source files already exist; refusing to overwrite them");
        thread = new Thread(this::write, "RoomWalkImuCalibrationWriter");
        thread.start();
    }
    public boolean offer(Sample sample) {
        if (!accepting.get()) return false;
        if (queue.offer(sample)) return true;
        droppedRecords.incrementAndGet();
        fail("imu_writer_queue_overflow");
        return false;
    }
    public void finish() { accepting.set(false); finishRequested = true; }
    public boolean isFinalized() { return finalized; }
    public long bytes() { return accelerometer.bytes + gyroscope.bytes; }
    public boolean awaitFinished(long timeoutMs) throws InterruptedException {
        Thread worker = thread;
        if (worker != null) worker.join(timeoutMs);
        return finalized;
    }
    private void fail(String reason) {
        accepting.set(false); finishRequested = true;
        if (failureReported.compareAndSet(false, true)) listener.onFailure(reason);
    }
    private void write() {
        try {
            MessageDigest accelHash = MessageDigest.getInstance("SHA-256"), gyroHash = MessageDigest.getInstance("SHA-256");
            try (FileOutputStream accelFile = new FileOutputStream(new File(directory, "accel.csv"));
                 FileOutputStream gyroFile = new FileOutputStream(new File(directory, "gyro.csv"));
                 DigestOutputStream accel = new DigestOutputStream(new BufferedOutputStream(accelFile, 65536), accelHash);
                 DigestOutputStream gyro = new DigestOutputStream(new BufferedOutputStream(gyroFile, 65536), gyroHash)) {
                byte[] header = HEADER.getBytes(StandardCharsets.UTF_8);
                accel.write(header); gyro.write(header);
                accelerometer.bytes = gyroscope.bytes = header.length;
                long lastFlush = System.nanoTime();
                while (!finishRequested || !queue.isEmpty()) {
                    Sample sample = queue.poll(100, TimeUnit.MILLISECONDS);
                    if (sample != null) {
                        Stats stats = sample.accel ? accelerometer : gyroscope;
                        if (stats.count >= maximumRows) {
                            droppedRecords.incrementAndGet(); fail("imu_stream_row_limit"); break;
                        }
                        byte[] row = sample.csv().getBytes(StandardCharsets.UTF_8);
                        if (bytes() + row.length > maximumBytes) {
                            droppedRecords.incrementAndGet(); fail("imu_storage_limit"); break;
                        }
                        (sample.accel ? accel : gyro).write(row);
                        stats.add(sample, row.length);
                        if (stats.nonmonotonic > 0) { fail("nonmonotonic_imu_timestamp"); break; }
                    }
                    if (System.nanoTime() - lastFlush >= 1_000_000_000L) {
                        accel.flush(); gyro.flush(); lastFlush = System.nanoTime();
                    }
                }
                accel.flush(); gyro.flush(); accelFile.getFD().sync(); gyroFile.getFD().sync();
            }
            accelerometer.sha256 = hex(accelHash.digest()); gyroscope.sha256 = hex(gyroHash.digest());
            finalized = true;
        } catch (Exception failure) {
            droppedRecords.incrementAndGet();
            fail("imu_writer_failed: " + failure.getClass().getSimpleName() + ": " + failure.getMessage());
        } finally {
            accepting.set(false);
            droppedRecords.addAndGet(queue.size()); queue.clear();
            listener.onFinished();
        }
    }
    private static String hex(byte[] values) {
        StringBuilder result = new StringBuilder(64);
        for (byte value : values) result.append(String.format(java.util.Locale.US, "%02x", value & 255));
        return result.toString();
    }
}

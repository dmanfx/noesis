package org.noesis.roomwalk;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Exact sensor/encoder association. It never estimates an offset or a nearest frame. */
public final class TimestampAssociation {
    private TimestampAssociation() {}

    public static final class SensorFrame {
        public final long frameNumber;
        public final long timestampNs;

        public SensorFrame(long frameNumber, long timestampNs) {
            this.frameNumber = frameNumber;
            this.timestampNs = timestampNs;
        }
    }

    public static final class EncodedFrame {
        public final int index;
        public final long ptsUs;

        public EncodedFrame(int index, long ptsUs) {
            this.index = index;
            this.ptsUs = ptsUs;
        }
    }

    public static final class Match {
        public final EncodedFrame encoded;
        public final SensorFrame sensor;

        Match(EncodedFrame encoded, SensorFrame sensor) {
            this.encoded = encoded;
            this.sensor = sensor;
        }
    }

    public static final class Result {
        public final List<Match> matches = new ArrayList<>();
        public int unmatchedEncodedFrames;
        public int duplicateEncodedPts;
        public int duplicateSensorTimestampUs;
        public int duplicateCameraFrameNumbers;
        public boolean sensorTimestampsMonotonic = true;
        public boolean encoderPtsMonotonic = true;
        public boolean verified;
    }

    public static Result associate(List<SensorFrame> sensors, List<EncodedFrame> encoded) {
        Result result = new Result();
        Map<Long, SensorFrame> byMicrosecond = new HashMap<>();
        Set<Long> ambiguousMicroseconds = new HashSet<>();
        Set<Long> cameraFrameNumbers = new HashSet<>();
        long previousSensorNs = -1;
        for (SensorFrame sensor : sensors) {
            if (!cameraFrameNumbers.add(sensor.frameNumber)) result.duplicateCameraFrameNumbers++;
            if (sensor.timestampNs <= previousSensorNs || sensor.timestampNs <= 0) {
                result.sensorTimestampsMonotonic = false;
            }
            previousSensorNs = sensor.timestampNs;
            long microsecond = sensor.timestampNs / 1000L;
            if (byMicrosecond.put(microsecond, sensor) != null) {
                result.duplicateSensorTimestampUs++;
                ambiguousMicroseconds.add(microsecond);
            }
        }
        Set<Long> seenEncoded = new HashSet<>();
        long previousPtsUs = -1;
        for (EncodedFrame frame : encoded) {
            if (frame.ptsUs <= previousPtsUs || frame.ptsUs <= 0) {
                result.encoderPtsMonotonic = false;
            }
            previousPtsUs = frame.ptsUs;
            boolean duplicate = !seenEncoded.add(frame.ptsUs);
            if (duplicate) result.duplicateEncodedPts++;
            SensorFrame sensor = byMicrosecond.get(frame.ptsUs);
            if (sensor == null || ambiguousMicroseconds.contains(frame.ptsUs) || duplicate) {
                result.unmatchedEncodedFrames++;
            } else {
                result.matches.add(new Match(frame, sensor));
            }
        }
        result.verified = !encoded.isEmpty()
                && result.unmatchedEncodedFrames == 0
                && result.duplicateEncodedPts == 0
                && result.duplicateSensorTimestampUs == 0
                && result.duplicateCameraFrameNumbers == 0
                && result.sensorTimestampsMonotonic
                && result.encoderPtsMonotonic;
        return result;
    }
}

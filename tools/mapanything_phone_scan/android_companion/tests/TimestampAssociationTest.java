package org.noesis.roomwalk;

import java.util.Arrays;
import java.util.Collections;

/** Runs on the host JVM, independent of Android stubs and a physical phone. */
public final class TimestampAssociationTest {
    private static int checks;

    public static void main(String[] args) {
        TimestampAssociation.Result exact = TimestampAssociation.associate(
                Arrays.asList(sensor(41, 9_000_000_123L), sensor(42, 9_033_333_456L)),
                Arrays.asList(encoded(0, 9_000_000L), encoded(1, 9_033_333L)));
        require(exact.verified && exact.matches.size() == 2, "Exact integer microsecond truncation must match");
        require(exact.matches.get(1).sensor.timestampNs == 9_033_333_456L,
                "Resolved output must retain the original nanoseconds");

        TimestampAssociation.Result offset = TimestampAssociation.associate(
                Arrays.asList(sensor(41, 9_000_000_123L), sensor(42, 9_033_333_456L)),
                Arrays.asList(encoded(0, 0), encoded(1, 33_333)));
        require(!offset.verified && offset.unmatchedEncodedFrames == 2,
                "A normalized encoder clock must never be repaired with an inferred offset");

        TimestampAssociation.Result ambiguous = TimestampAssociation.associate(
                Arrays.asList(sensor(41, 9_000_000_123L), sensor(42, 9_000_000_456L)),
                Collections.singletonList(encoded(0, 9_000_000L)));
        require(!ambiguous.verified && ambiguous.duplicateSensorTimestampUs == 1
                        && ambiguous.unmatchedEncodedFrames == 1,
                "Two sensor frames in one encoder microsecond must remain ambiguous");

        TimestampAssociation.Result duplicateOutput = TimestampAssociation.associate(
                Collections.singletonList(sensor(41, 9_000_000_123L)),
                Arrays.asList(encoded(0, 9_000_000L), encoded(1, 9_000_000L)));
        require(!duplicateOutput.verified && duplicateOutput.duplicateEncodedPts == 1,
                "A camera frame cannot be reused for duplicated encoder PTS");

        TimestampAssociation.Result reordered = TimestampAssociation.associate(
                Arrays.asList(sensor(41, 9_000_000_123L), sensor(42, 9_033_333_456L)),
                Arrays.asList(encoded(0, 9_033_333L), encoded(1, 9_000_000L)));
        require(!reordered.verified && !reordered.encoderPtsMonotonic,
                "Reordered encoded output must not be silently sorted");

        TimestampAssociation.Result reversedSensor = TimestampAssociation.associate(
                Arrays.asList(sensor(42, 9_033_333_456L), sensor(41, 9_000_000_123L)),
                Arrays.asList(encoded(0, 9_000_000L), encoded(1, 9_033_333L)));
        require(!reversedSensor.verified && !reversedSensor.sensorTimestampsMonotonic,
                "Nonmonotonic sensor evidence must fail verification");

        TimestampAssociation.Result extraCamera = TimestampAssociation.associate(
                Arrays.asList(sensor(41, 9_000_000_123L), sensor(42, 9_033_333_456L), sensor(43, 9_066_666_789L)),
                Arrays.asList(encoded(0, 9_000_000L), encoded(1, 9_066_666L)));
        require(extraCamera.verified && extraCamera.matches.size() == 2,
                "A camera frame omitted by the encoder must not shift the remaining associations");

        TimestampAssociation.Result duplicateFrame = TimestampAssociation.associate(
                Arrays.asList(sensor(41, 9_000_000_123L), sensor(41, 9_033_333_456L)),
                Arrays.asList(encoded(0, 9_000_000L), encoded(1, 9_033_333L)));
        require(!duplicateFrame.verified && duplicateFrame.duplicateCameraFrameNumbers == 1,
                "Repeated Camera2 frame numbers must fail verification");

        require(!TimestampAssociation.associate(Collections.emptyList(), Collections.emptyList()).verified,
                "An empty take cannot verify frame timing");
        System.out.println("Timestamp association: " + checks + " checks passed");
    }

    private static TimestampAssociation.SensorFrame sensor(long frame, long ns) {
        return new TimestampAssociation.SensorFrame(frame, ns);
    }
    private static TimestampAssociation.EncodedFrame encoded(int index, long us) {
        return new TimestampAssociation.EncodedFrame(index, us);
    }
    private static void require(boolean condition, String message) {
        checks++;
        if (!condition) throw new AssertionError(message);
    }
}

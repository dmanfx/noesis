package org.noesis.roomwalk;

/** Host-JVM checks of camera-backed TextureView geometry; no Android stubs are required. */
public final class PreviewTransformTest {
    private static int checks;
    private static int combinations;
    private static final double PIXEL_TOLERANCE = 0.002;

    public static void main(String[] args) {
        assertMatrix(PreviewTransform.matrix(1920, 1080, 1920, 1080, 0, 0),
                new double[]{1, 0, 0, 0, 1, 0, 0, 0, 1}, "Unrotated matching aspect ratio");
        assertMatrix(PreviewTransform.matrix(1000, 1000, 1440, 1080, 0, 0),
                new double[]{1, 0, 0, 0, 0.75, 125, 0, 0, 1}, "4:3 buffer in a square view");
        assertMatrix(PreviewTransform.matrix(1000, 1000, 1440, 1080, 90, 0),
                new double[]{0.75, 0, 125, 0, 1, 0, 0, 0, 1}, "Sensor-rotated 4:3 buffer in a square view");
        assertMatrix(PreviewTransform.matrix(768, 432, 1920, 1080, 90, 90),
                new double[]{0, 16.0 / 9, 0, -9.0 / 16, 0, 432, 0, 0, 1},
                "Landscape phone must undo implicit sensor rotation stretch and display rotation");

        int[][] views = {{1600, 440}, {1080, 1920}, {2208, 1840}, {999, 999}, {853, 479}, {321, 777}};
        int[][] buffers = {{1920, 1080}, {1440, 1080}, {1080, 1920}, {640, 480}, {37, 23}};
        for (int[] view : views) for (int[] buffer : buffers)
            for (int sensor : new int[]{0, 90, 180, 270})
                for (int display : new int[]{0, 90, 180, 270})
                    checkGeometry(view[0], view[1], buffer[0], buffer[1], sensor, display);

        // A sensor's full rotation is already applied by TextureView; only its axis swap
        // can affect this corrective matrix. Applying sensor rotation again breaks this.
        for (int display : new int[]{0, 90, 180, 270}) {
            assertMatrix(PreviewTransform.matrix(1600, 440, 1920, 1080, 0, display),
                    doubles(PreviewTransform.matrix(1600, 440, 1920, 1080, 180, display)),
                    "Sensor 180-degree compensation must not be applied twice");
            assertMatrix(PreviewTransform.matrix(1600, 440, 1920, 1080, 90, display),
                    doubles(PreviewTransform.matrix(1600, 440, 1920, 1080, 270, display)),
                    "Sensor 90/270 orientations have identical natural-orientation dimensions");
        }
        reject(() -> PreviewTransform.matrix(0, 440, 1920, 1080, 90, 90), "Unmeasured view width");
        reject(() -> PreviewTransform.matrix(1600, -1, 1920, 1080, 90, 90), "Negative view height");
        reject(() -> PreviewTransform.matrix(1600, 440, 0, 1080, 90, 90), "Missing buffer width");
        reject(() -> PreviewTransform.matrix(1600, 440, 1920, -1, 90, 90), "Negative buffer height");
        reject(() -> PreviewTransform.matrix(1600, 440, 1920, 1080, 45, 90), "Invalid sensor angle");
        reject(() -> PreviewTransform.matrix(1600, 440, 1920, 1080, -90, 90), "Negative sensor angle");
        reject(() -> PreviewTransform.matrix(1600, 440, 1920, 1080, 90, 1), "Surface enum passed as degrees");
        reject(() -> PreviewTransform.matrix(1600, 440, 1920, 1080, 90, 360), "Out-of-range display angle");
        System.out.println("Preview transform: " + checks + " checks passed across " + combinations + " geometry combinations");
    }

    private static void checkGeometry(int vw, int vh, int bw, int bh, int sensor, int display) {
        combinations++;
        String label = "view=" + vw + "x" + vh + " buffer=" + bw + "x" + bh
                + " sensor=" + sensor + " display=" + display;
        float[] matrix = PreviewTransform.matrix(vw, vh, bw, bh, sensor, display);
        require(matrix.length == 9, label + ": 3x3 matrix");
        for (float value : matrix) require(Float.isFinite(value), label + ": finite coefficient");
        near(matrix[6], 0, label + ": no perspective X");
        near(matrix[7], 0, label + ": no perspective Y");
        near(matrix[8], 1, label + ": homogeneous scale");

        // This oracle works in raw sensor pixels. Rotate the raw rectangle, fit it into
        // the view, and compare labeled corners with a separately simulated TextureView.
        int netTurns = ((sensor - display + 360) % 360) / 90;
        double rotatedWidth = netTurns % 2 == 0 ? bw : bh;
        double rotatedHeight = netTurns % 2 == 0 ? bh : bw;
        double fittedScale = Math.min(vw / rotatedWidth, vh / rotatedHeight);
        double left = (vw - rotatedWidth * fittedScale) / 2;
        double top = (vh - rotatedHeight * fittedScale) / 2;
        double right = vw - left;
        double bottom = vh - top;
        double[][] targetCorners = {{left, top}, {right, top}, {right, bottom}, {left, bottom}};
        double[][] rawCorners = {{0, 0}, {bw, 0}, {bw, bh}, {0, bh}};
        for (int corner = 0; corner < 4; corner++) {
            double[] actual = displayedPoint(matrix, rawCorners[corner][0], rawCorners[corner][1], vw, vh, bw, bh, sensor);
            double[] expected = targetCorners[(corner + netTurns) % 4];
            near(actual[0], expected[0], label + ": labeled corner X " + corner);
            near(actual[1], expected[1], label + ": labeled corner Y " + corner);
            require(actual[0] >= -PIXEL_TOLERANCE && actual[0] <= vw + PIXEL_TOLERANCE
                    && actual[1] >= -PIXEL_TOLERANCE && actual[1] <= vh + PIXEL_TOLERANCE,
                    label + ": full image stays inside the view");
        }
        require(Math.abs(left) < PIXEL_TOLERANCE || Math.abs(top) < PIXEL_TOLERANCE,
                label + ": fit uses all available space on one axis");
        double[] center = displayedPoint(matrix, bw / 2.0, bh / 2.0, vw, vh, bw, bh, sensor);
        near(center[0], vw / 2.0, label + ": centered X");
        near(center[1], vh / 2.0, label + ": centered Y");

        // Equal perpendicular pixel displacements must remain equal and perpendicular:
        // circles stay circular and squares stay square, including non-16:9 inputs.
        double step = Math.min(bw, bh) / 4.0;
        double[] x = displayedPoint(matrix, bw / 2.0 + step, bh / 2.0, vw, vh, bw, bh, sensor);
        double[] y = displayedPoint(matrix, bw / 2.0, bh / 2.0 + step, vw, vh, bw, bh, sensor);
        double ux = x[0] - center[0], uy = x[1] - center[1];
        double vx = y[0] - center[0], vy = y[1] - center[1];
        double xLength = Math.hypot(ux, uy), yLength = Math.hypot(vx, vy);
        near(xLength, step * fittedScale, label + ": uniform raw X scale");
        near(yLength, step * fittedScale, label + ": uniform raw Y scale");
        near((ux * vx + uy * vy) / (xLength * yLength), 0, label + ": orthogonal axes");
        require(ux * vy - uy * vx > 0, label + ": no unexpected reflection");

        // A 180-degree display change must be handled even if view dimensions do not change.
        float[] opposite = PreviewTransform.matrix(vw, vh, bw, bh, sensor, (display + 180) % 360);
        double[] reversed = displayedPoint(opposite, bw / 2.0 + step, bh / 2.0, vw, vh, bw, bh, sensor);
        near(reversed[0], vw - x[0], label + ": opposite display X");
        near(reversed[1], vh - x[1], label + ": opposite display Y");
    }

    private static double[] displayedPoint(float[] matrix, double x, double y,
            int vw, int vh, int bw, int bh, int sensor) {
        // Simulate Camera2's sensor-orientation compensation followed by TextureView's
        // default stretch to its bounds. These coordinates are the corrective matrix's input.
        double naturalX, naturalY;
        int naturalWidth, naturalHeight;
        switch (sensor) {
            case 0: naturalX = x; naturalY = y; naturalWidth = bw; naturalHeight = bh; break;
            case 90: naturalX = bh - y; naturalY = x; naturalWidth = bh; naturalHeight = bw; break;
            case 180: naturalX = bw - x; naturalY = bh - y; naturalWidth = bw; naturalHeight = bh; break;
            case 270: naturalX = y; naturalY = bw - x; naturalWidth = bh; naturalHeight = bw; break;
            default: throw new AssertionError("Invalid test sensor orientation");
        }
        double tx = naturalX * vw / naturalWidth;
        double ty = naturalY * vh / naturalHeight;
        return new double[]{matrix[0] * tx + matrix[1] * ty + matrix[2],
                matrix[3] * tx + matrix[4] * ty + matrix[5]};
    }

    private static double[] doubles(float[] values) {
        double[] result = new double[values.length];
        for (int i = 0; i < values.length; i++) result[i] = values[i];
        return result;
    }

    private static void assertMatrix(float[] actual, double[] expected, String message) {
        require(actual.length == expected.length, message + ": matrix length");
        for (int i = 0; i < expected.length; i++) near(actual[i], expected[i], message + ": coefficient " + i);
    }

    private static void reject(Runnable operation, String message) {
        boolean rejected = false;
        try { operation.run(); } catch (IllegalArgumentException expected) { rejected = true; }
        require(rejected, message + " must be rejected");
    }

    private static void near(double actual, double expected, String message) {
        require(Double.isFinite(actual) && Math.abs(actual - expected) <= PIXEL_TOLERANCE,
                message + ": expected " + expected + ", got " + actual);
    }

    private static void require(boolean condition, String message) {
        checks++;
        if (!condition) throw new AssertionError(message);
    }
}

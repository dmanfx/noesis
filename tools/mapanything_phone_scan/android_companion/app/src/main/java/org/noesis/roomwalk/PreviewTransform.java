package org.noesis.roomwalk;

/** Geometry for the camera-backed TextureView only; encoded image coordinates are unchanged. */
public final class PreviewTransform {
    private PreviewTransform() {}

    /**
     * Returns row-major values accepted by android.graphics.Matrix.setValues().
     *
     * TextureView already rotates a Camera2 buffer into the device's natural orientation,
     * then stretches it to the view bounds. This matrix reverses that stretch, compensates
     * display rotation, and fits the complete image uniformly around the view center.
     * Sensor orientation determines the dimensions of the already rotated image; it must
     * not be applied as another rotation here. Both angles are degrees, not Surface enums.
     *
     * All dimensions must be positive and angles must be 0, 90, 180, or 270. Call after
     * the preview buffer and view sizes are known, and again when display rotation changes.
     */
    public static float[] matrix(int viewWidth, int viewHeight, int bufferWidth, int bufferHeight,
            int sensorOrientationDegrees, int displayRotationDegrees) {
        if (viewWidth <= 0 || viewHeight <= 0 || bufferWidth <= 0 || bufferHeight <= 0)
            throw new IllegalArgumentException("Preview and view dimensions must be positive");
        requireRightAngle(sensorOrientationDegrees, "Sensor orientation");
        requireRightAngle(displayRotationDegrees, "Display rotation");

        boolean sensorSwapsAxes = sensorOrientationDegrees % 180 != 0;
        double naturalWidth = sensorSwapsAxes ? bufferHeight : bufferWidth;
        double naturalHeight = sensorSwapsAxes ? bufferWidth : bufferHeight;
        boolean displaySwapsAxes = displayRotationDegrees % 180 != 0;
        double displayedWidth = displaySwapsAxes ? naturalHeight : naturalWidth;
        double displayedHeight = displaySwapsAxes ? naturalWidth : naturalHeight;
        double fit = Math.min(viewWidth / displayedWidth, viewHeight / displayedHeight);

        // Exact quarter-turn coefficients avoid small trigonometric errors at the edges.
        int cos;
        int sin;
        switch (displayRotationDegrees) {
            case 0: cos = 1; sin = 0; break;
            case 90: cos = 0; sin = -1; break;
            case 180: cos = -1; sin = 0; break;
            case 270: cos = 0; sin = 1; break;
            default: throw new AssertionError("Validated rotation is not a quarter turn");
        }
        double undoX = naturalWidth / viewWidth;
        double undoY = naturalHeight / viewHeight;
        double a = fit * cos * undoX;
        double b = -fit * sin * undoY;
        double c = fit * sin * undoX;
        double d = fit * cos * undoY;
        double centerX = viewWidth / 2.0;
        double centerY = viewHeight / 2.0;
        double translateX = centerX - a * centerX - b * centerY;
        double translateY = centerY - c * centerX - d * centerY;
        return new float[]{
                (float) a, (float) b, (float) translateX,
                (float) c, (float) d, (float) translateY,
                0f, 0f, 1f
        };
    }

    private static void requireRightAngle(int degrees, String name) {
        if (degrees < 0 || degrees >= 360 || degrees % 90 != 0)
            throw new IllegalArgumentException(name + " must be 0, 90, 180, or 270 degrees");
    }
}

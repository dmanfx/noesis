package org.noesis.roomwalk;

import android.content.Context;
import android.content.SharedPreferences;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.os.Build;
import org.json.JSONObject;

/** The video and IMU-only recorders select the same actual native sensors. */
final class NativeSensors {
    static final String AXES = "android_device_x_right_y_up_z_out_of_screen";
    static Sensor accelerometer(SensorManager sensors) {
        Sensor sensor = sensors.getDefaultSensor(Sensor.TYPE_ACCELEROMETER_UNCALIBRATED);
        return sensor != null ? sensor : sensors.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
    }
    static Sensor gyroscope(SensorManager sensors) {
        Sensor sensor = sensors.getDefaultSensor(Sensor.TYPE_GYROSCOPE_UNCALIBRATED);
        return sensor != null ? sensor : sensors.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
    }
    static String deviceId(Context context) {
        SharedPreferences preferences = context.getSharedPreferences("native-device", 0);
        String id = preferences.getString("id", null);
        if (id == null) {
            id = "android-install:" + java.util.UUID.randomUUID();
            preferences.edit().putString("id", id).apply();
        }
        return id;
    }
    static JSONObject device(Context context) throws Exception {
        return new JSONObject().put("id", deviceId(context)).put("model", Build.MANUFACTURER + " " + Build.MODEL)
                .put("android_api_level", Build.VERSION.SDK_INT).put("build_fingerprint", Build.FINGERPRINT);
    }
    static String sensorId(Context context,Sensor sensor){return deviceId(context)+":sensor:"+sensor.getId()+":"+sensor.getType();}
    static JSONObject describe(Context context, Sensor sensor, String file, String units) throws Exception {
        boolean raw = sensor.getType() == Sensor.TYPE_ACCELEROMETER_UNCALIBRATED || sensor.getType() == Sensor.TYPE_GYROSCOPE_UNCALIBRATED;
        return new JSONObject().put("file", file).put("sensor_id", sensorId(context,sensor))
                .put("android_sensor_id", sensor.getId()).put("name", sensor.getName()).put("type", sensor.getType())
                .put("type_name", sensor.getStringType()).put("vendor", sensor.getVendor()).put("version", sensor.getVersion())
                .put("units", units).put("axes", AXES).put("uncalibrated", raw)
                .put("bias_fields_are_sensor_estimates", raw).put("bias_correction_applied", false)
                .put("requested_rate_hz", CaptureEngine.SENSOR_RATE_HZ).put("minimum_delay_us", sensor.getMinDelay())
                .put("maximum_range", sensor.getMaximumRange()).put("resolution", sensor.getResolution())
                .put("accelerometer_includes_gravity", units.equals("m/s^2"));
    }
    private NativeSensors() {}
}

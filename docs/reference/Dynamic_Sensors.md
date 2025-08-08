# Dynamic Sensor Management (nvmultiurisrcbin REST)

`DeepStreamVideoPipeline` provides helpers to add/remove sources at runtime using the REST API in `nvmultiurisrcbin`.

## Config

- Hosted on `ip-address` and `port` (`localhost:9000` by default).
- Set via element properties in the pipeline.

## Endpoints

- Add sensor
  - POST `http://localhost:<port>/stream`
  - Body: `{ "change": "add", "sensorId": "<int>", "uri": "<rtsp_or_file_uri>" }`
  - Python: `add_sensor(sensor_id, uri)`

- Remove sensor
  - POST `http://localhost:<port>/stream`
  - Body: `{ "change": "remove", "sensorId": "<int>", "uri": "" }`
  - Python: `remove_sensor(sensor_id)`

## Notes

- `sensorId` must be unique. On removal, the demux pad and its JPEG branch are torn down.
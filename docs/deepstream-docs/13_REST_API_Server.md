# 13. DeepStream REST API Server (DS8)

DeepStream 8 includes a REST server library used by sample apps and by bins like `nvmultiurisrcbin`. It enables runtime control of sources and select plugins.

## What it provides

- HTTP endpoints to list/add/remove/update sources and some plugin parameters.
- JSON payloads to configure sensors and behaviors (enable, URI, reconnect, FPS caps, etc.).
- Integration points for analytics/tracker in DS8 to expose limited runtime controls.

## Enabling in your app

- Some elements (e.g., `nvmultiurisrcbin`) can auto-start the REST server.
- Alternatively, link and start `nvds_rest_server` in your app (C/C++/Python bindings) and wire it to your graph state.

## Typical operations

- Add a source (POST)
- Remove a source (DELETE)
- List sources (GET)
- Update per-source properties (PATCH)

Example commands (adjust URL/port and payloads to match your build):

```
# Add a new RTSP source
curl -X POST http://localhost:9000/api/v1/sensors \
  -H 'Content-Type: application/json' \
  -d '{"id":"cam-01","uri":"rtsp://...","enable":true,"type":"rtsp"}'

# List sensors
curl http://localhost:9000/api/v1/sensors

# Remove a sensor
curl -X DELETE http://localhost:9000/api/v1/sensors/cam-01
```

## Notes

- Endpoints and payloads can vary between DS releases; inspect the running service for its OpenAPI/Swagger description if available, or check the sample sources under your DS install.
- Avoid excessive property churn on live pipelines; batch updates when possible to minimize re-negotiation and jitter.
- Secure the REST endpoint in production (bind to localhost or add authentication/proxy as needed).

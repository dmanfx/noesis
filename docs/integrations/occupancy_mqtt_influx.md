# MQTT and Influx integration security
_Status: validated against the active DS8, V3DT, and DS9 code on 2026-07-10._

## Current runtime truth

- No active runtime publishes occupancy to MQTT or InfluxDB. DS8, V3DT, and DS9
  explicitly bind the occupancy publisher slot to `None`.
- The active tree has no occupancy MQTT/Influx publisher implementation.
- `geometry/depth_publisher.py` contains an optional MapAnything depth-summary
  publisher, but no active module imports or constructs it.
- Canonical occupancy remains the authenticated Noesis tracking/world telemetry
  consumed through the appliance gateway.

Setting an integration flag does not wire a publisher into a runtime. Wiring is
an explicit future product change and must be validated in both DS8 and DS9.

## Credential contract

Secrets must never be stored in `config.py`, JSON config, command-line arguments,
or plaintext environment variables. `AppConfig.integrations` contains only
non-secret connection metadata, disabled sink flags, and optional secret-file
paths:

- MQTT enable: `ENABLE_DEPTH_DIAGNOSTICS_MQTT`
- MQTT metadata: `BASE_TOPIC`, `MQTT_HOST`, `MQTT_PORT`, `MQTT_USERNAME`,
  `MQTT_QOS`, `MQTT_RETAIN`
- MQTT credential path: `MQTT_PASSWORD_FILE`
- Influx enable: `ENABLE_DEPTH_DIAGNOSTICS_INFLUX`
- Influx metadata: `INFLUX_URL`, `INFLUX_ORG`, `INFLUX_BUCKET_RAW`
- Influx credential path: `INFLUX_TOKEN_FILE`

Deployment tooling may override only the paths with
`NOESIS_MQTT_PASSWORD_FILE` and `NOESIS_INFLUX_TOKEN_FILE`. Raw
`NOESIS_MQTT_PASSWORD` and `NOESIS_INFLUX_TOKEN` values are rejected.

The credential reader does not create a secret or repair its permissions. It
accepts only an existing UTF-8, single-line regular file that:

- is owned by the effective service user;
- has exactly mode `0400` or `0600`;
- is not a symlink and has exactly one hard link;
- contains a non-blank secret of at least 16 characters;
- is no larger than 16 KiB.

The open uses no-follow semantics and verifies the opened inode against the
inspected inode. Missing, insecure, ambiguous, or swapped files fail closed.
No parent directory or secret-file permission is changed at read time.

## Provisioning and rotation

Create the parent directory and credential files with an owner-controlled
provisioning tool or editor. Verify metadata without displaying contents:

```bash
install -d -m 700 "$HOME/.config/noesis/secrets"
stat -c '%a %U %F' "$HOME/.config/noesis/secrets/influx-token"
stat -c '%a %U %F' "$HOME/.config/noesis/secrets/mqtt-password"
```

The two files must report mode `400` or `600`, the runtime user as owner, and
`regular file`. Never place a real credential directly in a shell command,
source file, URL, log, or test fixture.

Removing an embedded source default does not revoke an already-issued
credential. Rotate deployed MQTT and Influx credentials out of band, update the
private files, and restart only during an approved cutover window. This code
change does not rotate credentials or restart live services.

The hardening audit also found the two retired defaults in a stale ignored
`crash.log`, produced in 2025 when the deprecated runtime logged the complete
configuration object. Only the exact credential byte sequences were redacted;
the remaining forensic log and its timestamp were preserved, and its mode was
tightened to `0600`. Current code must never log a complete configuration object
that could contain credential material.

## Explicit future activation

`DepthDiagnosticsPublisher.from_settings(...)` is the supported construction
boundary. When both sink flags are false, it returns without reading either
credential file. If a sink is true, its private credential, Python dependency,
and client initialization are mandatory. Multi-sink startup is transactional:
if either enabled sink fails, any client already initialized by that attempt is
closed and startup raises an error.

Transient publish calls currently report only the exception class, never raw
client exception text that might contain authentication material.

## Related references

- `docs/Occupancy_Publishing.md` describes the current DS8 telemetry contract.
- `docs/Integrations_Playbook.md` records the deliberate non-wiring boundary.
- `tests/test_depth_diagnostics_credentials.py` covers secure-file and
  fail-closed behavior.

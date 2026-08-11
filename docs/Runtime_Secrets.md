# Runtime Secrets

_Status: validated against DS8, V3DT, and DS9 source on 2026-07-10._

Active Noesis runtime configuration is public configuration. It contains stable
references, never camera locators or service credentials. The corresponding
values live in owner-only appliance state:

- Camera source registry:
  `~/.local/state/noesis/secrets/camera_sources.json`
- MapAnything RPC key:
  `~/.local/state/noesis/secrets/mapanything_rpc.key`

Both files must be owned by the service user, mode `0600`, single-link regular
files in a mode-`0700` directory. Existing paths are validated, never chmodded,
and symlinks, hardlinks, shared directories, malformed values, missing
references, and inline RTSP sources fail closed. Optional path overrides are
`NOESIS_CAMERA_SECRETS_FILE` and `NOESIS_MAPANYTHING_API_KEY_FILE`.

## Provisioning and rotation

For a new appliance checkout whose pipeline files already contain
`uri_secret` references, enter each URI without terminal echo:

```bash
python3 scripts/provision_runtime_secrets.py --interactive-camera-input
```

To atomically replace the camera registry or rotate only the MapAnything key:

```bash
python3 scripts/provision_runtime_secrets.py \
  --interactive-camera-input \
  --replace-camera-secrets
python3 scripts/provision_runtime_secrets.py --rotate-mapanything-key-only
```

The tool never accepts secret values on the command line and never prints
them. It does not restart a live process. A MapAnything-key rotation therefore
requires a coordinated restart of the service and its clients; do not rotate
one side while the other must remain online.

## Runtime and artifact behavior

- DS8, V3DT, and DS9 resolve `sources[*].uri_secret` only in process memory.
- `uri_secret` is removed before a source property map reaches Service Maker or
  GStreamer.
- Effective pipeline YAML, dev-console launch artifacts, source catalogs,
  depth-registration provenance, and world/config fingerprints use the public
  reference form. Camera credential rotation does not stale model or pipeline
  fingerprints.
- Complete RTSP locators are removed from application-owned exception text.
  SDK-native logging remains outside this guarantee, so logs must still be
  handled as appliance-private operational data.
- The MapAnything service requires an owner-only, authority-grade URL-safe key;
  there is no built-in key and no unauthenticated inference mode.

The old `config.py` application dataclass is a deprecated pre-DS8 artifact and
still contains legacy inline camera locators. Active MapAnything configuration
no longer imports it, and DS8/DS9 runtime code does not use it as a source
authority. Do not copy those values into active config; removing that deprecated
artifact is a separate explicitly authorized cleanup.

## Validation

```bash
python3 -m pytest \
  tests/test_runtime_secrets.py \
  tests/test_ma_service.py \
  tests/test_noesis_core_runtime_world.py \
  tests/test_depth_registration.py -q
```

The focused tests cover missing/insecure state, unsafe parents, symlinks,
hardlinks, weak keys, inline values, unknown references, DS8/DS9 plugin-property
parity, serialization/provenance redaction, and secret-independent
fingerprints.

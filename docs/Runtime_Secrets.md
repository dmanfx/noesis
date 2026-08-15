# Runtime secrets

Status: native DS9.1 boundary, 2026-08-15.

Version-controlled config contains stable secret references, never camera
locators or credentials. The native host supervisor requires three distinct
owner-only files:

| Variable | Content |
| --- | --- |
| `NOESIS_CAMERA_SECRETS_FILE` | Camera ID → RTSP URI registry |
| `NOESIS_MAPANYTHING_API_KEY_FILE` | MapAnything service key |
| `NOESIS_INTERNAL_AUTH_TOKEN_FILE` | Noesis↔Menon bearer token |

Each must be a regular, single-link, mode-`0600` file beneath a private
mode-`0700` directory. Symlinks, hardlinks, unsafe parents, malformed values,
missing references, inline RTSP sources, and weak keys fail closed. The
supervisor validates them without changing permissions.

## Provisioning and rotation

Enter camera URIs without terminal echo:

```bash
python3 scripts/provision_runtime_secrets.py --interactive-camera-input
```

Replace the registry or rotate the MapAnything key atomically:

```bash
python3 scripts/provision_runtime_secrets.py \
  --interactive-camera-input --replace-camera-secrets
python3 scripts/provision_runtime_secrets.py --rotate-mapanything-key-only
```

The tool does not restart services. Coordinate a bounded service restart after
rotation so both sides use the same value.

## Runtime behavior

- `sources[*].uri_secret` is resolved only in process memory and removed before
  source properties reach Service Maker/GStreamer.
- Effective configs, fingerprints, evidence, and logs retain only public
  references where the application controls serialization.
- The browser never receives the internal token. Menon exchanges authenticated
  browser sessions/tickets for loopback Noesis access.
- Legacy `config.py` inline values have no authority and must not be copied into
  DS9.1 config.

Use focused secret and native-supervisor tests after changing this boundary; do
not stage a release merely to validate a private-file parser.

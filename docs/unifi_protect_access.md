# Local UniFi Protect recording access

Status: local acquisition workflow documented 2026-10-15.

The host-local operating guide is stored outside the repository at:

```text
${XDG_CONFIG_HOME:-$HOME/.config}/noesis/unifi-protect/README.md
```

Read that guide before using the NVR. It records the local address and camera
IDs, certificate fingerprint, GNOME Keyring lookup, interactive MFA handling,
API routes, acquisition evidence and remaining timing/calibration limits.
Keep passwords, API keys, MFA codes and session cookies out of documentation,
source control and logs.

The included [local export helper](../unifi_protect_export_20261015.py) exports a
fixed date/time window; it does not have a generic date/time CLI. The local guide
describes its host-specific configuration and how to adapt it for another
authorized recording request. Credentials remain outside the repository.

Both expected Kitchen and Family Room files for the 2026-10-15 morning interval
were present and readable by `ffprobe`. Metadata inspection does not prove
full decode integrity, exact camera synchronization or calibration suitability.
NVR recordings remain offline evaluation inputs until those relevant checks
are performed.

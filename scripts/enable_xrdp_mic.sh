#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'EOF'
Usage: enable_xrdp_mic.sh [--user <name>]

Enables XRDP microphone redirection on Ubuntu by:
  - Ensuring the PipeWire XRDP autostart file exists for the user
  - Enabling needed XRDP channels in /etc/xrdp/xrdp.ini
  - Restarting xrdp services

If --user is omitted, the script uses $SUDO_USER or logname.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

if [ "$(id -u)" -ne 0 ]; then
  echo "Elevating to root (sudo required)..." >&2
  exec sudo -E bash "$0" "$@"
fi

TARGET_USER="${TARGET_USER:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --user)
      shift
      TARGET_USER="${1:-}"
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
  shift || true
done

if [ -z "$TARGET_USER" ]; then
  TARGET_USER="${SUDO_USER:-}"
fi

if [ -z "$TARGET_USER" ]; then
  TARGET_USER="$(logname 2>/dev/null || true)"
fi

[ -n "$TARGET_USER" ] || die "Unable to determine target user; rerun with --user <name>."

TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
[ -n "$TARGET_HOME" ] || die "User not found: $TARGET_USER"

XRDP_INI="/etc/xrdp/xrdp.ini"
[ -f "$XRDP_INI" ] || die "Missing $XRDP_INI"

if ! grep -q "^[[]Channels[]]" "$XRDP_INI"; then
  die "Missing [Channels] section in $XRDP_INI"
fi

backup="${XRDP_INI}.bak.$(date +%Y%m%d%H%M%S)"
cp -a "$XRDP_INI" "$backup"
echo "Backed up $XRDP_INI to $backup"

if grep -qE "^[[:space:]]*allow_channels[[:space:]]*=" "$XRDP_INI"; then
  perl -0pi -e 's/^[[:space:]]*allow_channels[[:space:]]*=.*$/allow_channels=true/m' "$XRDP_INI"
else
  perl -0pi -e 's/^[[]Globals[]]\n/[Globals]\nallow_channels=true\n/m' "$XRDP_INI"
fi

set_channel_bool() {
  local key="$1"
  local value="$2"

  if grep -qE "^[[:space:]]*${key}[[:space:]]*=" "$XRDP_INI"; then
    perl -0pi -e "s/^[[:space:]]*${key}[[:space:]]*=.*\$/${key}=${value}/m" "$XRDP_INI"
  else
    perl -0pi -e "s/^[[]Channels[]]\n/[Channels]\n${key}=${value}\n/m" "$XRDP_INI"
  fi
}

set_channel_bool "rdpsnd" "true"
set_channel_bool "drdynvc" "true"
set_channel_bool "audin" "true"

AUTOSTART_SRC="/etc/xdg/autostart/pipewire-xrdp.desktop"
AUTOSTART_DIR="${TARGET_HOME}/.config/autostart"
AUTOSTART_DST="${AUTOSTART_DIR}/pipewire-xrdp.desktop"

if [ -f "$AUTOSTART_SRC" ]; then
  install -d -m 0755 "$AUTOSTART_DIR"
  cp -f "$AUTOSTART_SRC" "$AUTOSTART_DST"
  chown "$TARGET_USER":"$TARGET_USER" "$AUTOSTART_DIR" "$AUTOSTART_DST"
  echo "Installed user autostart: $AUTOSTART_DST"
else
  echo "WARN: $AUTOSTART_SRC not found. Install 'pipewire-module-xrdp' if needed." >&2
fi

systemctl restart xrdp xrdp-sesman
systemctl is-active --quiet xrdp || die "xrdp is not active after restart."
systemctl is-active --quiet xrdp-sesman || die "xrdp-sesman is not active after restart."

cat <<'EOF'
Done. Reconnect your RDP session, then:
  - In Windows RDP client: Remote audio -> Settings -> Remote audio recording -> Record from this computer
  - In the XRDP session: check Input Devices (pavucontrol) for "xrdp-source"
EOF

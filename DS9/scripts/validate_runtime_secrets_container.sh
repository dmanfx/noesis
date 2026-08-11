#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DS9_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${DS9_ROOT}/.." && pwd)"
IMAGE_REF="noesis-ds9-runtime:9.0-20260710"
IMAGE_ID="sha256:ca33b4c6a84fc56b86b71feee2a444299cb2ce7f33ab018cae43ac730aaef5fc"
DOCKER_ROOT_RAW="${NOESIS_DS9_DOCKER_ROOT:-}"
CAMERA_FILE="${NOESIS_CAMERA_SECRETS_FILE:-${HOME}/.local/state/noesis/secrets/camera_sources.json}"
MAPANYTHING_FILE="${NOESIS_MAPANYTHING_API_KEY_FILE:-${HOME}/.local/state/noesis/secrets/mapanything_rpc.key}"
INTERNAL_AUTH_FILE="${NOESIS_INTERNAL_AUTH_TOKEN_FILE:-${HOME}/.local/state/noesis/gateway-token}"

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

[[ -n "${DOCKER_ROOT_RAW}" && "${DOCKER_ROOT_RAW}" == /* ]] \
  || fail "NOESIS_DS9_DOCKER_ROOT must be an explicit absolute path"
DOCKER_ROOT="$(realpath -m -- "${DOCKER_ROOT_RAW}")"
[[ -S "${DOCKER_ROOT}/run/docker.sock" ]] \
  || fail "secondary Docker socket is unavailable"
[[ -f "${CAMERA_FILE}" && ! -L "${CAMERA_FILE}" ]] \
  || fail "camera source secret file is unavailable or linked"
[[ -f "${MAPANYTHING_FILE}" && ! -L "${MAPANYTHING_FILE}" ]] \
  || fail "MapAnything RPC key file is unavailable or linked"
[[ -f "${INTERNAL_AUTH_FILE}" && ! -L "${INTERNAL_AUTH_FILE}" ]] \
  || fail "internal auth token file is unavailable or linked"

NOESIS_CAMERA_SECRETS_FILE="${CAMERA_FILE}" \
NOESIS_MAPANYTHING_API_KEY_FILE="${MAPANYTHING_FILE}" \
NOESIS_INTERNAL_AUTH_TOKEN_FILE="${INTERNAL_AUTH_FILE}" \
python3 - <<'PY'
from noesis_core.runtime_secrets import load_camera_uri_registry, load_mapanything_api_key
from noesis.server.internal_auth import load_internal_token
import os

assert load_camera_uri_registry()
assert load_mapanything_api_key()
assert load_internal_token(os.environ["NOESIS_INTERNAL_AUTH_TOKEN_FILE"])
PY

export DOCKER_HOST="unix://${DOCKER_ROOT}/run/docker.sock"
[[ "$(docker info --format '{{.DefaultRuntime}}')" == "runc" ]] \
  || fail "secondary Docker default runtime must remain runc"
actual_image_id="$(docker image inspect "${IMAGE_REF}" --format '{{.Id}}')"
[[ "${actual_image_id}" == "${IMAGE_ID}" ]] \
  || fail "canonical DS9 runtime image drifted: expected=${IMAGE_ID} actual=${actual_image_id}"

uid="$(id -u)"
gid="$(id -g)"
docker run --rm \
  --runtime=runc \
  --network=none \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --user "${uid}:${gid}" \
  --env HOME=/tmp/noesis-home \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --env NOESIS_CAMERA_SECRETS_FILE=/run/noesis-secrets/camera_sources.json \
  --env NOESIS_MAPANYTHING_API_KEY_FILE=/run/noesis-secrets/mapanything_rpc.key \
  --env NOESIS_INTERNAL_AUTH_TOKEN_FILE=/run/noesis-secrets/gateway-token \
  --tmpfs "/tmp:rw,exec,nosuid,nodev,size=67108864,uid=${uid},gid=${gid},mode=0700" \
  --tmpfs "/run/noesis-secrets:rw,noexec,nosuid,nodev,size=65536,uid=${uid},gid=${gid},mode=0700" \
  --mount "type=bind,src=${REPO_ROOT},dst=/workspace,readonly" \
  --mount "type=bind,src=${CAMERA_FILE},dst=/run/noesis-secrets/camera_sources.json,readonly" \
  --mount "type=bind,src=${MAPANYTHING_FILE},dst=/run/noesis-secrets/mapanything_rpc.key,readonly" \
  --mount "type=bind,src=${INTERNAL_AUTH_FILE},dst=/run/noesis-secrets/gateway-token,readonly" \
  --workdir /workspace \
  --entrypoint python3 \
  "${IMAGE_ID}" \
  -c 'import os; from pathlib import Path; from noesis_core.runtime_secrets import load_camera_uri_registry, load_mapanything_api_key; from noesis.server.internal_auth import load_internal_token; assert not list(Path("/dev").glob("nvidia*")); assert load_camera_uri_registry(); assert load_mapanything_api_key(); assert load_internal_token(os.environ["NOESIS_INTERNAL_AUTH_TOKEN_FILE"]); print("[OK] DS9 no-GPU runtime secret mounts validated")'

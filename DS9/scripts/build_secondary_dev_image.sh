#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS9_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="${NOESIS_DS9_DEV_IMAGE:-noesis-ds9-dev:9.0-20260710}"

if [[ -z "${NOESIS_DS9_DOCKER_ROOT:-}" ]]; then
  echo "[FAIL] NOESIS_DS9_DOCKER_ROOT must name the explicit secondary Docker staging root." >&2
  exit 2
fi

"${SCRIPT_DIR}/secondary_docker.sh" start
export DOCKER_HOST="unix://${NOESIS_DS9_DOCKER_ROOT}/run/docker.sock"

if [[ "$(docker info --format '{{.DefaultRuntime}}')" != "runc" ]]; then
  echo "[FAIL] Secondary Docker must use runc by default for the no-GPU build." >&2
  exit 3
fi

DOCKER_BUILDKIT="${NOESIS_DS9_DOCKER_BUILDKIT:-0}" docker build \
  --network=host \
  --file "${DS9_ROOT}/docker/Dockerfile" \
  --tag "${IMAGE}" \
  "${DS9_ROOT}/docker"

docker image inspect "${IMAGE}" --format \
  'id={{.Id}} created={{.Created}} size={{.Size}} base_digest={{index .Config.Labels "org.opencontainers.image.base.digest"}}'

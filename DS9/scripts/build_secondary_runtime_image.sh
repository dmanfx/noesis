#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS9_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_IMAGE_REF="noesis-ds9-dev:9.0-20260710"
PARENT_IMAGE_ID="sha256:7476b1021376cd67793c95d949cdc7d46eef7704ab98a5a76feed461e4f907a4"
RUNTIME_IMAGE="${NOESIS_DS9_RUNTIME_IMAGE:-noesis-ds9-runtime:9.0-20260710}"

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

actual_parent_id="$(docker image inspect "${PARENT_IMAGE_REF}" --format '{{.Id}}')"
if [[ "${actual_parent_id}" != "${PARENT_IMAGE_ID}" ]]; then
  echo "[FAIL] Runtime parent image drifted: expected=${PARENT_IMAGE_ID} actual=${actual_parent_id}" >&2
  exit 4
fi

DOCKER_BUILDKIT="${NOESIS_DS9_DOCKER_BUILDKIT:-0}" docker build \
  --pull=false \
  --network=host \
  --file "${DS9_ROOT}/docker/Dockerfile.runtime" \
  --tag "${RUNTIME_IMAGE}" \
  "${DS9_ROOT}/docker"

runtime_parent_reference="$(docker image inspect "${RUNTIME_IMAGE}" --format '{{index .Config.Labels "com.noesis.engine-build.image.reference"}}')"
runtime_parent_id="$(docker image inspect "${RUNTIME_IMAGE}" --format '{{index .Config.Labels "com.noesis.engine-build.image.id"}}')"
if [[ "${runtime_parent_reference}" != "${PARENT_IMAGE_REF}" || "${runtime_parent_id}" != "${PARENT_IMAGE_ID}" ]]; then
  echo "[FAIL] Runtime image parent labels do not match the exact build authority." >&2
  exit 5
fi

mapfile -t parent_layers < <(
  docker image inspect "${PARENT_IMAGE_ID}" --format '{{range .RootFS.Layers}}{{println .}}{{end}}'
)
mapfile -t runtime_layers < <(
  docker image inspect "${RUNTIME_IMAGE}" --format '{{range .RootFS.Layers}}{{println .}}{{end}}'
)
[[ -n "${parent_layers[-1]:-}" ]] || unset 'parent_layers[-1]'
[[ -n "${runtime_layers[-1]:-}" ]] || unset 'runtime_layers[-1]'
if [[ "${#parent_layers[@]}" -eq 0 || "${#runtime_layers[@]}" -ne "$(( ${#parent_layers[@]} + 2 ))" ]]; then
  echo "[FAIL] Runtime image must add exactly two layers to a non-empty build-image RootFS." >&2
  exit 6
fi
for index in "${!parent_layers[@]}"; do
  if [[ "${runtime_layers[${index}]}" != "${parent_layers[${index}]}" ]]; then
    echo "[FAIL] Runtime image RootFS does not have the exact build-image layer prefix." >&2
    exit 7
  fi
done

docker image inspect "${RUNTIME_IMAGE}" --format \
  'id={{.Id}} created={{.Created}} size={{.Size}} base_digest={{index .Config.Labels "org.opencontainers.image.base.digest"}} parent_reference={{index .Config.Labels "com.noesis.engine-build.image.reference"}} parent_id={{index .Config.Labels "com.noesis.engine-build.image.id"}}'

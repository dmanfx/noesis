#!/usr/bin/env bash
set -euo pipefail

umask 027

IMAGE="${NOESIS_DS9_IMAGE:-nvcr.io/nvidia/deepstream:9.0-triton-multiarch}"
ROOT_RAW="${NOESIS_DS9_DOCKER_ROOT:-}"

if [[ -z "${ROOT_RAW}" ]]; then
  echo "[FAIL] NOESIS_DS9_DOCKER_ROOT must name an explicit machine-local staging root." >&2
  exit 2
fi
if [[ "${ROOT_RAW}" != /* ]]; then
  echo "[FAIL] NOESIS_DS9_DOCKER_ROOT must be an absolute path: ${ROOT_RAW}" >&2
  exit 2
fi

ROOT="$(realpath -m -- "${ROOT_RAW}")"
if [[ "${ROOT}" == "/" || "${ROOT}" == "/var" || "${ROOT}" == "/var/lib" || "${ROOT}" == "/var/lib/docker" ]]; then
  echo "[FAIL] Refusing unsafe Docker staging root: ${ROOT}" >&2
  exit 2
fi

DATA_ROOT="${ROOT}/data"
EXEC_ROOT="${ROOT}/exec"
RUN_ROOT="${ROOT}/run"
SOCKET="${RUN_ROOT}/docker.sock"
PIDFILE="${RUN_ROOT}/dockerd.pid"
NAMESPACE="noesis-ds9"
PLUGINS_NAMESPACE="noesis-ds9-plugins"
UNIT="noesis-ds9-secondary-docker.service"
DOCKER=(docker --host "unix://${SOCKET}")

usage() {
  cat <<'EOF'
Usage: NOESIS_DS9_DOCKER_ROOT=<absolute-staging-root> \
  DS9/scripts/secondary_docker.sh <start|status|pull|inspect|env|stop>

The secondary daemon has its own Unix socket, data root, exec root, PID file,
and containerd namespaces. It creates no bridge and may not edit iptables,
IPv6 tables, forwarding, or masquerade state.
EOF
}

primary_root() {
  docker --host unix:///run/docker.sock info --format '{{.DockerRootDir}}' 2>/dev/null || true
}

reject_primary_root() {
  local primary
  primary="$(primary_root)"
  if [[ -n "${primary}" && "$(realpath -m -- "${primary}")" == "${DATA_ROOT}" ]]; then
    echo "[FAIL] Secondary data root resolves to the primary Docker root: ${DATA_ROOT}" >&2
    exit 3
  fi
}

is_ready() {
  "${DOCKER[@]}" info >/dev/null 2>&1
}

verify_isolation() {
  local actual_root bridge_count
  actual_root="$("${DOCKER[@]}" info --format '{{.DockerRootDir}}')"
  if [[ "$(realpath -m -- "${actual_root}")" != "${DATA_ROOT}" ]]; then
    echo "[FAIL] Secondary daemon reported an unexpected data root: ${actual_root}" >&2
    exit 4
  fi
  bridge_count="$("${DOCKER[@]}" network ls --filter driver=bridge --format '{{.ID}}' | wc -l)"
  if [[ "${bridge_count}" != "0" ]]; then
    echo "[FAIL] Secondary daemon unexpectedly exposes a bridge network." >&2
    exit 4
  fi
}

start_daemon() {
  reject_primary_root
  if is_ready; then
    verify_isolation
    echo "[OK] Secondary Docker is already ready at unix://${SOCKET}"
    return
  fi

  sudo -n true
  sudo -n install -d -m 0750 -o root -g docker \
    "${ROOT}" "${DATA_ROOT}" "${EXEC_ROOT}" "${RUN_ROOT}"

  sudo -n systemd-run \
    --unit="${UNIT}" \
    --collect \
    --property=Description="Noesis DS9 isolated staging Docker" \
    --property=Restart=no \
    --property=KillMode=mixed \
    --property=TimeoutStopSec=45s \
    /usr/bin/dockerd \
      --host="unix://${SOCKET}" \
      --data-root="${DATA_ROOT}" \
      --exec-root="${EXEC_ROOT}" \
      --pidfile="${PIDFILE}" \
      --bridge=none \
      --iptables=false \
      --ip6tables=false \
      --ip-forward=false \
      --ip-masq=false \
      --userland-proxy=false \
      --containerd-namespace="${NAMESPACE}" \
      --containerd-plugins-namespace="${PLUGINS_NAMESPACE}" \
      --storage-driver=overlay2 \
      --log-level=info >/dev/null

  local attempt
  for attempt in $(seq 1 60); do
    if is_ready; then
      verify_isolation
      echo "[OK] Secondary Docker ready at unix://${SOCKET}"
      return
    fi
    sleep 0.5
  done

  echo "[FAIL] Secondary Docker did not become ready." >&2
  sudo -n journalctl --unit="${UNIT}" --no-pager --lines=80 >&2 || true
  exit 5
}

status_daemon() {
  if ! is_ready; then
    echo "[STOPPED] No secondary Docker API at unix://${SOCKET}"
    return 1
  fi
  verify_isolation
  "${DOCKER[@]}" info --format \
    'id={{.ID}} root={{.DockerRootDir}} driver={{.Driver}} containers={{.Containers}} images={{.Images}} default_runtime={{.DefaultRuntime}}'
  "${DOCKER[@]}" network ls --format '{{.Name}}\t{{.Driver}}\t{{.Scope}}'
  "${DOCKER[@]}" images --digests --no-trunc \
    --format '{{.Repository}}:{{.Tag}}\t{{.Digest}}\t{{.ID}}\t{{.Size}}'
}

pull_image() {
  if ! is_ready; then
    echo "[FAIL] Start the secondary daemon before pulling." >&2
    exit 6
  fi
  verify_isolation
  "${DOCKER[@]}" pull "${IMAGE}"
  "${DOCKER[@]}" image inspect "${IMAGE}" --format \
    'id={{.Id}} created={{.Created}} os={{.Os}} arch={{.Architecture}} size={{.Size}} repo_digests={{json .RepoDigests}}'
}

inspect_image() {
  if ! is_ready; then
    echo "[FAIL] Start the secondary daemon before inspecting images." >&2
    exit 6
  fi
  verify_isolation
  "${DOCKER[@]}" image inspect "${IMAGE}"
}

stop_daemon() {
  sudo -n true
  if systemctl list-unit-files --type=service 2>/dev/null | awk '{print $1}' | grep -Fxq "${UNIT}" \
      || systemctl is-active --quiet "${UNIT}" 2>/dev/null; then
    sudo -n systemctl stop "${UNIT}"
  elif [[ -r "${PIDFILE}" ]]; then
    local pid command_line
    pid="$(<"${PIDFILE}")"
    if [[ ! "${pid}" =~ ^[0-9]+$ || ! -r "/proc/${pid}/cmdline" ]]; then
      echo "[FAIL] Refusing an invalid or stale secondary PID file: ${PIDFILE}" >&2
      exit 7
    fi
    command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
    if [[ "${command_line}" != *"--data-root=${DATA_ROOT}"* || "${command_line}" != *"--host=unix://${SOCKET}"* ]]; then
      echo "[FAIL] PID ${pid} does not own the expected secondary daemon; refusing to signal it." >&2
      exit 7
    fi
    sudo -n kill -TERM "${pid}"
  else
    echo "[OK] Secondary Docker is already stopped."
    return
  fi

  local attempt
  for attempt in $(seq 1 90); do
    if ! is_ready; then
      echo "[OK] Secondary Docker stopped; staged images remain under ${ROOT}."
      return
    fi
    sleep 0.5
  done
  echo "[FAIL] Secondary Docker did not stop within 45 seconds." >&2
  exit 8
}

case "${1:-}" in
  start) start_daemon ;;
  status) status_daemon ;;
  pull) pull_image ;;
  inspect) inspect_image ;;
  env) printf 'export DOCKER_HOST=%q\n' "unix://${SOCKET}" ;;
  stop) stop_daemon ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac

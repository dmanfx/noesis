#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DS9_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${DS9_ROOT}/.." && pwd)"
REQUIRED_IMAGE_REF="noesis-ds9-dev:9.0-20260710"
IMAGE="${NOESIS_DS9_DEV_IMAGE:-${REQUIRED_IMAGE_REF}}"
REQUIRED_IMAGE_ID="sha256:7476b1021376cd67793c95d949cdc7d46eef7704ab98a5a76feed461e4f907a4"
REQUIRED_DRIVER_MAJOR=590
GPU_DEVICE_INDEX=0
FINALIZER="${SCRIPT_DIR}/finalize_engine_realization.py"
GPU_OWNERSHIP_HELPER="${SCRIPT_DIR}/verify_gpu_process_ownership.py"
GPU_MEMORY_SAMPLER="${SCRIPT_DIR}/nvml_gpu_memory_sampler.py"
GPU_MEMORY_SAMPLE_INTERVAL_MS=25
GPU_MEMORY_MAX_GAP_MS=250
# Engine compilation is bursty. Keep 7 GiB outside this cgroup on the current
# 31 GiB swapless host; the runtime container's separate steady-state cap is
# intentionally higher.
MAINTENANCE_MEMORY_BYTES=25769803776
MAINTENANCE_PIDS_LIMIT=512
MAINTENANCE_LOG_MAX_SIZE="16m"
MAINTENANCE_LOG_MAX_FILES="2"

declare -Ar OUTPUTS=(
  [yolo11_seg]="yolo11s-seg_cust_fused.engine"
  [yolo26_m]="yolo26m_b3_fp16.engine"
  [yolo26_seg_s]="yolo26s-seg_fused_b3_fp16.engine"
  [reid_swin]="reid_swin_tiny_aicity156_dyn_b16_fp16.engine"
  [yolo26_pose_n]="yolo26n-pose_b3_fp16.engine"
  [depth_anything_v2_tracking]="depth_anything_v2_metric_hypersim_vits_294x518_b3_fp16.engine"
  [mapanything]="mapanything_images_294x518_b3_fp32.plan"
  [wholebody49_s_masks]="deimv2_wholebody49_dinov3_s_masks_640_b3_fp16.engine"
  [wholebody49_x_boxes]="deimv2_wholebody49_dinov3_x_boxes_640_b3_fp16.engine"
  [bodypose3dnet]="bodypose3dnet_accuracy_b1_fp16.engine"
  [v3dt_tracker_reid]="tracker_reid_resnet50_market1501_b32_fp16.engine"
)
declare -Ar TEMP_BUDGET_BYTES=(
  [yolo11_seg]=268435456
  [yolo26_m]=536870912
  [yolo26_seg_s]=268435456
  [reid_swin]=536870912
  [yolo26_pose_n]=134217728
  [depth_anything_v2_tracking]=536870912
  [mapanything]=4294967296
  [wholebody49_s_masks]=268435456
  [wholebody49_x_boxes]=805306368
  [bodypose3dnet]=268435456
  [v3dt_tracker_reid]=536870912
)
CANONICAL_ORDER=(yolo26_m reid_swin yolo26_pose_n depth_anything_v2_tracking mapanything)

usage() {
  cat <<'EOF'
Usage:
  NOESIS_DS9_DOCKER_ROOT=<absolute-secondary-docker-root> \
  NOESIS_DS9_ARTIFACT_ROOT=<absolute-artifact-root> \
    DS9/scripts/run_canonical_engine_maintenance.sh --only <engine> [--plan]

  NOESIS_DS9_DOCKER_ROOT=... NOESIS_DS9_ARTIFACT_ROOT=... \
    DS9/scripts/run_canonical_engine_maintenance.sh --all [--plan]

  NOESIS_DS9_DOCKER_ROOT=... NOESIS_DS9_ARTIFACT_ROOT=... \
    DS9/scripts/run_canonical_engine_maintenance.sh --v3dt [--plan]

Canonical engine names:
  yolo26_m
  reid_swin
  yolo26_pose_n
  depth_anything_v2_tracking
  mapanything

Explicit alternate PGIE engine names (never canonical fallbacks):
  yolo11_seg

Promoted Wholebody49 profile engine names (use --only):
  wholebody49_s_masks
  wholebody49_x_boxes

DS9 V3DT tracker engine names (build in this order, using --only):
  yolo26_seg_s
  bodypose3dnet
  v3dt_tracker_reid

--plan performs the exact rebuild-script dry run with runc and no GPU devices.
An actual build refuses to start while any GPU compute process exists.
EOF
}

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

ACTIVE_CONTAINER=""
ACTIVE_CONTAINER_ID=""
ACTIVE_CONTAINER_TRANSACTION_ID=""
ACTIVE_CONTAINER_ARTIFACT_ROOT_ID=""
ACTIVE_LAUNCH_ATTEMPTED=0
ACTIVE_ENGINE=""
ACTIVE_CONTAINER_INIT_PID=""
ACTIVE_CONTAINER_INIT_START_TIME_TICKS=""
ACTIVE_GPU_MEMORY_GUARD_MIB=""
ACTIVE_GPU_MEMORY_PEAK_MIB=0
ACTIVE_GPU_MEMORY_SAMPLE_COUNT=0
ACTIVE_GPU_MEMORY_SAMPLER_PID=""
ACTIVE_GPU_MEMORY_SAMPLER_START_TIME_TICKS=""
ACTIVE_GPU_MEMORY_EVIDENCE=""
ACTIVE_GPU_MEMORY_EVIDENCE_SHA256=""
ACTIVE_GPU_MEMORY_FIRST_SAMPLE_UTC=""
ACTIVE_GPU_MEMORY_LAST_SAMPLE_UTC=""
ACTIVE_GPU_MEMORY_MAXIMUM_GAP_MS=""
ACTIVE_GPU_MEMORY_BREACH_UTC=""
ACTIVE_GPU_MEMORY_BREACH_MIB=""
ACTIVE_GPU_MEMORY_CONTAINER_ID=""
ACTIVE_GPU_MEMORY_WRAPPER_START_TIME_TICKS=""
ACTIVE_TRANSACTION_MANIFEST=""
ACTIVE_TRANSACTION_SHA256=""
ACTIVE_TRANSACTION_REASON="wrapper exited before host transaction commit"
ACTIVE_GPU_MEMORY_STATE=""

start_active_gpu_memory_sampler() {
  local engine="$1" guard="$2" transaction_id="$3" transaction_sha256="$4"
  local container_id="$5" timeout="$6" attempt line_count summary max_samples
  ACTIVE_GPU_MEMORY_WRAPPER_START_TIME_TICKS="$(python3 - "$$" <<'PY'
import sys
from pathlib import Path
pid = int(sys.argv[1])
raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
closing = raw.rfind(")")
fields = raw[closing + 1:].split()
value = int(fields[19])
if closing <= 0 or len(fields) <= 19 or value <= 0:
    raise SystemExit("invalid wrapper process identity")
print(value)
PY
)" || return 1
  [[ "${ACTIVE_GPU_MEMORY_WRAPPER_START_TIME_TICKS}" =~ ^[1-9][0-9]*$ ]] \
    || return 1
  ACTIVE_GPU_MEMORY_EVIDENCE="$(dirname -- "${ACTIVE_TRANSACTION_MANIFEST}")/gpu-memory.jsonl"
  [[ ! -e "${ACTIVE_GPU_MEMORY_EVIDENCE}" && ! -L "${ACTIVE_GPU_MEMORY_EVIDENCE}" ]] \
    || return 1
  ACTIVE_GPU_MEMORY_CONTAINER_ID="${container_id}"
  ACTIVE_GPU_MEMORY_EVIDENCE_SHA256=""
  ACTIVE_GPU_MEMORY_STATE=""
  ACTIVE_GPU_MEMORY_PEAK_MIB=0
  ACTIVE_GPU_MEMORY_SAMPLE_COUNT=0
  max_samples=$((timeout * 1000 / GPU_MEMORY_SAMPLE_INTERVAL_MS + 1000))
  (
    umask 077
    exec python3 "${GPU_MEMORY_SAMPLER}" sample \
      --evidence "${ACTIVE_GPU_MEMORY_EVIDENCE}" \
      --device-index "${GPU_DEVICE_INDEX}" \
      --expected-uuid "${GPU_UUID}" \
      --engine "${engine}" \
      --transaction-id "${transaction_id}" \
      --prepared-transaction-sha256 "${transaction_sha256}" \
      --artifact-root-id "${ARTIFACT_ROOT_ID}" \
      --container-id "${container_id}" \
      --guard-mib "${guard}" \
      --interval-ms "${GPU_MEMORY_SAMPLE_INTERVAL_MS}" \
      --max-gap-ms "${GPU_MEMORY_MAX_GAP_MS}" \
      --parent-pid "$$" \
      --parent-start-time-ticks "${ACTIVE_GPU_MEMORY_WRAPPER_START_TIME_TICKS}" \
      --max-samples "${max_samples}"
  ) &
  ACTIVE_GPU_MEMORY_SAMPLER_PID="$!"
  ACTIVE_GPU_MEMORY_SAMPLER_START_TIME_TICKS="$(python3 - "${ACTIVE_GPU_MEMORY_SAMPLER_PID}" <<'PY'
import sys
import time
from pathlib import Path
pid = int(sys.argv[1])
for _ in range(100):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        closing = raw.rfind(")")
        fields = raw[closing + 1:].split()
        value = int(fields[19])
        if closing > 0 and len(fields) > 19 and value > 0:
            print(value)
            raise SystemExit(0)
    except (FileNotFoundError, ProcessLookupError, ValueError):
        pass
    time.sleep(0.01)
raise SystemExit("sampler process identity unavailable")
PY
)" || return 1
  for attempt in $(seq 1 200); do
    kill -0 "${ACTIVE_GPU_MEMORY_SAMPLER_PID}" 2>/dev/null || return 1
    if [[ -f "${ACTIVE_GPU_MEMORY_EVIDENCE}" ]]; then
      line_count="$(wc -l <"${ACTIVE_GPU_MEMORY_EVIDENCE}")"
    else
      line_count=0
    fi
    if [[ "${line_count}" =~ ^[0-9]+$ ]] && (( line_count >= 2 )); then
      if summary="$(python3 "${GPU_MEMORY_SAMPLER}" summarize \
          --allow-active \
          --evidence "${ACTIVE_GPU_MEMORY_EVIDENCE}" \
          --device-index "${GPU_DEVICE_INDEX}" \
          --expected-uuid "${GPU_UUID}" \
          --engine "${engine}" \
          --transaction-id "${transaction_id}" \
          --prepared-transaction-sha256 "${transaction_sha256}" \
          --artifact-root-id "${ARTIFACT_ROOT_ID}" \
          --container-id "${container_id}" \
          --guard-mib "${guard}" \
          --interval-ms "${GPU_MEMORY_SAMPLE_INTERVAL_MS}" \
          --max-gap-ms "${GPU_MEMORY_MAX_GAP_MS}" \
          --parent-pid "$$" \
          --parent-start-time-ticks "${ACTIVE_GPU_MEMORY_WRAPPER_START_TIME_TICKS}")"; then
        [[ "$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["state"])' <<<"${summary}")" == "active" ]] \
          && return 0
      fi
    fi
    sleep 0.025
  done
  return 1
}

collect_active_gpu_memory_sampler() {
  local require_clean="${1:-1}" status=0 signal_status=0 summary parsed
  [[ -n "${ACTIVE_GPU_MEMORY_SAMPLER_PID:-}" ]] || return 0
  if kill -0 "${ACTIVE_GPU_MEMORY_SAMPLER_PID}" 2>/dev/null; then
    python3 "${GPU_MEMORY_SAMPLER}" signal \
      --pid "${ACTIVE_GPU_MEMORY_SAMPLER_PID}" \
      --start-time-ticks "${ACTIVE_GPU_MEMORY_SAMPLER_START_TIME_TICKS}" \
      --signal TERM 2>/dev/null || signal_status=$?
  fi
  if wait "${ACTIVE_GPU_MEMORY_SAMPLER_PID}"; then
    status=0
  else
    status=$?
  fi
  summary="$(python3 "${GPU_MEMORY_SAMPLER}" summarize \
    --evidence "${ACTIVE_GPU_MEMORY_EVIDENCE}" \
    --device-index "${GPU_DEVICE_INDEX}" \
    --expected-uuid "${GPU_UUID}" \
    --engine "${ACTIVE_ENGINE}" \
    --transaction-id "${ACTIVE_CONTAINER_TRANSACTION_ID}" \
    --prepared-transaction-sha256 "${ACTIVE_TRANSACTION_SHA256}" \
    --artifact-root-id "${ACTIVE_CONTAINER_ARTIFACT_ROOT_ID}" \
    --container-id "${ACTIVE_GPU_MEMORY_CONTAINER_ID}" \
    --guard-mib "${ACTIVE_GPU_MEMORY_GUARD_MIB}" \
    --interval-ms "${GPU_MEMORY_SAMPLE_INTERVAL_MS}" \
    --max-gap-ms "${GPU_MEMORY_MAX_GAP_MS}" \
    --parent-pid "$$" \
    --parent-start-time-ticks "${ACTIVE_GPU_MEMORY_WRAPPER_START_TIME_TICKS}")" \
    || return 1
  parsed="$(python3 -c '
import json,sys
p=json.loads(sys.stdin.read())
print("\x1f".join(str(p.get(k) if p.get(k) is not None else "") for k in (
 "state","sample_count","maximum_observed_mib","first_sample_at_utc",
 "last_sample_at_utc","maximum_gap_ms","breach_at_utc","breach_mib",
 "evidence_sha256","guard_ok")))
' <<<"${summary}")" || return 1
  IFS=$'\x1f' read -r ACTIVE_GPU_MEMORY_STATE ACTIVE_GPU_MEMORY_SAMPLE_COUNT \
    ACTIVE_GPU_MEMORY_PEAK_MIB ACTIVE_GPU_MEMORY_FIRST_SAMPLE_UTC \
    ACTIVE_GPU_MEMORY_LAST_SAMPLE_UTC ACTIVE_GPU_MEMORY_MAXIMUM_GAP_MS \
    ACTIVE_GPU_MEMORY_BREACH_UTC ACTIVE_GPU_MEMORY_BREACH_MIB \
    ACTIVE_GPU_MEMORY_EVIDENCE_SHA256 guard_ok <<<"${parsed}"
  ACTIVE_GPU_MEMORY_SAMPLER_PID=""
  ACTIVE_GPU_MEMORY_SAMPLER_START_TIME_TICKS=""
  if (( signal_status != 0 )) && [[ "${ACTIVE_GPU_MEMORY_STATE}" == "active" ]]; then
    return 1
  fi
  if (( require_clean )) && { (( status != 0 )) \
      || [[ "${ACTIVE_GPU_MEMORY_STATE}" != "stopped" || "${guard_ok}" != "True" ]]; }; then
    return 1
  fi
  return 0
}

gpu_memory_sampler_alert() {
  collect_active_gpu_memory_sampler 0 || true
  local reason="${ACTIVE_ENGINE} NVML guard sampler failed closed"
  if [[ "${ACTIVE_GPU_MEMORY_STATE}" == "guard_breached" ]]; then
    reason="${ACTIVE_ENGINE} exceeded the ${ACTIVE_GPU_MEMORY_GUARD_MIB} MiB GPU-memory guard (${ACTIVE_GPU_MEMORY_BREACH_MIB} MiB maximum observed at ${ACTIVE_GPU_MEMORY_BREACH_UTC})"
  fi
  ACTIVE_TRANSACTION_REASON="${reason}"
  fail_active_container "${reason}" "" "" ""
}

cleanup_active_container() {
  local container_name="${ACTIVE_CONTAINER:-}"
  [[ -n "${container_name}" ]] || return 0
  if (( ! ACTIVE_LAUNCH_ATTEMPTED )); then
    ACTIVE_CONTAINER=""
    return 0
  fi

  local container_id="${ACTIVE_CONTAINER_ID:-}" residual ownership expected_ownership
  if [[ -z "${container_id}" ]]; then
    if ! residual="$(docker ps -aq --no-trunc --filter "name=^/${container_name}$" 2>&1)"; then
      echo "[CRITICAL] unable to resolve attempted maintenance launch: ${residual}" >&2
      return 96
    fi
    if [[ -z "${residual//[[:space:]]/}" ]]; then
      ACTIVE_CONTAINER=""
      ACTIVE_LAUNCH_ATTEMPTED=0
      return 0
    fi
    [[ "${residual}" != *$'\n'* ]] || {
      echo "[CRITICAL] multiple containers match attempted maintenance name: ${container_name}" >&2
      return 96
    }
    container_id="${residual//[[:space:]]/}"
  fi

  if ! ownership="$(
    docker inspect "${container_id}" --format \
      '{{index .Config.Labels "noesis.ds9.role"}}|{{index .Config.Labels "noesis.ds9.transaction"}}|{{index .Config.Labels "noesis.ds9.artifact-root-sha256"}}|{{.Name}}' 2>&1
  )"; then
    echo "[CRITICAL] unable to prove maintenance-container ownership: ${container_id}: ${ownership}" >&2
    return 96
  fi
  expected_ownership="engine-maintenance|${ACTIVE_CONTAINER_TRANSACTION_ID}|${ACTIVE_CONTAINER_ARTIFACT_ROOT_ID}|/${container_name}"
  if [[ "${ownership}" != "${expected_ownership}" ]]; then
    echo "[CRITICAL] refusing to remove container without exact transaction ownership: ${container_id}: ${ownership}" >&2
    return 96
  fi
  docker rm -f "${container_id}" >/dev/null 2>&1 || true
  if docker inspect "${container_id}" >/dev/null 2>&1; then
    echo "[CRITICAL] maintenance container still exists after forced removal: ${container_id}" >&2
    return 96
  fi
  if ! residual="$(docker ps -aq --no-trunc --filter "name=^/${container_name}$" 2>&1)"; then
    echo "[CRITICAL] unable to prove maintenance-container absence: ${residual}" >&2
    return 96
  fi
  if [[ -n "${residual//[[:space:]]/}" ]]; then
    echo "[CRITICAL] maintenance container remains listed after forced removal: ${container_name} (${residual})" >&2
    return 96
  fi
  if ! collect_active_gpu_memory_sampler 0; then
    echo "[CRITICAL] NVML sampler evidence could not be sealed after container removal" >&2
    return 95
  fi
  ACTIVE_CONTAINER=""
  ACTIVE_CONTAINER_ID=""
  ACTIVE_CONTAINER_TRANSACTION_ID=""
  ACTIVE_CONTAINER_ARTIFACT_ROOT_ID=""
  ACTIVE_LAUNCH_ATTEMPTED=0
  ACTIVE_ENGINE=""
  ACTIVE_CONTAINER_INIT_PID=""
  ACTIVE_CONTAINER_INIT_START_TIME_TICKS=""
}

persist_active_failure_evidence() {
  local reason="$1" owner_before="${2:-}" owner_after="${3:-}" ownership_proof="${4:-}"
  local engine="${ACTIVE_ENGINE:-unknown}" transaction_id="${ACTIVE_CONTAINER_TRANSACTION_ID:-unknown}"
  local evidence="${ARTIFACT_ROOT}/logs/${engine}_${transaction_id}.failure.log"
  [[ ! -e "${evidence}" && ! -L "${evidence}" ]] || {
    echo "[CRITICAL] refusing to replace maintenance failure evidence: ${evidence}" >&2
    return 98
  }
  (
    umask 077
    set -o noclobber
    {
      printf 'recorded_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'engine=%s\n' "${engine}"
      printf 'container_id=%s\n' "${ACTIVE_CONTAINER_ID:-unknown}"
      printf 'container_name=%s\n' "${ACTIVE_CONTAINER:-unknown}"
      printf 'container_init_pid=%s\n' "${ACTIVE_CONTAINER_INIT_PID:-unknown}"
      printf 'container_init_start_time_ticks=%s\n' "${ACTIVE_CONTAINER_INIT_START_TIME_TICKS:-unknown}"
      printf 'transaction_id=%s\n' "${transaction_id}"
      printf 'reason=%s\n' "${reason}"
      printf 'gpu_memory_total_mib=%s\n' "${GPU_MEMORY_MIB:-unknown}"
      printf 'gpu_memory_guard_mib=%s\n' "${ACTIVE_GPU_MEMORY_GUARD_MIB:-unknown}"
      printf 'gpu_memory_peak_mib=%s\n' "${ACTIVE_GPU_MEMORY_PEAK_MIB:-0}"
      printf 'gpu_memory_sample_count=%s\n' "${ACTIVE_GPU_MEMORY_SAMPLE_COUNT:-0}"
      printf 'gpu_memory_sampler_state=%s\n' "${ACTIVE_GPU_MEMORY_STATE:-unknown}"
      printf 'gpu_memory_evidence=%s\n' "${ACTIVE_GPU_MEMORY_EVIDENCE:-unknown}"
      printf 'gpu_memory_evidence_sha256=%s\n' "${ACTIVE_GPU_MEMORY_EVIDENCE_SHA256:-unknown}"
      printf 'gpu_memory_first_sample_utc=%s\n' "${ACTIVE_GPU_MEMORY_FIRST_SAMPLE_UTC:-unknown}"
      printf 'gpu_memory_last_sample_utc=%s\n' "${ACTIVE_GPU_MEMORY_LAST_SAMPLE_UTC:-unknown}"
      printf 'gpu_memory_maximum_gap_ms=%s\n' "${ACTIVE_GPU_MEMORY_MAXIMUM_GAP_MS:-unknown}"
      printf 'gpu_memory_breach_utc=%s\n' "${ACTIVE_GPU_MEMORY_BREACH_UTC:-}"
      printf 'gpu_owner_snapshot_before=%s\n' "${owner_before//$'\n'/,}"
      printf 'gpu_owner_snapshot_after=%s\n' "${owner_after//$'\n'/,}"
      printf 'ownership_proof=%s\n' "${ownership_proof//$'\n'/ }"
      echo '[container_inspect]'
      docker inspect "${ACTIVE_CONTAINER_ID}" 2>&1 || true
      echo '[container_logs]'
      docker logs --tail 320 "${ACTIVE_CONTAINER_ID}" 2>&1 || true
    } >"${evidence}"
  ) || {
    echo "[CRITICAL] failed to persist maintenance failure evidence: ${evidence}" >&2
    return 98
  }
  chmod 0600 -- "${evidence}"
  sync -f -- "${evidence}"
  echo "[FAILURE-EVIDENCE] ${evidence}" >&2
}

fail_active_container() {
  local reason="$1" owner_before="${2:-}" owner_after="${3:-}" ownership_proof="${4:-}"
  ACTIVE_TRANSACTION_REASON="${reason}"
  persist_active_failure_evidence \
    "${reason}" "${owner_before}" "${owner_after}" "${ownership_proof}" || exit 98
  fail "${reason}"
}

query_gpu_owner_pids() {
  local raw line normalized=""
  if ! raw="$(
    nvidia-smi --id="${GPU_DEVICE_INDEX}" \
      --query-compute-apps=pid --format=csv,noheader,nounits 2>&1
  )"; then
    printf '%s\n' "${raw}" >&2
    return 1
  fi
  while IFS= read -r line; do
    line="${line//[[:space:]]/}"
    [[ -z "${line}" ]] && continue
    [[ "${line}" =~ ^[1-9][0-9]*$ ]] || {
      printf 'invalid GPU owner PID row: %s\n' "${line}" >&2
      return 1
    }
    normalized+="${line}"$'\n'
  done <<<"${raw}"
  if [[ -n "${normalized}" ]]; then
    sort -nu <<<"${normalized}"
  fi
}

assert_gpu_owners_belong_to_container() {
  local engine="$1" container_init_pid="$2" init_start_time_ticks="$3" attempt before after final proof proof_final status status_final owner_pid container_state
  local -a proof_args
  for attempt in 1 2 3 4; do
    if ! before="$(query_gpu_owner_pids)"; then
      fail_active_container \
        "unable to query GPU compute owners during ${engine}" "" "" "query failed"
    fi
    proof_args=(
      "${GPU_OWNERSHIP_HELPER}"
      --container-init-pid "${container_init_pid}"
      --container-init-start-time-ticks "${init_start_time_ticks}"
    )
    while IFS= read -r owner_pid; do
      [[ -n "${owner_pid}" ]] && proof_args+=(--owner-pid "${owner_pid}")
    done <<<"${before}"
    if proof="$(python3 "${proof_args[@]}" 2>&1)"; then
      status=0
    else
      status=$?
    fi
    if (( status == 2 )); then
      fail_active_container \
        "GPU process ownership proof rejected during ${engine}: ${proof}" \
        "${before}" "" "${proof}"
    elif (( status != 0 && status != 4 )); then
      fail_active_container \
        "GPU owner ancestry proof failed during ${engine}: ${proof}" \
        "${before}" "" "${proof}"
    fi
    if ! after="$(query_gpu_owner_pids)"; then
      fail_active_container \
        "unable to requery GPU compute owners during ${engine}" \
        "${before}" "" "${proof}"
    fi
    if (( status == 4 )) || [[ "${before}" != "${after}" ]]; then
      sleep 0.1
      continue
    fi

    # Re-prove a stable PID set.  This binds a same-number PID replacement to
    # its new start time and ancestry before the sample is accepted.
    proof_args=(
      "${GPU_OWNERSHIP_HELPER}"
      --container-init-pid "${container_init_pid}"
      --container-init-start-time-ticks "${init_start_time_ticks}"
    )
    while IFS= read -r owner_pid; do
      [[ -n "${owner_pid}" ]] && proof_args+=(--owner-pid "${owner_pid}")
    done <<<"${after}"
    if proof_final="$(python3 "${proof_args[@]}" 2>&1)"; then
      status_final=0
    else
      status_final=$?
    fi
    if (( status_final == 2 )); then
      fail_active_container \
        "GPU process ownership proof rejected during ${engine}: ${proof_final}" \
        "${before}" "${after}" "${proof_final}"
    elif (( status_final == 4 )); then
      sleep 0.1
      continue
    elif (( status_final != 0 )); then
      fail_active_container \
        "GPU owner ancestry proof failed during ${engine}: ${proof_final}" \
        "${before}" "${after}" "${proof_final}"
    fi
    if ! final="$(query_gpu_owner_pids)"; then
      fail_active_container \
        "unable to complete GPU owner stability query during ${engine}" \
        "${before}" "${after}" "${proof_final}"
    fi
    if [[ "${after}" != "${final}" ]]; then
      sleep 0.1
      continue
    fi
    return 0
  done
  if ! container_state="$(
    docker inspect "${ACTIVE_CONTAINER_ID}" --format '{{.State.Running}}|{{.State.Pid}}' 2>&1
  )"; then
    fail_active_container \
      "unable to resolve container state after GPU owner churn: ${container_state}" \
      "${before:-}" "${after:-}" "${proof_final:-${proof:-}}"
  fi
  if [[ "${container_state}" == false\|* ]]; then
    return 0
  fi
  fail_active_container \
    "GPU owner set remained unstable during ${engine}; refusing ambiguous ownership (container=${container_state})" \
    "${before:-}" "${after:-}" "${proof_final:-${proof:-}}"
}

rollback_active_transaction() {
  local transaction_manifest="${ACTIVE_TRANSACTION_MANIFEST:-}"
  [[ -n "${transaction_manifest}" ]] || return 0
  local rollback_output
  if ! rollback_output="$(
    python3 "${FINALIZER}" rollback \
      --transaction-manifest "${transaction_manifest}" \
      --expected-transaction-sha256 "${ACTIVE_TRANSACTION_SHA256}" \
      --reason "${ACTIVE_TRANSACTION_REASON}" \
      --lock-fd 9
  )"; then
    echo "[CRITICAL] automatic host engine/realization rollback failed: ${transaction_manifest}" >&2
    return 97
  fi
  echo "[ROLLBACK] ${rollback_output}" >&2
  ACTIVE_TRANSACTION_MANIFEST=""
  ACTIVE_TRANSACTION_SHA256=""
}

cleanup_active_work() {
  local original_status="$?"
  trap - EXIT INT TERM
  if ! cleanup_active_container; then
    echo "[CRITICAL] host rollback skipped because builder containment is unproven; prepared transaction=${ACTIVE_TRANSACTION_MANIFEST:-unknown}" >&2
    exit 96
  fi
  if ! rollback_active_transaction; then
    exit 97
  fi
  exit "${original_status}"
}

terminate_active_work() {
  local code="$1"
  ACTIVE_TRANSACTION_REASON="wrapper received termination signal ${code} before commit"
  exit "${code}"
}

require_absolute_root() {
  local variable="$1" raw="${!1:-}" resolved
  [[ -n "${raw}" ]] || fail "${variable} must be set explicitly"
  [[ "${raw}" != *$'\n'* && "${raw}" != *$'\r'* && "${raw}" != *$'\t'* ]] \
    || fail "${variable} must not contain control characters"
  [[ "${raw}" == /* ]] || fail "${variable} must be absolute: ${raw}"
  resolved="$(realpath -m -- "${raw}")"
  [[ "${resolved}" != "/" && "${resolved}" != "${REPO_ROOT}" && "${resolved}" != "${DS9_ROOT}" ]] \
    || fail "refusing unsafe ${variable}: ${resolved}"
  [[ "${resolved}" != "${REPO_ROOT}/"* ]] \
    || fail "${variable} must not be inside the checkout: ${resolved}"
  printf '%s\n' "${resolved}"
}

print_command() {
  printf '[RUN]'
  printf ' %q' "$@"
  printf '\n'
}

MODE=""
ONLY=""
PLAN=0
select_mode() {
  local requested="$1"
  [[ -z "${MODE}" ]] || fail "engine selection modes are mutually exclusive: already selected --${MODE}, then received --${requested}"
  MODE="${requested}"
}
while (($#)); do
  case "$1" in
    --only)
      (($# >= 2)) || fail "--only requires an engine name"
      select_mode "only"
      ONLY="$2"
      shift 2
      ;;
    --all)
      select_mode "all"
      shift
      ;;
    --v3dt)
      select_mode "v3dt"
      shift
      ;;
    --plan)
      PLAN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

[[ -n "${MODE}" ]] || { usage >&2; exit 2; }
[[ "${IMAGE}" == "${REQUIRED_IMAGE_REF}" ]] \
  || fail "NOESIS_DS9_DEV_IMAGE must remain the reviewed reference ${REQUIRED_IMAGE_REF}"
CALLER_UID="$(id -u)"
CALLER_GID="$(id -g)"
[[ "${CALLER_UID}" =~ ^[0-9]+$ && "${CALLER_UID}" != "0" && "${CALLER_GID}" =~ ^[0-9]+$ ]] \
  || fail "DS9 engine maintenance must run as a non-root caller"
if [[ "${MODE}" == "only" ]]; then
  [[ -n "${OUTPUTS[${ONLY}]:-}" ]] || fail "unsupported canonical engine: ${ONLY}"
  ENGINES=("${ONLY}")
elif [[ "${MODE}" == "v3dt" ]]; then
  ENGINES=(yolo26_seg_s bodypose3dnet v3dt_tracker_reid)
else
  ENGINES=("${CANONICAL_ORDER[@]}")
fi

DOCKER_ROOT="$(require_absolute_root NOESIS_DS9_DOCKER_ROOT)"
ARTIFACT_ROOT="$(require_absolute_root NOESIS_DS9_ARTIFACT_ROOT)"
if [[ "${DOCKER_ROOT}" == "${ARTIFACT_ROOT}" \
      || "${DOCKER_ROOT}" == "${ARTIFACT_ROOT}/"* \
      || "${ARTIFACT_ROOT}" == "${DOCKER_ROOT}/"* ]]; then
  fail "NOESIS_DS9_DOCKER_ROOT and NOESIS_DS9_ARTIFACT_ROOT must be disjoint"
fi
ARTIFACT_MODELS="${ARTIFACT_ROOT}/models"
ARTIFACT_ROOT_ID="$(printf '%s' "${ARTIFACT_ROOT}" | sha256sum | awk '{print $1}')"
[[ "${ARTIFACT_ROOT_ID}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "unable to derive the artifact-root container identity"
for path in "${ARTIFACT_MODELS}" "${ARTIFACT_MODELS}/onnx" "${ARTIFACT_MODELS}/engines" "${ARTIFACT_MODELS}/engine_maintenance" "${ARTIFACT_ROOT}/logs"; do
  [[ ! -L "${path}" ]] || fail "artifact path must not be a symlink: ${path}"
done
[[ -d "${ARTIFACT_MODELS}/onnx" ]] || fail "staged ONNX directory is unavailable: ${ARTIFACT_MODELS}/onnx"
if (( PLAN )); then
  for path in "${ARTIFACT_MODELS}/engines" "${ARTIFACT_MODELS}/engine_maintenance" "${ARTIFACT_ROOT}/logs"; do
    [[ -d "${path}" ]] \
      || fail "plan mode requires the pre-existing artifact directory: ${path}"
  done
else
  mkdir -p -- "${ARTIFACT_MODELS}/engines" "${ARTIFACT_MODELS}/engine_maintenance" "${ARTIFACT_ROOT}/logs"
  chmod 0700 -- "${ARTIFACT_MODELS}/engine_maintenance" "${ARTIFACT_ROOT}/logs"
fi
for path in "${ARTIFACT_MODELS}/engines" "${ARTIFACT_MODELS}/engine_maintenance" "${ARTIFACT_ROOT}/logs"; do
  [[ "$(stat -c '%u' -- "${path}")" == "${CALLER_UID}" ]] \
    || fail "artifact directory must be owned by the caller: ${path}"
done
for path in "${ARTIFACT_MODELS}/engine_maintenance" "${ARTIFACT_ROOT}/logs"; do
  [[ "$(stat -c '%a' -- "${path}")" == "700" ]] \
    || fail "artifact evidence directory must already be owner-private mode 0700: ${path}"
done
INVOCATION_LOCK="${ARTIFACT_ROOT}/.noesis-ds9-artifact-transaction.lock"
if (( ! PLAN )) && [[ ! -e "${INVOCATION_LOCK}" && ! -L "${INVOCATION_LOCK}" ]]; then
  (umask 077; set -o noclobber; : >"${INVOCATION_LOCK}") \
    || fail "unable to create artifact transaction lock safely: ${INVOCATION_LOCK}"
fi
[[ ! -L "${INVOCATION_LOCK}" && -f "${INVOCATION_LOCK}" ]] \
  || fail "artifact transaction lock must be a regular non-symlink: ${INVOCATION_LOCK}"
[[ "$(stat -c '%u:%h' -- "${INVOCATION_LOCK}")" == "$(id -u):1" ]] \
  || fail "artifact transaction lock must be owned by the caller with one link"
if (( PLAN )); then
  [[ "$(stat -c '%a' -- "${INVOCATION_LOCK}")" == "600" ]] \
    || fail "plan mode requires the artifact transaction lock to already be mode 0600"
  exec 9<"${INVOCATION_LOCK}"
else
  chmod 0600 -- "${INVOCATION_LOCK}"
  exec 9>>"${INVOCATION_LOCK}"
fi
[[ ! -L "${INVOCATION_LOCK}" ]] \
  || fail "artifact transaction lock changed to a symlink during open"
[[ "$(stat -Lc '%d:%i' /proc/$$/fd/9)" == "$(stat -Lc '%d:%i' "${INVOCATION_LOCK}")" ]] \
  || fail "artifact transaction lock changed during secure open"
[[ "$(stat -Lc '%u:%h:%a' /proc/$$/fd/9)" == "$(id -u):1:600" ]] \
  || fail "artifact transaction lock descriptor has unsafe ownership/mode"
flock -n 9 || fail "another host DS9 engine-maintenance invocation owns ${INVOCATION_LOCK}"

trim_whitespace() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s\n' "${value}"
}

GPU_PROFILE_ROW=""
if ! GPU_PROFILE_ROW="$(
  nvidia-smi --id="${GPU_DEVICE_INDEX}" \
    --query-gpu=index,name,uuid,compute_cap,memory.total,driver_version \
    --format=csv,noheader,nounits 2>&1
)"; then
  fail "unable to query the exact GPU ${GPU_DEVICE_INDEX} maintenance profile: ${GPU_PROFILE_ROW}"
fi
[[ -n "${GPU_PROFILE_ROW}" && "${GPU_PROFILE_ROW}" != *$'\n'* ]] \
  || fail "GPU ${GPU_DEVICE_INDEX} query returned an unexpected row count"
IFS=',' read -r GPU_INDEX GPU_NAME GPU_UUID GPU_COMPUTE_CAPABILITY GPU_MEMORY_MIB HOST_DRIVER_VERSION <<<"${GPU_PROFILE_ROW}"
GPU_INDEX="$(trim_whitespace "${GPU_INDEX}")"
GPU_NAME="$(trim_whitespace "${GPU_NAME}")"
GPU_UUID="$(trim_whitespace "${GPU_UUID}")"
GPU_COMPUTE_CAPABILITY="$(trim_whitespace "${GPU_COMPUTE_CAPABILITY}")"
GPU_MEMORY_MIB="$(trim_whitespace "${GPU_MEMORY_MIB}")"
HOST_DRIVER_VERSION="$(trim_whitespace "${HOST_DRIVER_VERSION}")"
[[ "${GPU_INDEX}" == "${GPU_DEVICE_INDEX}" \
    && -n "${GPU_NAME}" \
    && -n "${GPU_UUID}" \
    && "${GPU_COMPUTE_CAPABILITY}" =~ ^[0-9]+\.[0-9]+$ \
    && "${GPU_MEMORY_MIB}" =~ ^[0-9]+$ ]] \
  || fail "unable to capture the exact GPU ${GPU_DEVICE_INDEX} maintenance profile"
HOST_DRIVER_MAJOR="${HOST_DRIVER_VERSION%%.*}"
[[ "${HOST_DRIVER_MAJOR}" =~ ^[0-9]+$ ]] \
  || fail "unable to determine the host NVIDIA driver version"
if (( HOST_DRIVER_MAJOR < REQUIRED_DRIVER_MAJOR )); then
  if (( PLAN )); then
    echo "[WARN] host NVIDIA driver ${HOST_DRIVER_VERSION} is below the installed DeepStream 9 requirement (${REQUIRED_DRIVER_MAJOR}+); no-GPU plan only"
  else
    fail "host NVIDIA driver ${HOST_DRIVER_VERSION} is below the installed DeepStream 9 requirement (${REQUIRED_DRIVER_MAJOR}+); upgrade the driver before any DS9 engine build or cutover"
  fi
fi

NOESIS_DS9_DOCKER_ROOT="${DOCKER_ROOT}" "${SCRIPT_DIR}/secondary_docker.sh" status >/dev/null
export DOCKER_HOST="unix://${DOCKER_ROOT}/run/docker.sock"
[[ "$(docker info --format '{{.DefaultRuntime}}')" == "runc" ]] \
  || fail "secondary Docker default runtime must remain runc"

list_maintenance_containers() {
  docker ps -aq --no-trunc \
    --filter label=noesis.ds9.role=engine-maintenance
}

assert_no_maintenance_containers() {
  local phase="$1" observed
  if ! observed="$(list_maintenance_containers 2>&1)"; then
    fail "unable to audit maintenance containers ${phase}: ${observed}"
  fi
  [[ -z "${observed//[[:space:]]/}" ]] \
    || fail "foreign/orphan maintenance container appeared ${phase}: ${observed}"
}

assert_container_name_available() {
  local container_name="$1" phase="$2" observed
  if ! observed="$(
    docker ps -aq --no-trunc --filter "name=^/${container_name}$" 2>&1
  )"; then
    fail "unable to audit maintenance-container name ${phase}: ${observed}"
  fi
  [[ -z "${observed//[[:space:]]/}" ]] \
    || fail "maintenance-container name is already owned ${phase}: ${container_name}: ${observed}"
}

if ! STALE_CONTAINERS="$(list_maintenance_containers 2>&1)"; then
  fail "unable to audit secondary-daemon maintenance containers: ${STALE_CONTAINERS}"
fi
if [[ -n "${STALE_CONTAINERS//[[:space:]]/}" ]]; then
  SAME_ROOT_CONTAINERS=()
  FOREIGN_ROOT_CONTAINERS=()
  while IFS= read -r stale_container; do
    stale_container="${stale_container//[[:space:]]/}"
    [[ -z "${stale_container}" ]] && continue
    if ! stale_root_id="$(
      docker inspect "${stale_container}" \
        --format '{{index .Config.Labels "noesis.ds9.artifact-root-sha256"}}' 2>&1
    )"; then
      fail "unable to inspect maintenance-container artifact identity: ${stale_container}: ${stale_root_id}"
    fi
    if [[ "${stale_root_id}" == "${ARTIFACT_ROOT_ID}" ]]; then
      SAME_ROOT_CONTAINERS+=("${stale_container}")
    else
      FOREIGN_ROOT_CONTAINERS+=("${stale_container}:${stale_root_id:-missing-label}")
    fi
  done <<<"${STALE_CONTAINERS}"
  ((${#FOREIGN_ROOT_CONTAINERS[@]} == 0)) \
    || fail "foreign-root maintenance container(s) are present and will not be removed: ${FOREIGN_ROOT_CONTAINERS[*]}"
  if (( PLAN )); then
    fail "plan mode is read-only and found same-root orphan maintenance container(s): ${SAME_ROOT_CONTAINERS[*]}"
  fi
  for stale_container in "${SAME_ROOT_CONTAINERS[@]}"; do
    docker rm -f "${stale_container}" >/dev/null \
      || fail "failed to remove stale maintenance container: ${stale_container}"
  done
  if ! STALE_CONTAINERS="$(list_maintenance_containers 2>&1)"; then
    fail "unable to prove stale maintenance-container cleanup: ${STALE_CONTAINERS}"
  fi
  [[ -z "${STALE_CONTAINERS//[[:space:]]/}" ]] \
    || fail "stale maintenance containers remain after cleanup: ${STALE_CONTAINERS}"
fi

RECOVERY_ARGS=(
  recover
  --artifact-root "${ARTIFACT_ROOT}"
  --lock-fd 9
  --reason "engine-maintenance startup recovery"
)
if (( ! PLAN )); then
  if ! GPU_OWNERS="$(
    nvidia-smi --id="${GPU_DEVICE_INDEX}" \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader 2>&1
  )"; then
    fail "unable to query existing GPU ${GPU_DEVICE_INDEX} compute owners: ${GPU_OWNERS}"
  fi
  [[ -z "${GPU_OWNERS//[[:space:]]/}" ]] || fail $'GPU compute owner(s) are still active; stop them before transaction recovery:\n'"${GPU_OWNERS}"
  RECOVERY_ARGS+=(--apply)
fi
if ! RECOVERY_JSON="$(python3 "${FINALIZER}" "${RECOVERY_ARGS[@]}")"; then
  fail "DS9 engine transaction recovery/audit failed: ${RECOVERY_JSON}"
fi
echo "[RECOVERY] ${RECOVERY_JSON}"

MIN_FREE_HEADROOM_BYTES="${NOESIS_DS9_MIN_FREE_HEADROOM_BYTES:-10737418240}"
[[ "${MIN_FREE_HEADROOM_BYTES}" =~ ^[0-9]+$ ]] \
  || fail "NOESIS_DS9_MIN_FREE_HEADROOM_BYTES must be a non-negative integer"

require_capacity() {
  local engine="$1" available required prior_bytes=0 output
  output="${ARTIFACT_MODELS}/engines/${OUTPUTS[${engine}]}"
  [[ ! -L "${output}" ]] || fail "engine target must not be a symlink: ${output}"
  if [[ -f "${output}" ]]; then
    prior_bytes="$(stat -c '%s' -- "${output}")"
    [[ "${prior_bytes}" =~ ^[0-9]+$ ]] || fail "unable to measure prior engine size: ${output}"
  fi
  available="$(df --output=avail -B1 -- "${ARTIFACT_ROOT}" | awk 'NR == 2 {print $1}')"
  [[ "${available}" =~ ^[0-9]+$ ]] || fail "unable to measure artifact-root free space"
  # The host finalizer and in-container builder each preserve an independent
  # prior-engine copy before candidate publication.
  required=$((MIN_FREE_HEADROOM_BYTES + TEMP_BUDGET_BYTES[${engine}] + (2 * prior_bytes)))
  (( available >= required )) || fail \
    "artifact root has insufficient free space for ${engine}: available=${available} required=${required} (includes ${MIN_FREE_HEADROOM_BYTES} bytes residual headroom and two ${prior_bytes}-byte prior-engine snapshots)"
}

ENGINE_CSV="$(IFS=,; echo "${ENGINES[*]}")"
VERIFY_PROFILE="canonical"
SOURCE_VERIFY_ARGS=(--verify-only --only "${ENGINE_CSV}")
if [[ "${MODE}" == "all" ]]; then
  SOURCE_VERIFY_ARGS=(--verify-only --profile canonical)
elif [[ "${MODE}" == "v3dt" ]]; then
  VERIFY_PROFILE="v3dt"
  SOURCE_VERIFY_ARGS=(--verify-only --profile v3dt)
fi
python3 "${SCRIPT_DIR}/validate_asset_manifest.py" --profile "${VERIFY_PROFILE}"
NOESIS_DS9_ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
  python3 "${SCRIPT_DIR}/stage_canonical_sources.py" "${SOURCE_VERIFY_ARGS[@]}"
if (( ! PLAN )) && [[ "${MODE}" == "v3dt" ]]; then
  # The v3dt profile inherits runtime_common. Refuse the GPU window before its
  # three V3DT-specific builds unless that shared base is already authoritative.
  python3 "${SCRIPT_DIR}/validate_asset_manifest.py" \
    --artifact-root "${ARTIFACT_ROOT}" \
    --check-files \
    --profile runtime_common \
    --require-provenance \
    --require-realization
fi

IMAGE_ID="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
[[ "${IMAGE_ID}" == "${REQUIRED_IMAGE_ID}" ]] \
  || fail "derived image ID drifted: required=${REQUIRED_IMAGE_ID} observed=${IMAGE_ID}"
# Resolve the reviewed tag exactly once, then inspect and launch only by the
# immutable image ID. A concurrent retag cannot change metadata or execution.
BASE_DIGEST="$(docker image inspect "${IMAGE_ID}" --format '{{index .Config.Labels "org.opencontainers.image.base.digest"}}')"
[[ "${BASE_DIGEST}" == "sha256:2e45070ad134b9ab2caa4a97ba4d52fa8744a4f0db30900bd92828d51425a69a" ]] \
  || fail "derived image has an unexpected DS9 base digest: ${BASE_DIGEST}"
IMAGE_TRT_VERSION="$(docker image inspect "${IMAGE_ID}" --format '{{index .Config.Labels "com.nvidia.tensorrt.version"}}')"
IMAGE_CUDA_VERSION="$(docker image inspect "${IMAGE_ID}" --format '{{range .Config.Env}}{{println .}}{{end}}' | awk -F= '$1 == "CUDA_VERSION" {print $2; exit}')"
[[ "${IMAGE_TRT_VERSION}" == "10.14.1.48+cuda13.0" ]] \
  || fail "derived image has an unexpected TensorRT version: ${IMAGE_TRT_VERSION}"
[[ "${IMAGE_CUDA_VERSION}" == "13.1.1.006" ]] \
  || fail "derived image has an unexpected CUDA version: ${IMAGE_CUDA_VERSION}"

run_engine() {
  local engine="$1" output container timeout minimum_timeout guard started now running status log_path
  local snapshot_json transaction_manifest transaction_sha256 transaction_id transaction_relative transaction_container
  output="${ARTIFACT_MODELS}/engines/${OUTPUTS[${engine}]}"
  ACTIVE_GPU_MEMORY_GUARD_MIB=""
  ACTIVE_GPU_MEMORY_PEAK_MIB=0
  ACTIVE_GPU_MEMORY_SAMPLE_COUNT=0
  require_capacity "${engine}"
  container="noesis-ds9-${engine//_/-}-$$"
  # rebuild_engines permits 30s probe + 1800s build + two 120s loads. The
  # outer watchdog must leave additional time for source checks and cleanup;
  # it must never interrupt the builder after atomic install but before its
  # final load/rollback evidence.
  timeout="${NOESIS_DS9_TRT_TIMEOUT_SECONDS:-2400}"
  minimum_timeout=2400
  guard="${NOESIS_DS9_GPU_GUARD_MB:-11000}"
  if [[ "${engine}" == "reid_swin" ]]; then
    timeout="${NOESIS_REID_SWIN_TRT_TIMEOUT_SECONDS:-2400}"
    guard="${NOESIS_REID_SWIN_GPU_GUARD_MB:-11000}"
  elif [[ "${engine}" == "mapanything" ]]; then
    timeout="${NOESIS_MAPANYTHING_TRT_TIMEOUT_SECONDS:-2400}"
    guard="${NOESIS_MAPANYTHING_GPU_GUARD_MB:-9000}"
  elif [[ "${engine}" == "wholebody49_s_masks" ]]; then
    timeout="${NOESIS_WHOLEBODY49_S_TRT_TIMEOUT_SECONDS:-2400}"
    guard="${NOESIS_WHOLEBODY49_S_GPU_GUARD_MB:-10000}"
  elif [[ "${engine}" == "wholebody49_x_boxes" ]]; then
    timeout="${NOESIS_WHOLEBODY49_X_TRT_TIMEOUT_SECONDS:-2400}"
    guard="${NOESIS_WHOLEBODY49_X_GPU_GUARD_MB:-11000}"
  elif [[ "${engine}" == "bodypose3dnet" ]]; then
    timeout="${NOESIS_BODYPOSE3DNET_TRT_TIMEOUT_SECONDS:-2400}"
    guard="${NOESIS_BODYPOSE3DNET_GPU_GUARD_MB:-9000}"
  elif [[ "${engine}" == "v3dt_tracker_reid" ]]; then
    # Tracker helper upper bounds total 1290s before non-command setup and
    # cleanup. Preserve a 510s host margin.
    timeout="${NOESIS_V3DT_TRACKER_REID_TRT_TIMEOUT_SECONDS:-1800}"
    minimum_timeout=1800
    guard="${NOESIS_V3DT_TRACKER_REID_GPU_GUARD_MB:-9000}"
  fi
  [[ "${timeout}" =~ ^[1-9][0-9]*$ ]] \
    || fail "${engine} build timeout must be a positive integer: ${timeout}"
  [[ "${guard}" =~ ^[1-9][0-9]*$ ]] \
    || fail "${engine} GPU-memory guard must be a positive integer: ${guard}"
  (( timeout >= minimum_timeout )) \
    || fail "${engine} host watchdog must be at least ${minimum_timeout}s to exceed all internal builder/load bounds: ${timeout}"
  ACTIVE_GPU_MEMORY_GUARD_MIB="${guard}"

  local rebuild_args
  if [[ "${engine}" == "v3dt_tracker_reid" ]]; then
    rebuild_args=(DS9/scripts/build_v3dt_tracker_engine.py --validate-load --evidence-root /workspace/DS9/models/engine_maintenance)
  else
    rebuild_args=(DS9/scripts/rebuild_engines.py --only "${engine}" --validate-load --evidence-root /workspace/DS9/models/engine_maintenance)
    [[ "${engine}" == "mapanything" ]] && rebuild_args+=(--include-mapanything)
  fi
  (( PLAN )) && rebuild_args+=(--dry-run)

  local common_args=(
    run
    --name "${container}"
    --network=none
    --read-only
    --cap-drop=ALL
    --security-opt=no-new-privileges
    --user "${CALLER_UID}:${CALLER_GID}"
    --init
    --memory "${MAINTENANCE_MEMORY_BYTES}"
    --memory-swap "${MAINTENANCE_MEMORY_BYTES}"
    --pids-limit "${MAINTENANCE_PIDS_LIMIT}"
    --log-driver local
    --log-opt "max-size=${MAINTENANCE_LOG_MAX_SIZE}"
    --log-opt "max-file=${MAINTENANCE_LOG_MAX_FILES}"
    --env HOME=/tmp/noesis-home
    --env PYTHONDONTWRITEBYTECODE=1
    --env CUDA_CACHE_PATH=/tmp/cuda-cache
    --env NOESIS_MODEL_DIR=/workspace/DS9/models
    --env "NOESIS_DS9_MAINT_IMAGE=${IMAGE}"
    --env "NOESIS_DS9_MAINT_IMAGE_ID=${IMAGE_ID}"
    --env "NOESIS_DS9_MAINT_BASE_DIGEST=${BASE_DIGEST}"
    --env "NOESIS_DS9_MAINT_TRT_VERSION=${IMAGE_TRT_VERSION}"
    --env "NOESIS_DS9_MAINT_CUDA_VERSION=${IMAGE_CUDA_VERSION}"
    --env "NOESIS_DS9_MAINT_DRIVER_VERSION=${HOST_DRIVER_VERSION}"
    --env "NOESIS_DS9_MAINT_GPU_NAME=${GPU_NAME}"
    --env "NOESIS_DS9_MAINT_GPU_UUID=${GPU_UUID}"
    --env "NOESIS_DS9_MAINT_GPU_COMPUTE_CAPABILITY=${GPU_COMPUTE_CAPABILITY}"
    --env "NOESIS_DS9_MAINT_GPU_MEMORY_MIB=${GPU_MEMORY_MIB}"
    --tmpfs /tmp:rw,exec,nosuid,nodev,size=2147483648
    --mount "type=bind,src=${REPO_ROOT},dst=/workspace,readonly"
    --mount "type=bind,src=${ARTIFACT_MODELS},dst=/workspace/DS9/models,readonly"
    --mount "type=bind,src=${ARTIFACT_MODELS}/engines,dst=/workspace/DS9/models/engines"
    --mount "type=bind,src=${ARTIFACT_MODELS}/engine_maintenance,dst=/workspace/DS9/models/engine_maintenance"
    --workdir /workspace
  )

  if (( PLAN )); then
    local plan_cmd=(docker "${common_args[@]}" --rm --runtime=runc --env NVIDIA_VISIBLE_DEVICES=void --entrypoint python3 "${IMAGE_ID}" "${rebuild_args[@]}")
    print_command "${plan_cmd[@]}"
    "${plan_cmd[@]}"
    return
  fi

  # The default daemon runtime remains runc; --gpus is added only after the
  # fail-closed host owner check above succeeds.
  assert_container_name_available "${container}" "before ${engine} snapshot"
  assert_no_maintenance_containers "before ${engine} transaction snapshot"
  if ! GPU_OWNERS="$(
    nvidia-smi --id="${GPU_DEVICE_INDEX}" \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader 2>&1
  )"; then
    fail "unable to query GPU ${GPU_DEVICE_INDEX} compute owners before ${engine}: ${GPU_OWNERS}"
  fi
  [[ -z "${GPU_OWNERS//[[:space:]]/}" ]] || fail $'GPU compute owner(s) appeared before the next engine; aborting:\n'"${GPU_OWNERS}"

  snapshot_json="$(
    python3 "${FINALIZER}" snapshot \
      --engine "${engine}" \
      --artifact-root "${ARTIFACT_ROOT}" \
      --base-manifest "${DS9_ROOT}/asset_manifest.yaml" \
      --lock-fd 9
  )"
  transaction_manifest="$(
    python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["transaction_manifest"])' \
      <<<"${snapshot_json}"
  )"
  transaction_sha256="$(
    python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["transaction_sha256"])' \
      <<<"${snapshot_json}"
  )"
  transaction_id="$(
    python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["transaction_id"])' \
      <<<"${snapshot_json}"
  )"
  [[ "${transaction_manifest}" == "${ARTIFACT_MODELS}/engine_finalize/"* \
      && -f "${transaction_manifest}" \
      && ! -L "${transaction_manifest}" \
      && "${transaction_sha256}" =~ ^[0-9a-f]{64}$ \
      && "${transaction_id}" =~ ^[0-9A-Za-z_.-]+$ ]] \
    || fail "host finalizer returned an unsafe prepared transaction"
  ACTIVE_TRANSACTION_MANIFEST="${transaction_manifest}"
  ACTIVE_TRANSACTION_SHA256="${transaction_sha256}"
  ACTIVE_TRANSACTION_REASON="${engine} wrapper exited before host transaction commit"
  trap cleanup_active_work EXIT
  trap 'terminate_active_work 130' INT
  trap 'terminate_active_work 143' TERM
  trap gpu_memory_sampler_alert USR1
  transaction_relative="${transaction_manifest#${ARTIFACT_MODELS}/}"
  transaction_container="/workspace/DS9/models/${transaction_relative}"
  rebuild_args+=(
    --transaction-manifest "${transaction_container}"
    --expected-transaction-sha256 "${transaction_sha256}"
  )
  common_args+=(
    --label noesis.ds9.role=engine-maintenance
    --label "noesis.ds9.transaction=${transaction_id}"
    --label "noesis.ds9.artifact-root-sha256=${ARTIFACT_ROOT_ID}"
    --mount "type=bind,src=${transaction_manifest},dst=${transaction_container},readonly"
    --runtime=nvidia
    --gpus "device=${GPU_DEVICE_INDEX}"
  )
  assert_container_name_available "${container}" "between ${engine} snapshot and launch"
  assert_no_maintenance_containers "between ${engine} transaction snapshot and launch"
  if ! GPU_OWNERS="$(
    nvidia-smi --id="${GPU_DEVICE_INDEX}" \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader 2>&1
  )"; then
    fail "unable to requery GPU ${GPU_DEVICE_INDEX} compute owners before ${engine} launch: ${GPU_OWNERS}"
  fi
  [[ -z "${GPU_OWNERS//[[:space:]]/}" ]] \
    || fail $'GPU compute owner(s) appeared after transaction snapshot; aborting before launch:\n'"${GPU_OWNERS}"
  common_args[0]=create
  local create_cmd=(docker "${common_args[@]}" --entrypoint python3 "${IMAGE_ID}" "${rebuild_args[@]}")
  print_command "${create_cmd[@]}"
  ACTIVE_CONTAINER="${container}"
  ACTIVE_CONTAINER_TRANSACTION_ID="${transaction_id}"
  ACTIVE_CONTAINER_ARTIFACT_ROOT_ID="${ARTIFACT_ROOT_ID}"
  ACTIVE_LAUNCH_ATTEMPTED=1
  local container_id
  if ! container_id="$("${create_cmd[@]}")"; then
    fail "failed to create maintenance container: ${container}"
  fi
  container_id="${container_id//[[:space:]]/}"
  [[ "${container_id}" =~ ^[0-9a-f]{64}$ ]] \
    || fail "Docker returned an invalid maintenance container ID: ${container_id}"
  ACTIVE_CONTAINER_ID="${container_id}"
  ACTIVE_ENGINE="${engine}"
  if ! start_active_gpu_memory_sampler \
      "${engine}" "${guard}" "${transaction_id}" "${transaction_sha256}" \
      "${container_id}" "${timeout}"; then
    fail "failed to establish the identity-bound NVML guard before ${engine} start"
  fi
  if ! docker start "${container_id}" >/dev/null; then
    fail "failed to start maintenance container under the active NVML guard: ${container}"
  fi

  local inspect_path container_init_pid init_identity_json init_start_time_ticks post_snapshot_state
  inspect_path="${ARTIFACT_ROOT}/logs/${engine}_$(date -u +%Y%m%dT%H%M%SZ).inspect.json"
  [[ ! -e "${inspect_path}" && ! -L "${inspect_path}" ]] \
    || fail "refusing to replace existing container-inspect evidence: ${inspect_path}"
  (umask 077; set -o noclobber; docker inspect "${container_id}" >"${inspect_path}") \
    || fail "failed to persist container-inspect evidence: ${inspect_path}"
  chmod 0600 -- "${inspect_path}"
  container_init_pid="$(python3 - \
    "${inspect_path}" "${container_id}" "${container}" "${IMAGE_ID}" "${CALLER_UID}:${CALLER_GID}" \
    "${REPO_ROOT}" "${ARTIFACT_MODELS}" \
    "${ARTIFACT_MODELS}/engines" "${ARTIFACT_MODELS}/engine_maintenance" \
    "${transaction_manifest}" "${transaction_container}" "${transaction_id}" "${ARTIFACT_ROOT_ID}" \
    "${MAINTENANCE_MEMORY_BYTES}" "${MAINTENANCE_PIDS_LIMIT}" \
    "${MAINTENANCE_LOG_MAX_SIZE}" "${MAINTENANCE_LOG_MAX_FILES}" <<'PY'
import json
import os
import sys
from pathlib import Path

(
    inspect_raw,
    container_id,
    container_name,
    image_id,
    expected_user,
    repo_source,
    model_source,
    engine_source,
    evidence_source,
    transaction_source,
    transaction_destination,
    transaction_id,
    artifact_root_id,
    memory_bytes,
    pids_limit,
    log_max_size,
    log_max_files,
) = sys.argv[1:]
rows = json.loads(Path(inspect_raw).read_text(encoding="utf-8"))
if len(rows) != 1:
    raise SystemExit("maintenance container inspect must contain exactly one object")
row = rows[0]
host = row.get("HostConfig") or {}
config = row.get("Config") or {}
state = row.get("State") or {}
if row.get("Id") != container_id or row.get("Name") != f"/{container_name}":
    raise SystemExit("maintenance container ID/name mismatch")
if row.get("Image") != image_id:
    raise SystemExit("maintenance container image ID mismatch")
if config.get("User") != expected_user:
    raise SystemExit("maintenance container user mismatch")
init_pid = state.get("Pid")
if not state.get("Running") or not isinstance(init_pid, int) or init_pid <= 1:
    raise SystemExit("maintenance container lacks a live inspected init PID")
labels = config.get("Labels") or {}
if labels.get("noesis.ds9.role") != "engine-maintenance" or labels.get(
    "noesis.ds9.transaction"
) != transaction_id:
    raise SystemExit("maintenance container transaction label mismatch")
if labels.get("noesis.ds9.artifact-root-sha256") != artifact_root_id:
    raise SystemExit("maintenance container artifact-root identity mismatch")
if not host.get("Init"):
    raise SystemExit("maintenance container init process is disabled")
if not host.get("ReadonlyRootfs") or host.get("NetworkMode") != "none":
    raise SystemExit("maintenance container root/network isolation mismatch")
if host.get("CapDrop") != ["ALL"]:
    raise SystemExit("maintenance container capability drop set is not exact")
if host.get("SecurityOpt") != ["no-new-privileges"]:
    raise SystemExit("maintenance container security options are not exact")
if host.get("Runtime") != "nvidia":
    raise SystemExit("maintenance container runtime is not nvidia")
if host.get("Memory") != int(memory_bytes) or host.get("MemorySwap") != int(memory_bytes):
    raise SystemExit("maintenance container memory/no-swap bound mismatch")
if host.get("PidsLimit") != int(pids_limit):
    raise SystemExit("maintenance container pids bound mismatch")
log_config = host.get("LogConfig") or {}
if log_config.get("Type") != "local" or log_config.get("Config") != {
    "max-file": log_max_files,
    "max-size": log_max_size,
}:
    raise SystemExit("maintenance container log bound mismatch")
requests = host.get("DeviceRequests") or []
if (
    len(requests) != 1
    or requests[0].get("DeviceIDs") != ["0"]
    or ["gpu"] not in (requests[0].get("Capabilities") or [])
):
    raise SystemExit("maintenance container is not pinned to GPU device 0")
mounts = row.get("Mounts") or []
by_destination = {entry.get("Destination"): entry for entry in mounts}
if len(by_destination) != len(mounts):
    raise SystemExit("maintenance container contains duplicate mount destinations")
expected_mounts = {
    "/workspace": (repo_source, False),
    "/workspace/DS9/models": (model_source, False),
    "/workspace/DS9/models/engines": (engine_source, True),
    "/workspace/DS9/models/engine_maintenance": (evidence_source, True),
    transaction_destination: (transaction_source, False),
}
if set(by_destination) != set(expected_mounts):
    raise SystemExit("maintenance container mount destination set is not exact")
for destination, (source, writable) in expected_mounts.items():
    entry = by_destination.get(destination) or {}
    if (
        entry.get("Type") != "bind"
        or os.path.realpath(str(entry.get("Source", ""))) != os.path.realpath(source)
        or bool(entry.get("RW")) is not writable
    ):
        raise SystemExit(f"maintenance container mount boundary mismatch: {mounts}")
print(init_pid)
PY
  )" || fail_active_container \
    "maintenance container inspect validation failed: ${container}" "" "" ""
  [[ "${container_init_pid}" =~ ^[1-9][0-9]*$ && "${container_init_pid}" != "1" ]] \
    || fail_active_container \
      "maintenance container returned an invalid inspected init PID: ${container_init_pid}" \
      "" "" ""
  ACTIVE_CONTAINER_INIT_PID="${container_init_pid}"
  if ! init_identity_json="$(
    python3 "${GPU_OWNERSHIP_HELPER}" \
      --container-init-pid "${container_init_pid}" --snapshot-init
  )"; then
    fail_active_container \
      "unable to snapshot the inspected container init identity: ${init_identity_json}" \
      "" "" "${init_identity_json}"
  fi
  init_start_time_ticks="$(
    python3 -c \
      'import json,sys; print(json.loads(sys.stdin.read())["container_init_start_time_ticks"])' \
      <<<"${init_identity_json}"
  )"
  [[ "${init_start_time_ticks}" =~ ^[1-9][0-9]*$ ]] \
    || fail_active_container \
      "container init identity returned an invalid start time: ${init_identity_json}" \
      "" "" "${init_identity_json}"
  if ! post_snapshot_state="$(
    docker inspect "${container_id}" --format '{{.State.Running}}|{{.State.Pid}}' 2>&1
  )"; then
    fail_active_container \
      "unable to re-inspect container after init identity snapshot: ${post_snapshot_state}" \
      "" "" "${init_identity_json}"
  fi
  [[ "${post_snapshot_state}" == "true|${container_init_pid}" ]] \
    || fail_active_container \
      "container init changed after identity snapshot: expected=true|${container_init_pid} observed=${post_snapshot_state}" \
      "" "" "${init_identity_json}"
  ACTIVE_CONTAINER_INIT_START_TIME_TICKS="${init_start_time_ticks}"

  started="$(date +%s)"
  local last_owner_check=0
  while [[ "$(docker inspect --format '{{.State.Running}}' "${container_id}" 2>/dev/null || echo false)" == "true" ]]; do
    if ! kill -0 "${ACTIVE_GPU_MEMORY_SAMPLER_PID}" 2>/dev/null; then
      collect_active_gpu_memory_sampler 0 || true
      fail_active_container \
        "${engine} NVML guard sampler exited before container completion (state=${ACTIVE_GPU_MEMORY_STATE:-unknown})" \
        "" "" ""
    fi
    now="$(date +%s)"
    if (( now - last_owner_check >= 3 )); then
      assert_gpu_owners_belong_to_container \
        "${engine}" "${container_init_pid}" "${init_start_time_ticks}"
      last_owner_check="${now}"
    fi
    if (( now - started > timeout )); then
      fail_active_container "${engine} exceeded the ${timeout}s build timeout" "" "" ""
    fi
    sleep 0.1
  done

  if ! collect_active_gpu_memory_sampler 1; then
    fail_active_container \
      "${engine} NVML guard evidence was not clean at container exit (state=${ACTIVE_GPU_MEMORY_STATE:-unknown})" \
      "" "" ""
  fi
  trap - USR1

  status="$(docker inspect --format '{{.State.ExitCode}}' "${container_id}")"
  log_path="${ARTIFACT_ROOT}/logs/${engine}_$(date -u +%Y%m%dT%H%M%SZ).log"
  [[ ! -e "${log_path}" && ! -L "${log_path}" ]] \
    || fail "refusing to replace existing container-log evidence: ${log_path}"
  (
    umask 077
    set -o noclobber
    {
      echo "image=${IMAGE}"
      echo "image_id=${IMAGE_ID}"
      echo "engine=${engine}"
      echo "output=${output}"
      echo "inspect=${inspect_path}"
      echo "gpu_memory_total_mib=${GPU_MEMORY_MIB}"
      echo "gpu_memory_guard_mib=${guard}"
      echo "gpu_memory_peak_mib=${ACTIVE_GPU_MEMORY_PEAK_MIB}"
      echo "gpu_memory_sample_count=${ACTIVE_GPU_MEMORY_SAMPLE_COUNT}"
      echo "gpu_memory_sampler_state=${ACTIVE_GPU_MEMORY_STATE}"
      echo "gpu_memory_evidence=${ACTIVE_GPU_MEMORY_EVIDENCE}"
      echo "gpu_memory_evidence_sha256=${ACTIVE_GPU_MEMORY_EVIDENCE_SHA256}"
      echo "gpu_memory_first_sample_utc=${ACTIVE_GPU_MEMORY_FIRST_SAMPLE_UTC}"
      echo "gpu_memory_last_sample_utc=${ACTIVE_GPU_MEMORY_LAST_SAMPLE_UTC}"
      echo "gpu_memory_maximum_gap_ms=${ACTIVE_GPU_MEMORY_MAXIMUM_GAP_MS}"
      docker logs "${container_id}" 2>&1 || true
    } >"${log_path}"
  ) || fail "failed to persist container-log evidence: ${log_path}"
  chmod 0600 -- "${log_path}"
  cat -- "${log_path}"
  cleanup_active_container \
    || fail "failed to prove completed maintenance-container removal: ${container}"

  [[ "${status}" == "0" ]] || fail "${engine} build container exited ${status}; see ${log_path}"
  [[ -s "${output}" ]] || fail "${engine} did not create ${output}"
  local manifest_virtual manifest_suffix manifest_host manifest_resolved evidence_prefix
  manifest_virtual="$(sed -n 's/^\[EVIDENCE\] //p' "${log_path}" | tail -1)"
  evidence_prefix="/workspace/DS9/models/engine_maintenance/"
  [[ "${manifest_virtual}" == "${evidence_prefix}"* ]] \
    || fail "${engine} did not report a canonical maintenance manifest; see ${log_path}"
  manifest_suffix="${manifest_virtual#${evidence_prefix}}"
  [[ -n "${manifest_suffix}" && "${manifest_suffix}" != /* && "${manifest_suffix}" != *".."* ]] \
    || fail "${engine} reported an unsafe maintenance manifest path: ${manifest_virtual}"
  manifest_host="${ARTIFACT_MODELS}/engine_maintenance/${manifest_suffix}"
  [[ -f "${manifest_host}" && ! -L "${manifest_host}" ]] \
    || fail "${engine} maintenance manifest is missing or unsafe: ${manifest_host}"
  manifest_resolved="$(realpath -e -- "${manifest_host}")"
  [[ "${manifest_resolved}" == "${ARTIFACT_MODELS}/engine_maintenance/"* ]] \
    || fail "${engine} maintenance manifest escaped its evidence root: ${manifest_resolved}"
  python3 - \
    "${manifest_resolved}" "${engine}" "${output}" \
    "${IMAGE}" "${IMAGE_ID}" "${BASE_DIGEST}" "${IMAGE_TRT_VERSION}" \
    "${IMAGE_CUDA_VERSION}" "${HOST_DRIVER_VERSION}" "${GPU_NAME}" \
    "${GPU_UUID}" "${GPU_COMPUTE_CAPABILITY}" "${GPU_MEMORY_MIB}" <<'PY'
import hashlib
import json
import stat
import sys
from pathlib import Path

(
    manifest_raw,
    engine_name,
    output_raw,
    image,
    image_id,
    base_digest,
    trt_version,
    cuda_version,
    driver_version,
    gpu_name,
    gpu_uuid,
    gpu_cc,
    gpu_memory,
) = sys.argv[1:]
manifest_path = Path(manifest_raw)
output = Path(output_raw)
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
if payload.get("contract") != "noesis.ds9.engine_maintenance":
    raise SystemExit("maintenance manifest contract mismatch")
if payload.get("status") != "complete" or payload.get("engine") != engine_name:
    raise SystemExit("maintenance manifest is not complete for the selected engine")
if "rollback_after_install_failure" in payload:
    raise SystemExit("completed maintenance manifest unexpectedly contains rollback evidence")
commands = {row.get("label"): row for row in payload.get("commands", [])}
build_labels = {"build", "build-tracker-engine"}
if not any(commands.get(label, {}).get("status") == "passed" for label in build_labels):
    raise SystemExit("maintenance manifest lacks a build command")
for label in ("probe-trtexec", "load-candidate", "load-installed"):
    if commands.get(label, {}).get("status") != "passed":
        raise SystemExit(f"maintenance manifest lacks passed {label} evidence")
installed = payload.get("installed") or {}
digest = hashlib.sha256(output.read_bytes()).hexdigest()
if installed.get("sha256") != digest or installed.get("size_bytes") != output.stat().st_size:
    raise SystemExit("installed output hash/size differs from maintenance manifest")
if engine_name == "mapanything":
    quality_command = commands.get("mapanything-functional-quality") or {}
    command = quality_command.get("command")
    if (
        quality_command.get("status") != "passed"
        or quality_command.get("proof") != "trtexec_inference"
        or quality_command.get("returncode") != 0
        or quality_command.get("timed_out") is not False
        or quality_command.get("output_exceeded") is not False
        or not isinstance(command, list)
        or "--dumpOutput" not in command
        or any("--skipInference" in str(item) for item in command)
    ):
        raise SystemExit("MapAnything maintenance lacks real-inference quality command evidence")
    receipt = (payload.get("evidence") or {}).get("functional_quality") or {}
    receipt_engine = receipt.get("engine") or {}
    if (
        receipt.get("contract") != "noesis.ds9.mapanything_functional_quality.v1"
        or receipt.get("status") != "passed"
        or receipt_engine.get("sha256") != digest
        or receipt_engine.get("size_bytes") != output.stat().st_size
    ):
        raise SystemExit("MapAnything functional-quality receipt differs from installed bytes")
platform = (payload.get("metadata") or {}).get("platform") or {}
expected = {
    "image": image,
    "image_id": image_id,
    "base_digest": base_digest,
    "tensorrt_version": trt_version,
    "cuda_version": cuda_version,
    "driver_version": driver_version,
    "gpu_name": gpu_name,
    "gpu_uuid": gpu_uuid,
    "gpu_compute_capability": gpu_cc,
    "gpu_memory_mib": gpu_memory,
    "expected_trtexec_banner": "TensorRT v101401",
}
if any(str(platform.get(key)) != str(value) for key, value in expected.items()):
    raise SystemExit("maintenance manifest platform provenance mismatch")
if stat.S_IMODE(manifest_path.stat().st_mode) != 0o600:
    raise SystemExit("maintenance manifest is not mode 0600")
if stat.S_IMODE(manifest_path.parent.stat().st_mode) != 0o700:
    raise SystemExit("maintenance evidence directory is not mode 0700")
PY
  local commit_json commit_status
  if ! commit_json="$(
    python3 "${FINALIZER}" commit \
      --transaction-manifest "${transaction_manifest}" \
      --expected-transaction-sha256 "${transaction_sha256}" \
      --maintenance-manifest "${manifest_resolved}" \
      --gpu-guard-evidence "${ACTIVE_GPU_MEMORY_EVIDENCE}" \
      --expected-gpu-guard-sha256 "${ACTIVE_GPU_MEMORY_EVIDENCE_SHA256}" \
      --gpu-guard-container-id "${ACTIVE_GPU_MEMORY_CONTAINER_ID}" \
      --gpu-guard-wrapper-pid "$$" \
      --gpu-guard-wrapper-start-time-ticks "${ACTIVE_GPU_MEMORY_WRAPPER_START_TIME_TICKS}" \
      --lock-fd 9
  )"; then
    ACTIVE_TRANSACTION_REASON="${engine} host finalizer commit failed"
    fail "${engine} engine/realization commit failed"
  fi
  # Exit zero is the finalizer's atomic commit boundary. Clear the rollback
  # trap before interpreting its informational JSON; committed transactions
  # must never be reverted by a presentation-layer parse failure.
  ACTIVE_TRANSACTION_MANIFEST=""
  ACTIVE_TRANSACTION_SHA256=""
  trap - EXIT INT TERM
  commit_status="$(
    python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["status"])' \
      <<<"${commit_json}"
  )"
  [[ "${commit_status}" == "committed" ]] \
    || fail "${engine} finalizer returned an unexpected success payload"
  echo "[FINALIZE] ${commit_json}"
  echo "[OK] ${engine}: $(stat -c '%s bytes' "${output}") sha256=$(sha256sum "${output}" | awk '{print $1}') manifest=${manifest_resolved}"
}

for engine in "${ENGINES[@]}"; do
  run_engine "${engine}"
done

if (( PLAN )); then
  echo "[OK] canonical DS9 engine maintenance plan complete"
else
  if [[ "${MODE}" == "all" || "${MODE}" == "v3dt" ]]; then
    PROFILE="canonical"
    [[ "${MODE}" == "v3dt" ]] && PROFILE="v3dt"
    python3 "${SCRIPT_DIR}/validate_asset_manifest.py" \
      --artifact-root "${ARTIFACT_ROOT}" \
      --check-files \
      --profile "${PROFILE}" \
      --require-provenance \
      --require-realization
  fi
  echo "[OK] canonical DS9 engine maintenance complete"
fi

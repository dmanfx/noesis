# DS9 Host Driver Migration And DS8 Rollback

Status: driver transaction, post-reboot platform checks, canonical DS8 lifecycle
acceptance, and the non-mutating offline rollback rehearsal are complete. DS8
product soak and DS9 build/runtime gates remain open. The rollback itself has
not been executed.

NVIDIA's DeepStream 9 dGPU stack requires Ubuntu 24.04, driver 590.48.01 or
later, CUDA 13.1, and TensorRT 10.14.1.48. On Noble,
`nvidia-driver-590-open` is a transitional package for 595. This host therefore
uses Ubuntu's previously exercised `nvidia-driver-595-open` 595.71.05 package,
not NVIDIA's runfile installer. Mixing the runfile with the existing APT/DKMS
stack would weaken rollback.

This is a driver-only migration. Keep the host CUDA default on 13.0 and host
TensorRT on 10.13.3 for DS8. DS9 TensorRT 10.14 builds run only in the pinned
isolated image.

The 2026-07-10 post-reboot audit confirms `595.71.05` across `nvidia-smi`, the
loaded kernel module, the on-disk module, and DKMS for both installed kernels.
Package audit is clean, the current boot has no Xid/API-mismatch/GPU-loss event,
and desktop/RDP recovery passed. All configured DS8 engines deserialize. The
canonical authenticated DS8 YOLO26m gate subsequently completed 30 seconds of
advancing world state, acknowledged downstream EOS, observed the expected EOS
callback and Service Maker `wait()` return, exited `0`, and required no forced
kill. This clears the DS8 native-shutdown blocker, but the driver result alone
still does not authorize cutover: remaining DS8 product/soak canaries apply,
and DS9 TensorRT work must use an announced exclusive-GPU window followed by
profile-specific runtime acceptance. V3DT still requires its two DS9 engines
and its own live bridge/world/identity/resource/shutdown gates.

## Required checkpoint

Set the private NVMe checkpoint created for the maintenance window:

```bash
CHECKPOINT_DIR="${CHECKPOINT_DIR:?set the verified pre-driver checkpoint}"
TARGET_CACHE="$CHECKPOINT_DIR/host/packages/target-595"
ROLLBACK_CACHE="$CHECKPOINT_DIR/host/packages/rollback-580"
test "$(find "$TARGET_CACHE" -maxdepth 1 -type f -name '*.deb' | wc -l)" -eq 21
test "$(find "$ROLLBACK_CACHE" -maxdepth 1 -type f -name '*.deb' | wc -l)" -eq 33
sha256sum -c "$CHECKPOINT_DIR/SHA256SUMS"
python3 DS9/scripts/driver_rollback_rehearsal.py \
  --checkpoint "$CHECKPOINT_DIR"
```

The checkpoint must also contain the dirty working sources, complete Git
bundles, exact live DS8 engines/native bridges, package selections and holds,
DKMS and boot artifacts, prior DS9 upgrade evidence, process commands/private
environments, and the current listener/service state.

Before mutation, require matching kernel headers, Secure Boot disabled, a clean
`dpkg --audit`, a persistent text/SSH recovery path, and an announced exclusive
GPU window. Stop every GPU owner and the display manager. Do not start another
GPU process after the userspace libraries change.

## Exact 595 transaction

Re-run this once with `--simulate`; it must still report 21 new packages,
25 removals, no CUDA/TensorRT change, and the exact 595.71.05 target. Then omit
`--simulate`:

```bash
sudo apt-get --simulate --allow-change-held-packages \
  -o Dir::Cache::archives="$TARGET_CACHE" install \
  nvidia-driver-595-open=595.71.05-0ubuntu0.24.04.1 \
  libnvidia-cfg1-580- \
  libnvidia-compute-580:amd64- libnvidia-compute-580:i386- \
  libnvidia-decode-580:amd64- libnvidia-decode-580:i386- \
  libnvidia-egl-gbm1:amd64- libnvidia-egl-gbm1:i386- \
  libnvidia-egl-xcb1:amd64- libnvidia-egl-xcb1:i386- \
  libnvidia-egl-xlib1:amd64- libnvidia-egl-xlib1:i386- \
  libnvidia-encode-580:amd64- libnvidia-encode-580:i386- \
  libnvidia-fbc1-580:amd64- libnvidia-fbc1-580:i386- \
  libnvidia-gl-580:amd64- libnvidia-gl-580:i386- \
  nvidia-compute-utils-580- nvidia-dkms-580-open- \
  nvidia-driver-580-open- nvidia-kernel-common-580- \
  nvidia-kernel-source-580-open- nvidia-persistenced- \
  nvidia-utils-580- xserver-xorg-video-nvidia-580-
```

The explicit EGL removals prevent the file collision observed during the first
June 2026 attempt. For apply, add `-y` and remove `--simulate`; do not change
the package list.

Before reboot, `dpkg --audit` must be empty and both `dkms status` and
`modinfo -k "$(uname -r)" -F version nvidia` must report 595.71.05 for the
running kernel. A pre-reboot NVML kernel/userspace mismatch is expected because
the loaded module is still 580; reboot directly instead of launching workloads.

## Post-reboot acceptance

Do not build a DS9 engine until all of these pass:

1. `nvidia-smi`, `/proc/driver/nvidia/version`, DKMS, and the on-disk module all
   agree on 595.71.05; `dpkg --audit` is empty.
2. The current boot has no `NVRM: Xid`, API mismatch, GPU-fallen-off, or
   `RmInitAdapter` failure.
3. `/usr/local/cuda` still resolves to CUDA 13.0, Python TensorRT remains
   10.13.3.9, and `deepstream-app --version-all` remains DS8.0.
4. Every configured DS8 engine deserializes, then the focused 30-second DS8
   runtime gate passes.
5. Authenticated capability health, RTSP decode, decoded WebRTC, identity
   telemetry, MapAnything/floorplan quality, resource bounds, shutdown, and a
   DS8 soak pass.
6. The graphical session and RDP path pass. Two historical `nvidia_drm`
   NvKms memory-page warnings were non-fatal, but this is still an explicit
   canary because the appliance also serves as a desktop host.

Only after DS8 passes may the isolated DS9 image build BodyPose3DNet, then the
NvMOT tracker-ReID engine. MapAnything must be rebuilt and validated from
tensor finiteness through height/floorplan quality; a loadable engine alone is
not acceptance.

## Offline 580 rollback

Rollback triggers include module/userspace incoherence, DKMS failure, Xids,
desktop/RDP failure, any DS8 engine deserialization failure, or any DS8
runtime/media/identity/MapAnything regression.

Use the exact cached 580.167.08 packages. The former install-only command is not
a valid transaction from the held 595 state: APT correctly refuses to leave
the mutually conflicting 595 and 580 packages installed together. The exact
rollback must therefore name the 33 local archives and explicitly remove the
held 595 cohort. `nvidia-prime` is the only non-595 removal; the exact
`nvidia-driver-580-open` archive declares `Conflicts` and `Replaces` on it, an
otherwise identical simulation without the explicit removal still removes it,
and the verified pre-driver baseline records it as removed/config-files-only.

First run the non-mutating validator. It rehashes the complete checkpoint and
all package/control-relationship bytes, verifies current 595 module/DKMS/boot
coherence without using the GPU API, and runs the exact local-only solver:

```bash
python3 DS9/scripts/driver_rollback_rehearsal.py \
  --checkpoint "$CHECKPOINT_DIR"
```

Require `status: validated_rehearsal`, `rollback_executed: false`, 33 verified
archives, 25 installs, 21 removals, and no CUDA/TensorRT/DeepStream changes.
Then preview this exact transaction. It permits no package download:

```bash
sudo apt-get --simulate --no-download --allow-downgrades \
  --allow-change-held-packages install "$ROLLBACK_CACHE"/*.deb \
  libnvidia-cfg1-595- libnvidia-common-595- \
  libnvidia-compute-595:amd64- libnvidia-compute-595:i386- \
  libnvidia-decode-595:amd64- libnvidia-decode-595:i386- \
  libnvidia-encode-595:amd64- libnvidia-encode-595:i386- \
  libnvidia-extra-595- \
  libnvidia-fbc1-595:amd64- libnvidia-fbc1-595:i386- \
  libnvidia-gl-595:amd64- libnvidia-gl-595:i386- \
  nvidia-compute-utils-595- nvidia-dkms-595-open- \
  nvidia-driver-595-open- nvidia-kernel-common-595- \
  nvidia-kernel-source-595-open- nvidia-prime- nvidia-utils-595- \
  xserver-xorg-video-nvidia-595-
```

Only in an approved exclusive maintenance window, repeat the identical command
with `--simulate` replaced by `-y`. Do not omit `--no-download`, alter the local
archive glob, add `autoremove`, or broaden the removal set.

Require clean package state, coherent 580 DKMS/module output, regenerated
initramfs, and a reboot. Then repeat the same DS8 acceptance gates; restoring a
package version without product evidence is not a successful rollback. The
ordered evidence and acceptance boundary are recorded in
`driver_580_rollback_rehearsal_2026-07-10.md`.

References:

- NVIDIA DeepStream 9 installation requirements:
  <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html>
- Ubuntu Noble transitional 590 package:
  <https://packages.ubuntu.com/noble-updates/nvidia-driver-590-open>

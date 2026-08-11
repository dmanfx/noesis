# Driver 580 Rollback Rehearsal - 2026-07-10

## Result

Status: **validated non-mutating rehearsal; rollback not executed**.

The current host stayed on driver 595.71.05. No package, hold, DKMS, initramfs,
boot, service, process, display, listener, network, or GPU state was changed.
The validator does not call `nvidia-smi`; its platform inspection is CPU-only.

The real JSON evidence is stored owner-only at
`20260710T203037Z-driver595-postboot/driver-580-rollback-rehearsal.json` under
the private checkpoint root. It is a single-link 0600 file with SHA-256
`94620d38ac34f8864de27e0ddcb3d10cc650677c8f3eebe005495a10a3adcde4`.

The exact pre-driver checkpoint passed a complete byte rehash:

- `SHA256SUMS` authority:
  `1b7b431c70459b1bed5ab45b1f076a378755f17a3f720fdac5ff3069713da839`.
- 845 declared regular single-link files verified.
- 9,061,703,053 declared bytes verified.
- The owner-only checkpoint boundary and manifest remained intact.

The rollback cache passed its independent contract:

- 33 exact local Debian archives, totaling 392,013,262 bytes.
- Architecture set: 23 amd64, 9 i386, and 1 all.
- Every archive SHA-256 matched both the versioned rollback specification and
  the checkpoint inventory.
- Every package identity, version, architecture, and normalized `Pre-Depends`,
  `Depends`, `Recommends`, `Suggests`, `Breaks`, `Conflicts`, `Replaces`,
  `Provides`, and `Multi-Arch` relationship digest matched.

The current starting state also passed: loaded and on-disk modules plus DKMS
for both installed boot kernels agree on 595.71.05, `dpkg --audit` is empty,
i386 is enabled, and the host CUDA default remains 13.0.

## Solver finding

The prior install-only rollback command was not executable from the current
held 595 package state. APT refused the mutually conflicting 580 and 595
cohorts. The corrected exact local-only simulation explicitly removes the 595
cohort and succeeds with:

- 0 upgrades;
- 25 exact local installs;
- 21 removals;
- 8 exact rollback packages already installed;
- no CUDA, TensorRT, cuDNN, or DeepStream package action;
- no package download permitted.

`nvidia-prime` is the only removal without `595` in its package name. This is
not discretionary desktop cleanup: the exact cached `nvidia-driver-580-open`
metapackage declares both `Conflicts` and `Replaces` on `nvidia-prime`. An
otherwise identical simulation with only the 595 removals still removes
`nvidia-prime`, and the known-working pre-driver snapshot records it as `rc`
rather than installed. The versioned specification rejects any other generic
removal unless an exact package relationship documents its solver necessity.

## Reproduce the rehearsal

Set the checkpoint path without copying it into the repository:

```bash
CHECKPOINT_DIR="${CHECKPOINT_DIR:?set the private pre-driver checkpoint}"
POSTBOOT_EVIDENCE="${POSTBOOT_EVIDENCE:?set a private owner-only evidence directory}"
python3 DS9/scripts/driver_rollback_rehearsal.py \
  --checkpoint "$CHECKPOINT_DIR" \
  --output "$POSTBOOT_EVIDENCE/driver-580-rollback-rehearsal.json"
```

A pass reports `status: validated_rehearsal`, `rollback_executed: false`, and
`host_mutation_performed: false`. The versioned authority is
`DS9/config/driver_rollback_580.json`. The validator fails closed on checkpoint,
archive, control-relationship, architecture, current platform, solver action,
or protected SDK drift. Evidence output uses an atomic, fsynced, no-replace
publish into an owner-only directory, creates mode 0600, and rejects symlinks.

## Ordered future execution boundary

This section is authorization guidance, not evidence that these steps ran.

1. Announce an exclusive maintenance window. Preserve current evidence, stop
   DS8/DS9 and every GPU owner, stop the display manager, and prove a persistent
   console or SSH recovery path.
2. Rerun the complete validator and require a green result against the same
   checkpoint authority.
3. Run the exact `--simulate --no-download` command printed by the validator;
   require 25 installs, 21 removals, and no protected SDK action.
4. Repeat that exact command with `--simulate` replaced by `-y`. Do not run
   `autoremove`, permit downloads, or broaden its removal set.
5. Require an empty `dpkg --audit`, 580.167.08 DKMS and `modinfo` for every
   installed boot kernel, then run `sudo update-initramfs -u -k all`.
6. Reboot once. Do not launch GPU work while loaded-module and userspace
   versions disagree.
7. Complete the post-rollback DS8 acceptance contract before restoring any
   appliance or DS9 service.

An actual rollback therefore still requires an approved disruptive maintenance
window and reboot. The rehearsal is sufficient to prove the checkpoint,
offline inputs, dependency transaction, and recovery order; it cannot prove the
post-reboot driver or product behavior.

## Post-rollback DS8 acceptance contract

A rollback is successful only when all of these pass after reboot:

1. `nvidia-smi`, the loaded module, every installed boot-kernel module, and
   DKMS agree on 580.167.08; `dpkg --audit` is empty; the current boot has no
   Xid, API mismatch, GPU-loss, or `RmInitAdapter` failure.
2. `/usr/local/cuda` remains CUDA 13.0, Python TensorRT remains 10.13.3.9, and
   `deepstream-app --version-all` remains DeepStream 8.0.
3. Every configured DS8 TensorRT engine deserializes from its exact preserved
   bytes.
4. The authenticated baseline `scripts/ds8_runtime_30s_gate.py` passes real
   readiness, advancing REST/WebSocket/world state, RTSP DESCRIBE, acknowledged
   downstream EOS, expected callback and wait completion, exit 0, and no
   forced kill.
5. The authenticated V3DT profile of the same gate passes its tracker and
   metadata contract plus the same lifecycle requirements.
6. Decoded WebRTC, identity telemetry, MapAnything depth/floorplan quality,
   persistence, and bounded CPU/GPU/memory behavior match the accepted DS8
   baseline.
7. Desktop and RDP recover, followed by the required supervised DS8 soak.

Package versions alone never satisfy this acceptance contract.

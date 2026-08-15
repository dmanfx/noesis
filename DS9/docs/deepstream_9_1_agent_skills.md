# DeepStream 9.1 agent skill routing

DeepStream work in this repository starts with the official NVIDIA skill that
matches the task. Read that skill's complete `SKILL.md`, then the references it
selects, before editing SDK-facing code. Repository policy and `DS9/AGENTS.md`
are authoritative when an upstream example still names DeepStream 9.0, older
CUDA/TensorRT releases, or Docker.

## First route

| Work | Start here | Expected result |
| --- | --- | --- |
| Service Maker, GStreamer, configs, tracker, debugging | `.agents/skills/deepstream-dev/SKILL.md` | Verified APIs and plugin/config keys before code changes |
| New pipeline graph or generated configuration | `.agents/skills/deepstream-generate-pipeline/SKILL.md` | Generated graph/config is reviewed and validated before integration |
| ONNX/model import or TensorRT engine realization | `.agents/skills/deepstream-import-vision-model/SKILL.md` | Model contract is inspected, engine is built on the pinned stack, and the direct lane is exercised |
| FPS, latency, utilization, or capacity work | `.agents/skills/deepstream-profile-pipeline/SKILL.md` | Matched-input measurements and a bounded capacity result |
| Multi-view 3D tracking | `.agents/skills/deepstream-run-mv3dt/SKILL.md` | Official MV3DT prerequisites/config are followed without collapsing into SV3DT |
| SOP reference application | `.agents/skills/deepstream-sop/SKILL.md` | SOP-specific workflow only; it is not the default Noesis application route |

If the selected skill does not cover a required Noesis contract, record the gap
and continue with the verified SDK API plus repository contracts. Never invent a
DeepStream method, property, or config key.

## Pinned runtime facts

- DeepStream SDK: 9.1
- CUDA: 13.2
- TensorRT: 10.16.0.72
- SDK root: `/opt/nvidia/deepstream/deepstream-9.1` (or the vendor link that
  resolves there)
- Minimum driver: `595.58.03` (current host `595.71.05` passes)
- Python: 3.12 in the native root selected by `NOESIS_DS91_NATIVE_ROOT`
- DS8/DS9.0/container engines, parsers, GStreamer plugins, and native Python
  extensions are incompatible inputs and must not be loaded.

The accepted execution baseline is recorded in `docs/runtime_baseline.md` and
`DS9/docs/runtime_host_boundary.md`.

## MV3DT and AMC status

SV3DT remains the existing single-view lane. MV3DT is a separate capability;
selecting it must never fall back to or alias SV3DT.

The current physical topology does not provide Living Room/Family Room overlap.
The only prospective MV3DT edge is Kitchen/Family Room, and it remains disabled
until the Kitchen geometry is corrected and synchronized occupied overlap data
passes review. AMC execution is deferred for the same reason. The four AMC
skills are installed as documentation and future workflow entrypoints only:

- `amc-setup-calibration-stack`
- `amc-run-sample-calibration`
- `amc-run-video-calibration`
- `amc-run-rtsp-calibration`

Do not start the calibration stack, generate an AMC dataset, or claim a valid
multi-view calibration while this deferment is in effect.

## Practical validation

Use the smallest direct checks that cover the changed producer, contract, and
consumer. Build each affected native/parser family once on 9.1, deserialize the
affected engines, run a short recorded smoke, and exercise the live camera path
only when its required private assets are present. Profiling uses matched inputs;
documentation-only skill routing changes require only the docs-consistency
check.

## Default execution mode

The skill is an implementation and application-validation entrypoint, not a
release-promotion workflow. After reading the skill and routed references,
prefer direct native-host runtime checks, focused tests, and one bounded
live or recorded smoke. Do not create immutable appliance releases, clone state,
render bundles, build deployment selectors, publish candidates, or rehearse
rollback for ordinary DS9 work. Use those Menon/appliance mechanics only for an
explicitly requested production promotion or a change whose behavior cannot be
exercised without an external service/state lifecycle transition. A missing
private asset or unavailable GPU is a reported blocker, not a reason to widen
the validation ceremony.

The upstream `docker_containers.md` reference is not an execution option for
Noesis. Read it only when comparing a generic vendor example; do not copy its
commands, image pins, mounts, or installation assumptions into this repository.

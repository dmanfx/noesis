# C03: Matched depth outputs for Chisel geometry preservation

State: experiment specification; use the operator status command for observed runs.

Task: `atlasChisel`. Dependencies: C01, C02.
Operator reviews: SMOKE, C02.

Prepare an external bundle under the [shared contract](../../config/preparation.schema.json).
It must bind the frozen baseline, every local asset hash, world frame, units,
source identities, camera poses and transform/renderer provenance. Keep real
holdouts out of all context and appearance input assets. Returned/generated
content remains generated evidence and cannot become calibration authority.

Render accepted PCF surfels/geometry into **metric optical-axis planar-z**
EXRs at explicit 1280x720 target cameras. Record which retained PCF-to-backend
transform was selected and why. Never apply the DA3 transform to fused raw.
Use one depth-only context per target in target order; source photographs carry
only image and matching camera. Keep unknown depth non-finite/zero. Capture
render visibility and depth/ray conversion. A depth-return experiment needs at
least two translated targets. Prompt, geometry, cameras and sampling controls
stay fixed when changing anchor count; C03 compares actual returned depth.

```bash
python3 tools/atlas_bench/atlas_bench.py prepare C03 --bundle /external/reviewed-inputs.json
python3 tools/atlas_bench/atlas_bench.py run C03
python3 tools/atlas_bench/atlas_bench.py report C03
```

Missing preparation or review is a blocked state, never a substituted input.

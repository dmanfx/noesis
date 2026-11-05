# R01: Atlas view at independent calibrated fixed-camera pose

State: experiment specification; use the operator status command for observed runs.

Task: `atlasGenerate`. Dependencies: A04.
Operator reviews: SMOKE, A04.

Prepare an external bundle under the [shared contract](../../config/preparation.schema.json).
It must bind the frozen baseline, every local asset hash, world frame, units,
source identities, camera poses and transform/renderer provenance. Keep real
holdouts out of all context and appearance input assets. Returned/generated
content remains generated evidence and cannot become calibration authority.

Register Atlas cameras to the retained metric PCF/DA3 gauge in an explicit,
reviewed offline step using **input camera correspondences only**. Record the
Sim(3), residuals, scale and transform hash. Map known target poses through that
registration while scaling camera positions and depth together. Do not fit the
transform on held-out RGB/depth or change canonical calibration. Compare generated
views with untouched real targets as a separate metric step. Static frames are
validation only, never phone inference inputs. R02 is capped at 10 cm and 2 degrees.

```bash
python3 tools/atlas_bench/atlas_bench.py prepare R01 --bundle /external/reviewed-inputs.json
python3 tools/atlas_bench/atlas_bench.py run R01
python3 tools/atlas_bench/atlas_bench.py report R01
```

Missing preparation or review is a blocked state, never a substituted input.

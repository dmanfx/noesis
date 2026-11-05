# A05: Inspect generated completion outside measured coverage

State: experiment specification; use the operator status command for observed runs.

Task: `atlasGenerate`. Dependencies: A04.
Operator reviews: SMOKE, A04.

Prepare an external bundle under the [shared contract](../../config/preparation.schema.json).
It must bind the frozen baseline, every local asset hash, world frame, units,
source identities, camera poses and transform/renderer provenance. Keep real
holdouts out of all context and appearance input assets. Returned/generated
content remains generated evidence and cannot become calibration authority.

Record target camera paths and exactly which reviewed generated frames are
reused as context. A05 covers unknown areas; A06 extends from reviewed A05 output.
Do not label generated completion as new measurement or measured coverage.

```bash
python3 tools/atlas_bench/atlas_bench.py prepare A05 --bundle /external/reviewed-inputs.json
python3 tools/atlas_bench/atlas_bench.py run A05
python3 tools/atlas_bench/atlas_bench.py report A05
```

Missing preparation or review is a blocked state, never a substituted input.

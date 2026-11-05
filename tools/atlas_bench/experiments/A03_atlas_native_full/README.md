# A03: Atlas-native full selected cohort excluding six holdouts

State: experiment specification; use the operator status command for observed runs.

Task: `images2PosedRGBD`. Dependencies: A02.
Operator reviews: SMOKE, A02.

Use the frozen `full` cohort, preserving source order and image hashes.
Pose refinement stays false unless selected with an explicit new variant run ID.

```bash
python3 tools/atlas_bench/atlas_bench.py run A03
python3 tools/atlas_bench/atlas_bench.py report A03
```

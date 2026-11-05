# A02: Atlas-native medium reconstruction

State: experiment specification; use the operator status command for observed runs.

Task: `images2PosedRGBD`. Dependencies: A01.
Operator reviews: SMOKE, A01.

Use the frozen `medium` cohort, preserving source order and image hashes.
Pose refinement stays false unless selected with an explicit new variant run ID.

```bash
python3 tools/atlas_bench/atlas_bench.py run A02
python3 tools/atlas_bench/atlas_bench.py report A02
```

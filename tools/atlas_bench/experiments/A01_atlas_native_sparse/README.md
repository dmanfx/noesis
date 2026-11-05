# A01: Atlas-native sparse reconstruction

State: experiment specification; use the operator status command for observed runs.

Task: `images2PosedRGBD`. Dependencies: SMOKE.
Operator reviews: SMOKE.

Use the frozen `sparse` cohort, preserving source order and image hashes.
Pose refinement stays false unless selected with an explicit new variant run ID.

```bash
python3 tools/atlas_bench/atlas_bench.py run A01
python3 tools/atlas_bench/atlas_bench.py report A01
```

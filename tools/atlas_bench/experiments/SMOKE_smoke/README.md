# SMOKE: Prove two-frame private upload, posing, download and parsing

State: experiment specification; use the operator status command for observed runs.

Task: `images2PosedRGBD`. Dependencies: none.
Operator reviews: none.

Use the frozen `smoke` cohort, preserving source order and image hashes.
Pose refinement stays false unless selected with an explicit new variant run ID.

```bash
python3 tools/atlas_bench/atlas_bench.py smoke
python3 tools/atlas_bench/atlas_bench.py report SMOKE
```

# Evaluation hooks

Every successful run writes `metrics.json`. Available transport/runtime/parser
facts are measured; evaluation hooks remain `not_measured` until the required
matched evidence is supplied. This first pass supplies those hooks rather than
claiming a reconstruction evaluation from a smoke test.

| Hook | Required evidence |
| --- | --- |
| Plane/wall/floor residuals | Recorded Atlas-to-metric registration and independent labeled planes |
| Cross-view consistency | Joint views, exact cameras and depth visibility/occlusion masks |
| Metric scale error | Independent known metric geometry; scale depth and camera positions together |
| Coverage/completeness | Matched world crop, pixel grid and visibility; missing/generated/measured distinguished |
| Floaters/outliers | Finite positive depth and independent support/visibility |
| Held-out RGB | Generated at exact held-out cameras; untouched real source pixels; declared color metric |
| Held-out depth | Matched rays; provider depth comparisons labeled correlated; static compared separately |
| Confidence versus error | Independent residuals on matched valid pixels; provider score semantics retained |
| Runtime and API cost | Command/operation timing; returned cost/usage, otherwise unavailable |
| Chisel geometry preservation | Input planar-z render and returned depth in identical cameras/scale |

Never infer metric accuracy from parser success, confidence alone, or the
retained PCF even-to-odd diagnostic. The static reference participated in
alignment and is model-derived; its errors are consistency evidence. Do not fit
registration to held-out validation pixels.

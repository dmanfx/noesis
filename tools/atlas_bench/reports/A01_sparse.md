# A01 sparse — completed, awaiting operator review

The operator's SMOKE approval is recorded with the note
"Reviewed smoke; proceed with sparse A01". The existing `A01/default` process
was observed running and followed through completion; no duplicate task or
second writer was started.

`images2PosedRGBD` processed the 12 frozen sparse input views at 1280×720 with
`refinePoses: false` and `canonicalizeToFirstFrame: false`. The source prepared
indices are `0, 15, 34, 49, 64, 79, 98, 113, 128, 143, 162, 178`; none is one of
the six prospective Atlas holdouts. The baseline hash remains
`22ea5f3fe87a233f9c444125474e61c44852569e561afec1426b9287aa7bd0a7`.

Operation: `bb50cce43e914929bdc45a5354e690eb`.
Started October 22 at 23:58:57 EDT and finished October 23 at 00:00:06 EDT.
Client elapsed time was 69.70 seconds, including upload, polling, downloads and
parsing. The response exposed no billing/cost field; cost remains unavailable.

All 36 artifacts were downloaded and hashed: 12 RGB images, 12 linear-depth
EXRs and 12 confidence maps, totaling 5,505,306 bytes. All 12 camera bundles
parsed as camera-to-world RUB with XYZW quaternions. RGB, depth and confidence
share the requested 1280×720 grids. OpenCV decoded EXR depth as float32, retaining
non-finite pixels as unknown.

| Returned view | Prepared index | Finite positive depth |
| --- | --- | --- |
| 0 | 0 | 95.85% |
| 1 | 15 | 96.51% |
| 2 | 34 | 77.88% |
| 3 | 49 | 89.01% |
| 4 | 64 | 82.17% |
| 5 | 79 | 81.29% |
| 6 | 98 | 94.21% |
| 7 | 113 | 81.57% |
| 8 | 128 | 40.25% |
| 9 | 143 | 63.99% |
| 10 | 162 | 59.09% |
| 11 | 178 | 85.79% |

Mean coverage is 78.97%; median is 81.87%. The contact sheet was inspected;
returned RGB is upright and room surfaces are visible in the depth previews,
with substantial missing support in some views. These are artifact and validity
observations, not a geometry, scale, calibration or PCF accuracy acceptance.
No registered cloud, independent residuals, held-out generation or Chisel run
was performed in this step.

The local output directory is:

```text
/mnt/noesis_storage/noesis_devel/atlas_bench/runs/A01/default/
  REPORT.md
  review.png
  run.json
  request.json
  input-assets.json
  operation-id.json
  submission.json
  operation.json
  response.json
  cameras.json
  artifacts.json
  parse-report.json
  metrics.json
  artifacts/frames_{0..11}_imageAsset.png
  artifacts/frames_{0..11}_depth_depthAsset.exr
  artifacts/frames_{0..11}_depth_confidenceAsset.png
```

The directory also retains baseline/holdout/config snapshots, upload and
download receipts, poll observations, tool versions and per-command timing.
Completed output was only read during this handoff. PCF/runtime state and
unrelated checkout work were preserved. No code change, commit or push was
needed for this execution.

Exact next step is operator review, with no further task submitted:

```bash
python3 tools/atlas_bench/atlas_bench.py report A01
```

`next` recommends A02 (48 views), gated by `review A01`. If the operator chooses
to proceed after reviewing this sparse result:

```bash
python3 tools/atlas_bench/atlas_bench.py review A01 --note "Reviewed sparse A01; proceed with medium A02"
python3 tools/atlas_bench/atlas_bench.py run A02
python3 tools/atlas_bench/atlas_bench.py report A02
```

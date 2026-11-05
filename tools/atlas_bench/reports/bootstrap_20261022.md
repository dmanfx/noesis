# Issue 37 bootstrap — first live smoke completed

Checked out: `DS9`, based on `b299ec3`. Issue 37 is assigned to `dmanfx`.

Created the [operator/workspace](../README.md), frozen baseline and prospective
holdouts, 11 evaluation experiment folders plus smoke, reusable recorded
upload/submit/poll/download, resumed submission, parser/contact sheet, bounded
cloud derivation, metric hooks, preparation validation and dependency/review
gates. All large media/API output belongs under `NOESIS_ATLAS_BENCH_ROOT`.
This workstation's ignored local configuration selects
`/mnt/noesis_storage/noesis_devel/atlas_bench`.

Baseline: completed Living179 review lane; 179 exact prepared images, 564 named
evidence guards, 199,135 supported PCF surfels. It remains a review candidate.
Six untouched Atlas holdouts were already seen by retained PCF; static evidence
is model-derived, used in alignment and temporally later than the phone video.
No metric accuracy superiority or canonical acceptance is claimed.

Installed Node 24.22.1, World Labs CLI 0.1.0 and the user-local World Labs Codex
skill. After private terminal login, `doctor` passed all checks, including real
asset authentication. `tasks list` contains images2PosedRGBD, atlasGenerate and
atlasChisel. The successful smoke proves images2PosedRGBD access; the other task
permissions remain unexercised.

Validation: 25 focused offline tests passed. They exercised a recorded simulated
upload/submit/poll/download/parser path, refused changed source/request inputs,
missing credentials, prerequisites and reviews, preserved completed output,
reused uncertain-submission keys, excluded holdout/static input, and enforced
Chisel depth/camera contracts. Real OpenCV float EXR/NaN decoding passed.
Agent/docs consistency and whitespace checks passed. One two-frame task ran;
no Axx/Cxx/Rxx experiment or bulk submission was started.

Observed execution on October 22 at 22:42 EDT:

```bash
python3 tools/atlas_bench/atlas_bench.py doctor
python3 tools/atlas_bench/atlas_bench.py smoke
python3 tools/atlas_bench/atlas_bench.py report SMOKE
python3 tools/atlas_bench/atlas_bench.py next
```

The smoke used exactly prepared indices 0 and 2 at 0.499789 and 1.749256 seconds,
with `targetResolution: [1280, 720]`, `refinePoses: false`, and
`canonicalizeToFirstFrame: false`. Client elapsed time was 16.99 seconds.
Operation ID: `2511976f16134f95987f395cad2b8bd9`.

All six returned artifacts downloaded and were hashed. Both returned cameras
are camera-to-world RUB with valid XYZW quaternions. Both RGBs, grayscale
confidence images and single-channel float depth EXRs parse at 1280×720.
OpenCV decodes depth as float32 and preserves NaN. Finite positive depth covers
46.88% and 36.82% of the two images; the rest is non-finite unknown geometry.
The review contact sheet was inspected: RGB is upright and overlapping, while
the depth/confidence show substantial missing support. This is transport/parser
success, not geometric acceptance. Metric scale, quality and alignment remain
unverified. The API returned no billing/cost field, so cost is unavailable.

All private output is saved beneath:

```text
/mnt/noesis_storage/noesis_devel/atlas_bench/runs/SMOKE/default/
  request.json
  input-assets.json
  submission.json
  operation-id.json
  operation.json
  response.json
  cameras.json
  artifacts.json
  parse-report.json
  metrics.json
  run.json
  REPORT.md
  review.png
  uploads/                         # both private input upload receipts
  polls/                           # operation observations and command timing
  artifacts/
    frames_0_imageAsset.png
    frames_0_depth_depthAsset.exr
    frames_0_depth_confidenceAsset.png
    frames_1_imageAsset.png
    frames_1_depth_depthAsset.exr
    frames_1_depth_confidenceAsset.png
```

The root also contains baseline/holdout/experiment snapshots, exact tool
versions, authenticated CLI status, request/idempotency identity and per-command
download receipts/timings. All 45 completed-run files were fingerprinted before
the handoff updates and checked unchanged afterward.

Submissions stopped immediately after this first successful smoke. No review
record was created. `next` identifies A01 and reports `review SMOKE` as its
remaining gate. Exact next step: inspect the local report/contact sheet and
decide whether to unlock the 12-frame sparse run.

After the smoke has been reviewed, run from the repository root:

```bash
python3 tools/atlas_bench/atlas_bench.py review SMOKE --note "Reviewed smoke outputs; proceed with sparse A01"
python3 tools/atlas_bench/atlas_bench.py next
python3 tools/atlas_bench/atlas_bench.py run A01
python3 tools/atlas_bench/atlas_bench.py report A01
```

No existing PCF output, calibration, Scene Prior binding or runtime state was
changed. Concurrent dashboard changes and existing documentation edits were
preserved. No commit or push was made.

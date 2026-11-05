# Atlas / Chisel offline benchmark

Operator entry point for [issue #37](https://github.com/dmanfx/noesis/issues/37).
Run from the repository root on `DS9`:

```bash
python3 tools/atlas_bench/atlas_bench.py doctor
python3 tools/atlas_bench/atlas_bench.py status
python3 tools/atlas_bench/atlas_bench.py smoke
python3 tools/atlas_bench/atlas_bench.py report SMOKE
```

Normal flow: **doctor → status → smoke → review → next → run ID → report ID**.
The bootstrap stops after the first successful smoke. It does not record an
operator review or start A01 automatically.

The [October 22 bootstrap report](reports/bootstrap_20261022.md) records a
successful `SMOKE/default`: two 1280×720 frames, six downloaded/parsed artifacts,
and 16.99 seconds client time. Depth support is incomplete; this is not a quality
acceptance. That completed run is retained. An explicit future rerun needs
`smoke --run-id NAME`; the next current action is to inspect `report SMOKE`.

After inspecting smoke RGB, EXR parsing, cameras and provenance, the operator
can record the review and start exactly one sparse experiment:

```bash
python3 tools/atlas_bench/atlas_bench.py review SMOKE --note "Reviewed the two-frame smoke artifacts; proceed with sparse A01"
python3 tools/atlas_bench/atlas_bench.py next
python3 tools/atlas_bench/atlas_bench.py run A01
python3 tools/atlas_bench/atlas_bench.py report A01
```

The review command is an explicit local operator decision. This first pass
leaves it unset. Later view tiers and generated-context reuse have their own
review gates. `next` recommends work; it never submits anything.

## Local setup and private authentication

Use Node 24+, `npm install -g @worldlabsai/cli`, then
`worldlabs skills install --client codex`. The installed World Labs skill is
user-local, outside this repository. Login in your own terminal:

```bash
worldlabs auth login
worldlabs auth status
worldlabs tasks list
```

Enter the key only at the hidden login prompt. The CLI owns credentials; the
runner never reads the saved key or includes it in requests, logs or reports.
Automation may supply `WLT_API_KEY` privately in its environment. Task discovery
is the SDK's local registry; successful authentication checks asset access.
Task-specific access is only established by an actual task result.

`NOESIS_ATLAS_BENCH_ROOT` selects the private external output directory. The
bootstrap's ignored `data/atlas_bench/operator.json` records its workstation
root and Node/CLI command, so the ordinary commands above need no setup flags.
`--root /external/path` overrides that selection. Without either, the default is
`~/.local/share/noesis/atlas_bench`. Outputs inside the repository are refused.
`NOESIS_STORAGE_ROOT` can relocate the retained source evidence, and
`NOESIS_WORLDLABS_COMMAND` can select an explicitly installed Node/CLI pair.
No API key belongs in local operator configuration.

The runner uses Python 3 with NumPy, Pillow and OpenCV built with OpenEXR.
`doctor` tests actual float EXR decoding, including top-left row order and NaN
preservation, without task submission. It checks all frozen inputs, credentials,
task names, CLI version and output permissions/free space.

## Frozen baseline and authority

[Baseline](manifests/baseline/living179.json),
[holdouts](manifests/holdouts/living179.json), and
[manifest locks](config/baseline.lock.json) freeze the completed, reviewed
Living Room 179-view lane from `pcf_august_more_views_20261009/attempt_64_32`.
Its prepared manifest is SHA256
`5347671f7b77070737acaf4a1aa7d27b8a9362b781576b574a61701f1be38c97`.
The original August video, 1920×1080 prepared images and all provider outputs
are reused in place. Nothing is reconstructed, reprepared or copied into git.

The retained lane is `review_only=true`, `accepted_for_canonical_use=false`.
It is a current reviewed comparison baseline, with 199,135 supported surfels,
not a claimed physical-accuracy winner. Its comparison improved recorded static
coverage but had mixed residual results. Both existing alignment reports passed
15 unchanged checks, preserving the target cloud and fixed-camera pose.
The historical Family Room bundle was not selected because its saved validation
retains a target-frame correction; no such correction is reapplied here.

The manifest preserves original frame IDs/timestamps/hashes, provider outputs,
camera solutions, calibration and static revision, coordinate frames, metric
units, reference transforms and source evidence. Small named metadata/camera
files and all 179 prepared images are hashed. The explicit 537 raw provider
arrays, PCF surfels, original video and static cloud have size/mtime guards and
retained manifest/view-set identities. This does not claim new content hashing
of entire large trees. Any guard mismatch stops a run; do not update locks to
hide a mismatch. Review and freeze a distinct baseline instead.

DA3 raw, conditioned MapAnything and fused PCF do **not** share one implicit
coordinate frame. Conditioned MapAnything already uses backend carried poses;
fused PCF is in a normalized phone-local gauge. Keep the recorded direct PCF
alignment distinct from the evaluator's DA3 camera-correspondence transfer.
Never apply the DA3 world transform directly to fused PCF raw output.
All retained cameras are OpenCV camera-to-world; depth is optical-axis z in
metres. Atlas returns a separate estimated gauge whose scale needs validation.

Six real prepared views, indices `29, 59, 89, 119, 149, 177`, are reserved from
**all Atlas context and appearance uploads**. They were seen by retained PCF
and DA3, so they are prospective Atlas holdouts, not independent PCF holdouts.
The independent static reference was excluded from phone inference but used
for alignment. Its model-derived depth is not surveyed truth; its September
capture postdates the August phone video. IMU and synchronized acquisition
acceptance are unavailable. Preserve those limitations in comparisons.

The [PCF workflow](../../docs/PCF_Workflow.md) and
[phone reconstruction guidance](../mapanything_phone_scan/AGENTS.md) retain
their existing authority. This tool imports no PCF or runtime implementation,
does not stop services, and writes no calibration, catalog, Scene Prior,
tracking/world state, existing scan output or presentation binding.

## Experiments and order

| ID | Purpose | Inputs / dependencies |
| --- | --- | --- |
| SMOKE | Upload, pose, download and parse two nearby real frames | Prepared indices 0 and 2; no refinement |
| A01 | Sparse Atlas-native reconstruction | 12 frozen non-holdout frames; reviewed smoke |
| A02 | Medium Atlas-native reconstruction | 48 frames; completed and reviewed A01 |
| A03 | Full Atlas-native reconstruction | 173 frames, preserving order; reviewed A02 |
| A04 | Untouched held-out view generation | A03; reviewed offline gauge registration and known target poses |
| A05 | Hidden-area completion | Reviewed A04; generated evidence only |
| A06 | Extension using generated views | Reviewed A05 and explicitly chosen generated context |
| C01 | PCF geometry with Chisel and real RGB anchors | Reviewed smoke; rendered metric planar-z and matching cameras |
| C02 | RGB anchor variants | Reviewed C01; hold depth, cameras, prompt and seed fixed |
| C03 | Chisel geometry preservation | C01/C02; reviewed C02; returned depth required |
| R01 | Fixed-camera pose match | Reviewed A04; exact independent calibration target |
| R02 | Bounded local pose search | Reviewed R01; ≤10 cm and ≤2°; diagnostic only |

The sparse cohort is nested inside medium, and medium inside full. Smoke uses
two strongly overlapping frames at 0.499789 and 1.749256 seconds. Every posing
request explicitly uses `targetResolution: [1280, 720]`,
`canonicalizeToFirstFrame: false`, and `refinePoses: false`.

Use `--refine-poses --run-id refinement01` for a **separate controlled posing
variant**. It never changes a default request or a completed run.
Refinement variants do not satisfy dependencies of the unrefined control lane.

```bash
python3 tools/atlas_bench/atlas_bench.py list
python3 tools/atlas_bench/atlas_bench.py run A01 --refine-poses --run-id refinement01
python3 tools/atlas_bench/atlas_bench.py derive A01
```

`derive` is a separate bounded offline cloud step, writing under `derived/`.
It uses returned camera intrinsics, quaternion convention and EXR pixels, retains
the Atlas gauge and records source artifact hashes. It does not register or
replace the PCF reconstruction. A new derivation needs a new `--derived-id`.

Generation and registration have their complete experiment/config/runner
surfaces, with deliberate preparation requirements. The first pass does not
invent an Atlas-to-PCF transform, render unreviewed depth layouts, or submit
generated views. At the relevant phase, materialize and review external assets
under the [preparation contract](config/preparation.schema.json), then:

```bash
python3 tools/atlas_bench/atlas_bench.py prepare C01 --bundle /external/reviewed-chisel-inputs.json
python3 tools/atlas_bench/atlas_bench.py run C01
python3 tools/atlas_bench/atlas_bench.py report C01
```

Each experiment's README specifies the necessary geometry and provenance.
`prepare` checks hashes, cameras, grids, source IDs, holdout/static exclusion,
Chisel depth-only context, metric planar-z declaration and target order. The
same shared helper uploads local assets and constructs wire references; the
operator never needs raw API calls. Registration inputs must document the
offline transform and residuals. Canonical calibration is never a write target.
Rendering, fitting and matched comparison metrics are subsequent offline work;
they are not claimed as implemented/accepted by this smoke bootstrap.

## Output, retries and review

Each experiment defaults to `runs/ID/default` beneath the external root. An
existing directory is refused, including failed work. Explicit `--run-id NAME`
creates a different run; it never overwrites the default. Completed runs are
immutable through this CLI. For interrupted work use:

```bash
python3 tools/atlas_bench/atlas_bench.py resume SMOKE --run-id default
```

Submission uses a persisted idempotency key, submits without waiting, and saves
the operation ID before polling. Resume reuses the exact saved request and key;
a changed request is refused. Local timeout does not cancel server work.
Inspect a failed operation before choosing another paid run.

Each run records baseline/config copies and hashes, input assets/source IDs,
request JSON, idempotency key, submission, operation ID, poll responses, final
response, all returned media, cameras, artifact hashes, parser findings,
parameters/prompts, versions, command timings/errors, and review contact sheet.
Signed URLs and private imagery remain external. Report generation occurs
before completion; `report ID` reads that report and artifact index without
changing the completed directory. `report ID --open` opens the local report.

[Metric hooks](metrics/README.md) explicitly mark missing evaluation inputs and
unmeasured values. Runtime is recorded; API cost is preserved when returned,
otherwise reported unavailable. Missing cost is never treated as zero.
Smoke proves transport, file availability and parsing. It does not prove metric
scale, reconstruction quality, held-out agreement or Chisel geometry fidelity.

Focused offline checks:

```bash
python3 -m pytest tools/atlas_bench/tests -q
./scripts/check_agents_docs_consistency.py
git diff --check
```

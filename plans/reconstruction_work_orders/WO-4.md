# WO-4A/B proposal and implementation record

## Proposed scoring and compatibility contract

The existing raw `confidence` arrays remain unchanged and are the model
outputs. The per-model empirical CDF is retained only as a monotonic,
within-capture `rank_score` in `[0.01, 0.99]`; its manifest semantics will say
that it is an uncalibrated ranking weight, never a probability or calibrated
reliability. The fused evidence weight combines that rank score with internal
multi-view consistency and depth-boundary support. DA3-conditioned MapAnything
and DA3 are marked as correlated evidence, so agreement receives no
independence bonus. The existing selection and quality gates remain in place
while the output records the relationship and source weights, allowing the
matched raw comparison to measure any coverage change.

The public helper names and existing NPZ fields remain readable. The consensus
manifest keeps its v1 envelope and adds explicit semantics fields. New surfel
NPZ fields are `support_view_count` and bounded `support_view_ids`; the old
`points`, `colors`, and `weights` fields remain unchanged. A surfel is admitted
only when it has samples from at least two distinct source views, in addition
to the existing sample and weight gates. Raw per-view arrays retain source
selection and disagreement provenance.

Even-to-odd reprojection is reported as same-inference internal consistency.
It reports all finite residuals, including errors above 2 m, plus comparison
coverage, skipped/evaluated frames, and explicit empty status. An optional
independent-evaluation manifest accepts withheld points in declared backend
world meters with a source identity, provenance, and exact
`frame_identity{frame_id,revision,registration_fingerprint}`. The evaluator
compares that identity with the selected static-target revision and
world-registration/calibration fingerprint before scoring; it rejects
validation points used for alignment and reports candidate accuracy and
coverage against matching points.

## Selected matched real input

The bounded before/after run used the retained 48-view landscape
capture `20260802-162254-8bcc7dd7` and the exact current conditioned candidate:

- MapAnything raw: `$NOESIS_PCF_STORAGE_ROOT/mapanything_prior_variants/20260802-162254-8bcc7dd7/da3_prior_suite_20260809/mapanything_da3_pose_sparse_depth/raw`
- DA3 raw: `$REPO_ROOT/data/mapanything_phone_scans/20260802-162254-8bcc7dd7/da3_outputs/raw`
- Existing candidate manifest: `$NOESIS_PCF_STORAGE_ROOT/mapanything_prior_variants/20260802-162254-8bcc7dd7/da3_prior_suite_20260809/prior_conditioned_consensus_da3_carrier/consensus_manifest.json`
- New output root: `$NOESIS_RECONSTRUCTION_WORK_ROOT/wo4_landscape_consensus_v4`

This preserves the selected DA3 pose-carrier path and uses no GPU or service.
The existing candidate reports 48 views, 88.68% valid fused pixels, 150,901
surfel voxels, and 3.23% large disagreement under the old censored report;
these are baseline values only and will be compared with the uncensored,
distinct-view-support output after implementation.

The machine-local command used the configured `$NOESIS_PCF_STORAGE_ROOT`,
`$NOESIS_RECONSTRUCTION_WORK_ROOT`, and `$REPO_ROOT` roots. Full resolved
paths and logs are retained with the work output rather than embedded in this
portable work order.

The downstream reintegration stage still has an agreement weight bonus setting;
WO-5 must consume this uncalibrated evidence semantics and avoid reintroducing
the same correlated-agreement bonus.

## Status

Implementation is complete in the owned builder/evaluator paths. The CLI still
auto-detects the conditioned relationship from the variant manifest, and the
explicit `--evidence-relationship` option is available for controlled inputs.
The DA3 pose-carrier path remains intact; VI3 is unchanged and deferred.

The builder now writes the accepted `surfel_points.npz` with exact
`support_view_count`, bounded `support_view_ids`, a zero-based raw-view index
semantics field, truncation flags, and `support_class=multi_view_accepted`.
Single-view candidates that pass the prior sample and accumulated-weight gates
are written separately as `surfel_points_single_view.npz` with
`support_class=single_view_withheld`; they never enter the accepted multi-view
set. The input raw model outputs are not modified. Candidates rejected earlier
by the sample or weight gates remain available in those raw inputs for later
review. `surfel_support_diagnostics.png` compares accepted and withheld classes
over shared bounds, and `surfel_support_before_after.png` adds the legacy output
and matched before/after view-support comparison.

## Focused validation

The focused suite passed with 20 tests. Python compilation passed for both
owned implementation files and both focused test files; `git diff --check` is
clean for the owned files and this report. The native DS9 Python environment
does not contain Matplotlib, so the bounded offline report was run with the
configured phone-scan runtime environment. No GPU or inference service was
started.

The matched CPU run used the exact 48-view landscape pair above with
`--pose-carrier da3`; its resolved command and log are kept under the output
root. The old candidate had 150,901 emitted surfels and the historical
agreement report censored errors above 2 m. The new fused pixel validity stayed
at 0.886770715, with the same source-selection counts and the explicit
conditioned relationship `da3_conditioned_mapanything`. The agreement quality
combination is recorded as
`maximum_correlated_source_weight_without_independence_bonus`.

Support-class results from the matched raw comparison are:

| Support view | Legacy output | Legacy raw recheck | New output |
| --- | ---: | ---: | ---: |
| Accepted surfels with 2+ distinct views | 150,901 (mixed) | 93,751 | 83,817 |
| Single-view candidates withheld | not separated | 57,150 | 45,339 |
| Pre-filter voxels | not recorded | 199,629 | 170,081 |
| Pre-filter single-view voxels | not recorded | 105,649 | 85,933 |
| Input selected samples | 1,380,801 | 1,380,801 | 1,249,291 |

The legacy raw recheck applies the new distinct-view calculation to the old
raw output, separating its former one-view false support. The new input sample
count is lower because the removed agreement bonus no longer admits as many
pixels to the surfel input; source selection and raw model outputs remain
unchanged. New accepted support has view-count minimum/median/maximum 2/3/16,
48 source views, and zero rows requiring ID truncation.

The same-bounds X/Z review uses 4 cm cells. The legacy output occupied 25,229
cells; the new accepted layer occupies 20,671 cells (81.93% of the shared
union), while the withheld single-view layer occupies 16,331 (64.73%). Their
union occupies 24,214 cells (95.98% of the shared union); this is a cell union,
not a sum of the two fractions. The exact bounds and counts are in
`$NOESIS_RECONSTRUCTION_WORK_ROOT/wo4_landscape_consensus_v4/surfel_support_comparison.json`.

The fresh evaluator comparison is in
`$NOESIS_RECONSTRUCTION_WORK_ROOT/wo4_landscape_evaluation_v2/evaluation_metrics.json`.
For the new candidate, full-cloud source/target medians are 0.105882 m and
0.134431 m, with 0.846900 source overlap and 0.651276 target overlap within
0.30 m. The matched legacy control is 0.105401 m and 0.137786 m, with 0.848638
source overlap and 0.627720 target overlap. Fixed-camera visible-cloud source
median/overlap changed from 0.078387 m/0.925757 to 0.078907 m/0.924473, so
these measurements show a tradeoff rather than a blanket improvement. The
new point-preserving render sampled 553,390 points, occupied 14,598 observed
5 cm cells, and 13,710 walkable cells; the matched legacy control sampled
549,403 points, occupied 14,420 observed cells, and 13,719 walkable cells.

The builder's internal even-to-odd result is explicitly labeled
`internal_same_inference_even_to_odd_consistency` with `independent=false`.
For consensus it reports median 0.066391 m, p80 0.191500 m, 1,192,821 finite
comparisons, 6,662 errors above 2 m (0.005585 fraction), 0.394334 odd-frame
coverage, and 24/24 odd frames evaluated. The evaluator's independent block
is `not_supplied` for this run; no claim of external accuracy or calibration is
made. The optional independent manifest path requires declared backend-world
metres, source identity, provenance, finite points, exact matching frame
identity, and alignment fitting set to false, then reports source and target
coverage metrics without fitting on those points.

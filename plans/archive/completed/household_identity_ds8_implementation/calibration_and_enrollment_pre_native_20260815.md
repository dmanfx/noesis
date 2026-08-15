# Identity v2 Calibration, Enrollment, and Migration Review

Status: correlation-aware artifact v2 is implemented at the offline/unit
contract level on 2026-07-10. Authority remains blocked until both a qualifying
provenance-locked benchmark corpus and independent local household verification
evidence exist for the active model profile.

## Authority doctrine

`NOESIS_IDENTITY_V2_MODE=authoritative` accepts only a strict
`noesis.identity.open_set_calibration` v2 artifact. Version 1 artifacts are
ambiguous about correlation and fail closed. Version 2 binds the actual ReID
engine bytes, output layer, embedding dimension, a deterministic digest of the
active YAML+nvinfer crop/preprocessing/tensor/normalization/scoring semantics,
the evidence-unit policy, and two separately hashed datasets:

- `benchmark`: provenance-locked open-set evidence. Its subject-disjoint
  holdout supplies the large-N generic FAR and misidentification claim.
- `household`: score evidence captured in the installed cameras/home. It is a
  domain-shift verification gate, not a replacement for the benchmark.

The benchmark fit establishes the logistic mapping and baseline rejection
policy. Household training may keep or raise appearance, quality, confidence,
ambiguity, and unknown-preference gates; it may never lower them or alter the
benchmark calibration/resident prior. The final policy must then pass both
untouched holdouts. Product ceilings cannot be loosened through CLI arguments
or artifact fields.

The fitted logistic value is a balanced monotonic match score, not a posterior
probability for deployment prevalence. The artifact authorizes only the
open-set scorer policy. Whole-frame assignment, overlap permits, DS8/DS9
adapter parity, and occupied-scene behavior retain separate replay/live gates.
The runtime enforces that separation: a valid scorer artifact alone cannot
start public authoritative mode.

An arbitrary policy JSON is not a calibration artifact. Shadow remains the
default until the household has real held-out evidence.

## Collect score-only shadow evidence

Set these before starting a non-live validation runtime or a coordinated future
live restart:

```bash
export NOESIS_IDENTITY_V2_MODE=shadow
export NOESIS_IDENTITY_V2_EVIDENCE_PATH="$HOME/.noesis/household/calibration/shadow.jsonl"
export NOESIS_IDENTITY_V2_EVIDENCE_SESSION_ID="explicit-independent-session-id"
export NOESIS_IDENTITY_V2_EVIDENCE_SOURCE=shadow
export NOESIS_IDENTITY_V2_EVIDENCE_RUNTIME=ds8
```

The capture file and its parent are private state: the parent must be mode
`0700` and any existing file must already be mode `0600`, owner-owned,
single-link, and regular. The recorder never chmods an existing path. Its
rolling retention defaults to 100,000 complete records, 256 MiB, and 30 days;
`NOESIS_IDENTITY_V2_EVIDENCE_MAX_RECORDS`,
`NOESIS_IDENTITY_V2_EVIDENCE_MAX_BYTES`, and
`NOESIS_IDENTITY_V2_EVIDENCE_MAX_AGE_S` may make those bounds smaller (or the
age window up to one year) but cannot exceed the product record/byte ceilings.
Pruning retains only a contiguous prefix-trimmed window of complete records.
Evidence contract v2 links every record with a monotonic sequence and previous
event digest; an owner-only chain checkpoint binds the retained head, tail,
count, and bytes. Interior deletion, unexplained prefix/suffix loss, a missing
final newline, reordered timestamp, duplicate, symlink, hard link, or digest
mismatch therefore fails loudly. The checkpoint is required for labeling and
calibration; it is not an optional recovery fallback.

Replay runs use `EVIDENCE_SOURCE=replay` and `EVIDENCE_RUNTIME=replay`. Each
record contains quality, gallery candidate scores, the pre-prior winner, final
shadow outcome, and exact model/observation provenance. Every hash-chained row
also carries the semantic-profile SHA-256 derived by the active runtime, so a
label file cannot invent preprocessing provenance after capture. It never
contains a query embedding or gallery vector. The owner browser API exposes
capture status and counts only; it does not serve the private evidence file.

Validate the exact append-only file:

```bash
python3 scripts/identity_v2_calibrate.py validate-evidence \
  --evidence "$HOME/.noesis/household/calibration/shadow.jsonl"
```

## Label and split

Create a strict `noesis.identity.evidence_labels` v2 JSON document. Every
evidence event needs exactly one label with:

- `event_id`: exact evidence event SHA-256;
- `partition`: `train` or `holdout`;
- `truth_kind`: `resident`, `visitor`, or `unknown`;
- `truth_subject_id`: exact candidate subject for known truth, otherwise null;
- `truth_person_key`: private stable person pseudonym, including unknown people;
- `encounter_id`: one continuous physical encounter, never a frame or a 5-second
  window.

The label set also declares `evidence_stratum` (`benchmark` or `household`) and
immutable provenance: source name/revision, source-manifest SHA-256,
runtime-derived `model_semantic_profile_sha256`, and labeling-protocol
revision. The semantic digest must equal the value printed by
`validate-evidence`. `population_basis` attests `licensed_real_people` for the
benchmark or `owner_consented_household_people` for local capture. Synthetic,
generated, or unidentified-person truth cannot carry authority.

Benchmark events must be recorded as `source=replay`; shadow/live rows cannot be
rebranded as generic benchmark authority. Household rows may be shadow capture
or an exact replay of the separately manifested home capture.

Every split is encounter-, capture-session-, and runtime-run-disjoint.
`subject_disjoint` additionally prevents every `truth_person_key`—including
unknown people—from crossing. Benchmark authority requires subject-disjoint;
household verification may use session-disjoint so the same residents can be
tested on different visits without contaminating frames or sessions. Household
unknown people remain person-disjoint across train/holdout.

Every encounter needs at least one deployment-representative challenge row:
two or more hard-allowed candidates with non-empty galleries, at least one
resident, at least one visitor, and at least one impostor. Training rows must
also include the genuine candidate. A missing genuine candidate in holdout is
kept and counted as a recall failure. Hard-blocked and zero-exemplar candidates
do not enter the fit or gallery authority envelope.

Raw observations are deterministically grouped by capture session, runtime
run, source, camera, tracker, and fixed 5-second bucket. A unit has at most 300
observations. Logistic fitting uses one center-time representative per selected
unit, at most eight temporally spread units per encounter, and equal encounter
mass before positive/negative class balancing. All observations remain in
evaluation. Multiple windows from one encounter never become multiple
confidence trials.

Authority first computes encounter-worst-case outcomes: any false accept makes
an unknown encounter fail; any wrong accepted identity makes a known encounter
misidentified; otherwise any rejected observation makes it a false-reject
encounter. Benchmark confidence then takes the worst outcome for each private
truth person across all of that person's encounters. Its subject-disjoint,
challenge-covered holdout needs at least 300 resident people and 300 unknown
people. With zero person failures, each one-sided 95% Clopper-Pearson safety
upper bound is about 0.9936%, below the 1% product ceiling. Benchmark train
needs 50 challenge-covered resident people and 50 unknown people for fitting
and policy selection but makes no large-N confidence claim. More encounters or
tracklets for one person never inflate these denominators.

The feasible local gate needs 10 known/10 unknown household train encounters
and 20 known/20 unknown household holdout encounters. Both household partitions
must have zero false-accept and zero misidentification encounters, FRR at most
35%, and zero harmful or rejection-rescuing prior changes. Its necessarily
wider confidence bounds are reported honestly and are not mislabeled as the
generic 1% claim. Known calibration truth is resident in both strata; visitor
continuity is validated separately and cannot substitute for family recall.

Build and validate the deterministic score-only dataset:

```bash
python3 scripts/identity_v2_calibrate.py build-dataset \
  --evidence "$HOME/.noesis/household/calibration/benchmark.jsonl" \
  --labels "$HOME/.noesis/household/calibration/benchmark-labels.json" \
  --output "$HOME/.noesis/household/calibration/benchmark-dataset.json"

python3 scripts/identity_v2_calibrate.py build-dataset \
  --evidence "$HOME/.noesis/household/calibration/shadow.jsonl" \
  --labels "$HOME/.noesis/household/calibration/household-labels.json" \
  --output "$HOME/.noesis/household/calibration/household-dataset.json"

python3 scripts/identity_v2_calibrate.py validate-dataset \
  --dataset "$HOME/.noesis/household/calibration/benchmark-dataset.json"

python3 scripts/identity_v2_calibrate.py validate-dataset \
  --dataset "$HOME/.noesis/household/calibration/household-dataset.json"
```

Validation prints a biometric-free owner review with observation, evidence-unit,
encounter, session, run, and person counts for each partition plus the zero-error
95% upper bound its challenge-covered person count could support.

## Calibrate and gate

The tool fits the monotonic logistic score mapping from encounter-balanced
benchmark train representatives, chooses the benchmark policy, then searches
only monotonic household tightening. The artifact separately reports benchmark
train/holdout and household train/holdout metrics, their exact bounds, and
beneficial/harmful/rejection-rescuing resident-prior changes. The resident prior
remains post-gate assignment utility; it cannot make an otherwise rejected
visitor look resident.

The artifact records a conservative runtime gallery envelope: the smallest
resident, visitor, total-candidate, and per-candidate exemplar coverage across
every qualifying row in both strata. One unusually large gallery cannot raise
authority. Startup rejects an existing gallery outside the envelope, and
enrollment or visitor growth is rejected before store mutation.

Use an explicit timestamp and source revision so repeated runs with identical
inputs and arguments produce identical output:

```bash
python3 scripts/identity_v2_calibrate.py calibrate \
  --benchmark-dataset "$HOME/.noesis/household/calibration/benchmark-dataset.json" \
  --household-dataset "$HOME/.noesis/household/calibration/household-dataset.json" \
  --output "$HOME/.noesis/household/calibration/open-set-v2.json" \
  --generated-at-us 1783728000000000 \
  --generator-revision '<reviewed-source-revision>'
```

Do not set `NOESIS_IDENTITY_V2_SCORING_ARTIFACT` for authoritative mode until
the generated benchmark provenance, both holdout metric blocks, correlation
units, and labeled examples have been reviewed by the owner. The current
repository contains fixture evidence only; no real benchmark or home capture is
present, so authority remains blocked.

Cutover also requires two independently reviewed pins:

```bash
export NOESIS_IDENTITY_V2_SCORING_ARTIFACT='<owner-private-open-set-v2.json>'
export NOESIS_IDENTITY_V2_SCORING_ARTIFACT_SHA256='<sha256-of-exact-artifact-bytes>'
export NOESIS_IDENTITY_V2_MODEL_SEMANTIC_PROFILE_SHA256='<validate-evidence-profile-sha256>'
```

Startup hashes one securely read artifact snapshot. Both the semantic pin and
artifact field must equal the digest newly derived from the active engine,
model YAML, parsed nvinfer configuration, the actually loaded
`noesis_reid_meta_ext` binary, and the Python normalization/gallery/scoring
implementations. An arbitrary environment label cannot authorize changed crop,
normalization, extraction, tensor, or model semantics.

Calibration evidence, labels, datasets, and artifacts must be owner-only mode
`0600` files. Create their containing calibration directory with mode `0700`;
the tools reject unsafe or linked paths instead of changing their permissions.

## Public authority cutover gate

After scorer calibration passes, keep `NOESIS_IDENTITY_V2_MODE=shadow` until two
separate reports exist for the exact intended runtime:

1. a whole-frame coordinator replay covering one-to-one assignment, explicit
   unknowns, overlap permits, continuity, and exact downstream OSD joins; and
2. an occupied-scene DS8 or DS9 report covering resident/visitor/unknown
   behavior, camera handoff, gallery load, and runtime performance.

The owner-reviewed gate file uses
`noesis.identity.authority_cutover` v1 (generated schema:
`contracts/schema/identity_authority_cutover.schema.json`). It binds the active
runtime (`ds8` or `ds9`), engine/layer/dimension, semantic digest, exact scorer
artifact SHA-256, executable authority-runtime profile, topology bytes, sorted
camera IDs, and two distinct report paths, sizes, SHA-256 values, revisions,
completion times, and literal `passed=true` results. Report paths may be
absolute or relative to the gate file. The gate file and both reports must be
owner-only regular single-link files; startup reads and hashes every file.

Only after reviewing those exact bytes set all four independent pins:

```bash
export NOESIS_IDENTITY_V2_MODE=authoritative
export NOESIS_IDENTITY_V2_SCORING_ARTIFACT='<owner-private-open-set-v2.json>'
export NOESIS_IDENTITY_V2_SCORING_ARTIFACT_SHA256='<sha256-of-exact-scorer-artifact>'
export NOESIS_IDENTITY_V2_MODEL_SEMANTIC_PROFILE_SHA256='<active-profile-sha256>'
export NOESIS_IDENTITY_V2_AUTHORITY_CUTOVER_ARTIFACT='<owner-private-cutover-v1.json>'
export NOESIS_IDENTITY_V2_AUTHORITY_CUTOVER_ARTIFACT_SHA256='<sha256-of-exact-cutover-artifact>'
```

The scorer pin, cutover pin, runtime code profile, topology, camera set, and both
report files are independently rechecked on every startup. Missing, stale,
linked, shared-mode, altered, cross-runtime, or scorer-mismatched evidence fails
before the identity store or public runtime is opened. DS8 evidence cannot
authorize DS9, and vice versa.

## Enrollment product

Menon exposes Household Identity only to the authenticated owner. A live
tracking row contributes the exact short-lived `identity_observation_key` and
no biometric vector. The UI first creates a proposal, shows its action, quality,
and expiry, then requests a Menon action-bound confirmation and confirms the
same key plus evidence digest. HTTP 409/410 forces a fresh observation; stale or
consumed evidence is never silently retried under another person.

## Legacy migration review

Generate a private, biometric-free review without applying migration:

```bash
python3 scripts/identity_v2_migrate.py dry-run \
  --source-root '<legacy-state-directory>' \
  --model-fingerprint '<active-engine-sha256>' \
  --embedding-dim 256 \
  --assert-source-model-fingerprint '<verified-legacy-model-sha256>' \
  --assert-source-embedding-dim 256 \
  --report-output "$HOME/.noesis/household/migration-review.json"
```

Set `NOESIS_IDENTITY_V2_MIGRATION_REVIEW_REPORT` to that reviewed file for a
future coordinated runtime start. Menon shows duplicate normalized names,
blocked records, and empty galleries. It intentionally has no migration apply
route and never auto-resolves a duplicate.

## Post-resolution video labels

Authoritative video labels are restamped through an explicit Service Maker
`BatchMetadataOperator` probe at the tiler sink. That point is downstream of
whole-frame resolution but upstream of source-ID collapse and `nvdsosd`. The
operator re-acquires each object wrapper once, joins only an exact
camera/frame/tracker decision from the bounded service cache, and retains no SDK
wrapper. Resident labels are `#<sid> <name>`, visitors are `#<sid>`, and every
miss, mismatch, unknown, or provisional result is `#XX`. The cache defaults to
eight frames and is bounded to `NOESIS_IDENTITY_V2_OSD_CACHE_FRAMES=2..64`.

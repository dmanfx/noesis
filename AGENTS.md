# AGENTS.md (Root) – Guidance for Codex Agents

This repository is centered on the DeepStream 8 stack:

- **DS8 (canonical):** `noesis/` tree (`ds8_runtime.py`, `pipelines/`, `server/`, `telemetry/`, `metadata/`, `config/`) built on DeepStream 8 Service Maker and Flow APIs.
- **Pre-DS8 artifacts (deprecated):** retained only for historical context or explicit one-off maintenance.

## Policy precedence

- This file is the canonical repo-wide policy baseline.
- A nearer `AGENTS.md` in a child directory may add narrower rules and overrides this file for that subtree.
- Files under `docs/history/` are archival and non-normative for active implementation work.

When you (the agent) work in this repo:

1. **Respect the deprecated-stack vs DS8 split**
   - Do not silently route DS8 failures through removed pre-DS8 runtime paths. If DS8 code cannot be made to work, fail loudly in logs/docs and stop.
   - Only modify deprecated pre-DS8 artifacts if the user explicitly asks for those changes.
   - Prefer implementing new functionality in the DS8 stack under `noesis/`.
   - Launch and validate runtime behavior through `noesis/ds8_runtime.py` for active work.

2. **No fallbacks without explicit user approval**
   - Do not introduce, enable, or rely on fallback paths, degraded modes, substitute workflows, legacy routes, or "temporary" alternates unless the user explicitly asks for that fallback or agrees after you surface the blocker.
   - If the canonical path fails, report the failure clearly, explain what is blocked, and stop instead of masking the problem with a fallback.
   - Treat hidden fallbacks as bugs because they can conceal the real defect or misconfiguration.

3. **Follow the DS8 plan and checklists**
   - Before changing DS8-related code, read:
     - `plans/DS8/ds8_master_work_orders.md`
     - The relevant checklist(s) under `plans/DS8/` (e.g., `ds8_migration_checklist_ds8_pipeline.md`, `ds8_migration_checklist_hooks.md`).
   - Treat these documents as the authoritative description of what to do and in what order. If code and docs disagree, surface the discrepancy in a doc update rather than guessing.

4. **No guessing of DeepStream/Service Maker APIs**
   - For any call into DeepStream 8 or Service Maker (e.g., `pyservicemaker`, `pyds`, `nvinfer`, `nvdsanalytics`):
     - Use only methods, properties, and config keys that you can verify in the official docs or the installed Python modules.
     - Do **not** invent new method names or config keys.
   - The main DS8 docs to consult (non-exhaustive):
     - `docs/DS8_README_FOR_AGENTS.md` in this repo (local orientation + links).
     - Service Maker for Python docs (Pipeline APIs, Flow APIs, Advanced Features).
     - Inference Builder docs.
     - DS8 analytics / nvdsanalytics YAML docs.

5. **GPU-first policy for DS8**
   - Keep decode, pre-processing, inference, tracking, analytics, tiling, and OSD on the GPU (NVMM) wherever DeepStream supports it.
   - Convert to CPU only at the very edge when required (e.g., final JPEG encoding if not handled by `nvjpegenc`, BEV composition, or explicit ReID crops).
   - Do not add new appsink/CPU-based branches to the DS8 path; those patterns belong only to deprecated pre-DS8 code.

6. **Testing and validation**
   - When you make DS8 changes, use `docs/DS8_testing_guide.md` as your reference for how to validate behavior (unit tests, functional tests, and DS8 regression checks).
   - Prefer small, focused validations (e.g., “does analytics ROI reload work?”) rather than only end-to-end tests.
   - For interface/contract changes, also consult:
     - `docs/DS8_api_contracts_ws.md` (WebSocket message contracts).
     - `docs/DS8_api_contracts_rest.md` (REST contracts, including ReID alias endpoints from `noesis/server/reid_api.py`).
     - `docs/DS8_metadata_contracts.md` (metadata/user-meta schemas).

7. **Progress documentation**
   - When completing work defined in `plans/DS8/*.md`:
     - Update the relevant checklist item from `[ ]` to `[x]`.
     - Add a short, dated note under that item describing what you changed and how you validated it.
   - Do this **often**: after each logically separate task or small cluster of related edits, so the plans reflect near real-time progress.
   - Do **not** add progress notes into code comments unless the user explicitly asks; keep them in the planning docs.

8. **Design decisions**
   - When you make a non-trivial design choice for DS8 (API shape, data schema, etc.), capture it in `plans/DS8/ds8_design_decisions.md` with a short rationale and any doc references you relied on.

9. **Docs and instruction hygiene**
   - When editing docs, AGENTS files, plans, or prompts, verify any referenced local file paths and commands still exist in this workspace before finalizing.
   - After changing `AGENTS.md` or files under `docs/`, run `./scripts/check_agents_docs_consistency.py` and fix failures before closing the task.
   - In user-facing summaries and explanations, prefer plain semantic descriptions of behavior, variables, and outcomes. Do not default to file/line citations unless the user explicitly asks for code references.

10. **Native extension discipline**
   - If you change V3DT/pose metadata plumbing or contracts (`noesis/pipelines/hooks.py`, `noesis/metadata/*`, `native/*`), rebuild the relevant native extension(s) and run focused smoke tests (for example `scripts/sv3dt_meta_smoke_test.py`) to confirm metadata extraction still works.

11. **Portable path requirement**
   - Do not add new hardcoded machine-local absolute paths (for example `/home/...`) in runtime code, docs, or scripts.
   - Prefer `REPO_ROOT`, config paths, or environment variables for path resolution.

These rules are meant to make your work predictable and auditable for future Codex runs.

## Development and Testing Policy

### Default operating mode

Work in an operational development fast path unless the task explicitly requests production release hardening, formal release validation, security-critical assurance, or another unusually high-assurance workflow.

The primary objective is to produce working, maintainable application behavior and make measurable progress toward the requested outcome.

Do not allow testing, packaging, provenance, release mechanics, or procedural infrastructure to become the main body of work unless they are directly required by the task.

### Implementation priorities

Use the following order of priority:

1. Restore or implement the requested application behavior.
2. Resolve actual runtime failures, blockers, regressions, and incorrect outputs.
3. Run focused tests covering the changed code and its immediate dependencies.
4. Run relevant integration tests when the change crosses component boundaries.
5. Add a small number of high-value edge or adversarial tests for realistic failure modes.
6. Broaden testing only when the observed risk justifies it.

Prefer direct, understandable solutions over generalized frameworks, new orchestration systems, release machinery, or abstract infrastructure.

Do not redesign unrelated systems while solving a localized problem.

### Testing scope

Testing must be proportional to the change.

Default to the narrowest test set that provides meaningful confidence:

- Run tests associated with the modified component.
- Run tests for directly affected contracts, APIs, data formats, and downstream consumers.
- Run an integration or smoke test for the actual application path when practical.
- Add edge-case testing when the edge case is realistic, consequential, or exposed by the current work.

Do not automatically run the complete test suite after every change.

Do not repeatedly rerun unchanged test suites unless:

- relevant code changed;
- a prior failure may have been affected;
- a shared contract or dependency changed;
- the task explicitly requires a full validation pass; or
- the agent is preparing a user-requested final release candidate.

A typo, documentation correction, comment change, formatting change, or modification outside the application execution path must not trigger a full application, packaging, release, or artifact-validation sequence.

### Escalation ladder

Use this testing escalation sequence:

1. Static checks or syntax validation for the changed files.
2. Targeted unit tests.
3. Targeted component tests.
4. Relevant integration or application smoke tests.
5. Broader regression tests only when justified by the affected dependency surface.
6. Full-suite or release validation only when explicitly requested or clearly required by a high-risk cross-system change.

Stop escalating once the changed behavior is validated and no evidence suggests a wider regression.

### Edge and adversarial testing

Edge-case and adversarial testing are encouraged when they protect real application behavior.

Keep them focused on plausible and meaningful failures such as:

- malformed or missing runtime data;
- stale or incompatible contracts;
- process interruption;
- unavailable models or hardware resources;
- incorrect coordinate, timestamp, or identity data;
- realistic lifecycle failures;
- actual bugs discovered during implementation.

Do not create exhaustive adversarial matrices, large synthetic fault frameworks, formal verification systems, or broad procedural test harnesses unless specifically requested.

A few high-value failure tests are preferable to hundreds of low-probability procedural cases.

### Release and provenance mechanics

Normal development work must not introduce or repeatedly execute elaborate release-sequence mechanics.

Unless explicitly requested, do not:

- build immutable release-publication systems;
- create cryptographic provenance or evidence sealing;
- construct multi-stage authorization or promotion ceremonies;
- repeatedly rebuild release bundles;
- enforce formal publication gates;
- create independent reviewer workflows;
- perform release replay validation;
- generate extensive compliance-style evidence;
- turn ordinary local development into a production cutover simulation.

Existing release or provenance tooling may remain intact, but it should not be invoked during ordinary development unless directly relevant to the current task.

### Filesystem and staging constraints

Avoid recursive traversal of large staging, artifact, cache, model, dataset, recording, build-output, or generated-file directories unless the task specifically requires inspecting them.

Do not repeatedly scan, hash, copy, validate, inventory, or package hundreds of gigabytes of unchanged files.

Use explicit paths, manifests, changed-file lists, timestamps, targeted globs, or known artifact identifiers.

Exclude large generated and staging directories from broad repository scans wherever possible.

Before running an operation expected to process a large portion of the filesystem, determine whether a smaller targeted operation can answer the same question.

### Reuse of prior results

Do not repeat expensive work whose inputs have not changed.

Reuse valid results from previous tests, builds, calibrations, and inspections when:

- the relevant inputs are unchanged;
- the environment has not materially changed;
- the result remains applicable to the current decision.

Rerun only the portion invalidated by new changes.

### Infrastructure restraint

Do not create new infrastructure solely to make the development process more formal.

New abstractions, frameworks, validators, lifecycle managers, artifact formats, and orchestration layers require a direct practical need.

Before adding procedural infrastructure, ask:

- Does this solve a current application failure or blocker?
- Is it required for the requested feature?
- Will it be used during normal operation?
- Is there a simpler existing mechanism?
- Is the implementation cost proportional to the risk?

If the answer is no, defer it.

### Completion criteria

A normal development task is complete when:

- the requested behavior works;
- actual blockers are resolved;
- relevant targeted tests pass;
- the affected integration path has been exercised where practical;
- known limitations are documented;
- no material regression is evident.

A normal development task does not require proving every possible failure path, creating a sealed release artifact, or executing a complete production cutover process.

### Capability-scoped production promotion

Production promotion is capability-scoped. Select checks from the behavior and
boundaries actually changed, not from the perceived importance of the project
or from broad filename categories.

Use these three levels:

1. **Standard promotion (default):** localized application changes. Run
   focused static, unit, component, and direct-consumer checks; build the
   affected deliverable once; activate through the existing bounded
   selector/readiness/rollback path; and exercise the changed behavior live.
   Target completion in 30 minutes or less.
2. **Scoped integration promotion:** changes that require real integration but
   affect a bounded capability such as network exposure, TLS, systemd
   lifecycle, frontend delivery, one REST/WebSocket contract, one state
   migration, or one model/native lane. Test that capability and its direct
   consumers only. Network and systemd changes do not by themselves require
   perception-quality suites, occupied-scene evidence, every application
   surface, or sealed full-runtime ceremonies.
3. **Full-system assurance:** use only when correctness genuinely depends on
   running the complete pipeline and application, when a change crosses core
   inference/tracking/world/Menon semantics, when the impact cannot be bounded
   after inspection, for a named release candidate, or when the user
   explicitly requests exhaustive validation.

Map changes to capabilities such as `network`, `service_lifecycle`, `gateway`,
`frontend`, `depth`, `tracking`, `identity`, `world`, `state`, and
`deployment`. Run the smallest transitive set of producer, contract, consumer,
and live-smoke gates that proves those capabilities. A path name alone must
not escalate a change to full-system assurance.

Standard and scoped promotions must:

- stage from an explicit changed-file or release-delta manifest rather than
  treating every dirty-worktree file as part of the release;
- reuse passed tests, builds, calibrations, and staged artifacts when their
  input digests are unchanged;
- rerun only a failed phase and the downstream phases invalidated by the fix;
- keep selector validation, required asset/config checks, bounded
  stop/start/readiness, and automatic rollback;
- classify unrelated or unavailable evidence as a warning or deferred check,
  not a blocker, unless it is a direct dependency of the changed capability;
- stop and report the exact blocker if the practical promotion cannot finish
  within the 30-minute target instead of silently expanding into
  full-system assurance.

Full release hardening, exhaustive adversarial validation, complete-suite
repetition, long-duration soak testing, sealed evidence replay, and formal
cutover rehearsals remain available, but they are not the default consequence
of touching a model, native component, network rule, systemd unit, state file,
or security-related file. Escalate only to the assurance needed by the actual
behavioral impact.

## Noesis and DS9 Development Fast Path

During active Noesis and DS9 development, prioritize a runnable multi-camera application path over release-process completeness.

The preferred validation loop is:

1. Reproduce the actual failure or incomplete behavior.
2. Modify the smallest relevant implementation surface.
3. Run targeted tests for the affected runtime, contract, model, calibration, or visualization component.
4. Exercise the relevant live or recorded pipeline path.
5. Inspect concrete outputs such as detections, tracks, depth, world coordinates, identities, latency, GPU usage, and Menon visualization.
6. Continue iterating against observed application failures.

Do not restart the complete DS9 release, artifact-publication, provenance, packaging, or cutover sequence after an unrelated or low-risk change.

Do not scan entire staging, recording, model, calibration, or artifact directories when known paths or manifests are available.

Depth calibration work should validate the specific cameras, calibration inputs, transformations, and runtime consumer paths involved. It should not automatically trigger repository-wide release validation.

Contract changes should test affected producers and consumers. They do not require testing every unrelated runtime or historical artifact.

A complete DS8/DS9 parity suite, full packaging pass, long-duration soak, or formal cutover rehearsal should be run only when explicitly requested or when preparing a named release candidate.

The development goal is working perception, tracking, identity, depth, world fusion, and visualization. Procedural confidence mechanisms support that goal; they must not displace it.

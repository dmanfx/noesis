# Noesis Commit, PR, and Release Notes Conventions
_Status: current as of 2026-02-02._

This repo uses a pragmatic blend of Conventional Commits, Keep a Changelog, and Semantic Versioning.

## Versioning
- Use Semantic Versioning: `MAJOR.MINOR.PATCH`.
- Bumps:
  - MAJOR: breaking API/behavior changes.
  - MINOR: new features, non-breaking refactors/UX.
  - PATCH: fixes, docs, small perf, chore.

## Commit Messages (Conventional Commits)
Format:
```
<type>(<scope>): <short summary>

[optional body]

[optional footer]
```
- Types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`, `chore`, `revert`.
- Scope: subsystem or area, e.g. `pipeline`, `ui`, `telemetry`, `docs`, `models`.
- Summary: imperative, lowercase first word, <= 72 chars.
- Breaking: add a footer `BREAKING CHANGE: <description>` (or `!` after type/scope, e.g. `feat(pipeline)!:`).
- Issue links: `Refs: #123` or `Fixes: #123` in footer.

Examples:
- `feat(pipeline): add batched multistream support`
- `fix(telemetry): correct fps duplication in drawer`
- `docs: add DeepStream pipeline map`
- `refactor(ui): simplify telemetry drawer layout`
- `perf(encoding): enable dynamic jpeg quality in nvjpegenc`

## Pull Request Titles
- Mirror commit style: `<type>(<scope>): <summary>`.
- Prefer one PR per logical change; draft if WIP.
- Add a clear description including motivation, approach, screenshots (if UI), and test plan.

Examples:
- `feat(ui): new Electron interface + telemetry drawer`
- `refactor(pipeline): switch to batched multistream`

## Release Notes (Keep a Changelog)
Group changes under consistent sections:
- Added, Changed, Fixed, Removed, Performance, UI/UX, Docs, Models, Pipeline, Security, Breaking Changes, Migration Notes.

Template:
```
## [vX.Y.Z] - YYYY-MM-DD

### Added
- ...

### Changed
- ...

### Fixed
- ...

### Performance
- ...

### UI/UX
- ...

### Docs
- ...

### Models
- ...

### Pipeline
- ...

### Breaking Changes
- ...

### Migration Notes
- ...
```

## Labels (optional for automation)
Map labels to sections to drive auto-release notes (e.g., Release Drafter):
- `type: feature` → Added
- `type: fix` → Fixed
- `type: refactor` → Changed
- `type: perf` → Performance
- `type: docs` → Docs
- `area: ui` → UI/UX
- `area: pipeline` → Pipeline
- `area: models` → Models
- `breaking` → Breaking Changes

Create labels as needed; keep them simple and stable.

## Practical Tips
- Keep PRs small and focused; link issues.
- Use checklists in PRs: tests, docs, screenshots, migration notes.
- Squash merge to keep a clean history with a well-written title.

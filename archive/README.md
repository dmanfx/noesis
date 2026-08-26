# External archive locators

This directory contains only repository-tracked locators for material removed
from the live checkout. Historical documentation belongs under `docs/history/`
or `plans/archive/`; executable legacy source does not remain here.

Resolve each locator's `archive_relative_path` beneath
`NOESIS_WORKSPACE_PRESERVATION_ROOT`. On this workstation the preservation root
is `/mnt/noesis_storage/noesis-workspace-preservation`, but that machine-local
path is not repository authority.

The JSON files under `manifests/` record the archive digest, capture scope, and
verification results. Verify the archive SHA-256 before listing or extracting
it, and extract into a new empty directory for review rather than restoring it
over the repository.

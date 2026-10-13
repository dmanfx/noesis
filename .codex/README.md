# Project agent profiles

The project configuration selects the recorded Sol/Luna model and effort policy.
Alternate effort profiles refer to the same durable role; they do not authorize
concurrent owners or expand application permissions.

Profiles resolve the existing private coordination root through the local
`.noesis-team.json` locator. Copy [the locator template](../.noesis-team.example.json)
to that ignored filename and fill in the actual installed root, host and executor.
Verify those values and the repository root against the private `root-map.json`,
then verify the current role generation, session, task and ownership in its roster.
The template is not an installation or a fallback team root. Missing or mismatched
bindings permit safe reads only.

The private root, locator, credentials, role records and evidence remain local.
Committing these profiles neither activates a team nor changes the existing
coordination records. Root [repository policy](../AGENTS.md) still governs direct
development, validation, branch and publication authority.

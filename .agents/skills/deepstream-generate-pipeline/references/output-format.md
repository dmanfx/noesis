# Pipeline delivery examples

Use the outcome and evidence that apply to the actual task. These examples
illustrate status wording, not required headings, badges, or a fixed response
shape. Never copy an example's check results into a different run.

## Artifact exercised successfully

“Saved `scripts/demo.sh`. Static checks passed and a bounded run with the requested
recording produced the expected encoded output. The live RTSP source was not
exercised.” Link the actual artifact and relevant output evidence. Only describe
tracking, identity, or performance as verified when those consumers were checked.

## Static checks passed, runtime incomplete

“Created the three-source graph. Syntax, element, and property checks passed;
live parse was skipped for named mux pads. End-to-end operation remains
unverified.” Include the command or artifact if useful and identify the missing
runtime evidence. Retrieval confidence may describe example selection, never
replace a runtime check.

## Failed validation or missing prerequisites

“Draft created; validation still fails after two fixes: `<actual error>`.
`<specific check>` passed; `<specific check>` was skipped. The graph is incomplete.”
If an actual input/config is missing, identify that path and the affected check.
Do not replace the source silently or label the result validated/ready to run.

## Command and file conventions

- Quote shell paths and values. Readable multiline commands and explicit
  environment variables are allowed; explain required variables.
- Follow repository portability rules for checked-in scripts and configs.
- Create a file when the request calls for an artifact, and link it in the reply.
- Do not add a mandatory table, save-file question, or refinement interview to a
  result that is already complete.

# Plans archive

This directory contains completed, superseded, or experimental work orders.
They preserve the reasoning and chronology of DS7, DS8, DS9.0, container-era
DS9.1, zero-copy, tracker, and prototype work, but they are not current
implementation instructions.

- `ds8/`: curated tracked DS8 work-order and V3DT history.
- `completed/`: completed cross-version feature programs, including the
  original occupancy and WebRTC rollout plans.
- `migrations/`: detailed migration worklogs whose remaining actions were
  distilled into current plans. `migrations/ds9_native/` contains the completed
  DS9.0/container-to-native-DS9.1 preparation and upgrade records formerly
  stored beside current DS9 documentation.
- `superseded/`: plans replaced by the current architecture or application
  behavior, including the pre-household-identity RealIDs proposal.

Local-only prototype, merge, and additional DS8 work records were removed from
the checkout after verified preservation in
[`../../archive/manifests/supplemental_history_20260826.json`](../../archive/manifests/supplemental_history_20260826.json).

Active plans live one level above and must target the native DS9.1 runtime.
Files named `agent_policy_snapshot.md` preserve obsolete local instructions as
history only; repository and `plans/` policy always take precedence.

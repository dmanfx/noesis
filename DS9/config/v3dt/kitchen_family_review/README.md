# Kitchen–Family Room MV3DT review profile

Status: isolated recorded-input evaluation only. It is not selected by the
canonical launcher, does not alter the baseline or accepted per-room SV3DT
profiles, and cannot start without `NOESIS_MV3DT_EVALUATION=1`.

Family Room is the fixed gauge. Kitchen's accepted per-room world-to-camera
matrix is composed with the inverse of the review transform:

`E_kitchen_in_family = E_kitchen * inverse(T_kitchen_to_family)`

Living Room retains its accepted local SV3DT geometry and has no cross-camera
peer edge. Only Kitchen and Family Room publish/subscribe to one another. The
Living entry subscribes to its own topic because the DS9.1 MQTT communicator
blocks the entire batched tracker when an ordered stream entry is empty; the
self-loop supplies synchronization messages without exposing Living to another
camera's measurements or IDs.

The bound transform is useful for direct dynamic testing but remains
`review_only`. Its static reconstruction report failed held-out observation
and temporal-support gates, so results from this profile cannot promote the
geometry or enable canonical MV3DT. A short Kitchen/Family connector walk is
still the preferred way to close that geometry gate.

The July synchronized replay is the single-person handoff test. The longer
three-file recording is the multi-person false-merge/false-split stress test.
The recorded cohort uses complete batches (`batched-push-timeout: -1`) with
`sync-inputs: 0` plus `BaseConfig.useBatchNumForFrameId: 1`, assigning every
stream in a tracker batch the same frame ID. Separately attached system clocks
are not used to synchronize already frame-aligned MP4s. These settings and the
Living self-loop exist only in the review profile.

# Detector validation: output validity and expected behavior

Evaluate three independent checks. Do not diagnose a parser from scene occupancy.

| Check | Evidence required | Failure means |
| --- | --- | --- |
| Execution and coverage | Successful bounded run or explained bounded stop, per-source frame progress, expected inference cadence, output-writer behavior and fresh artifacts | Determine which producer/consumer stopped or omitted evidence; no parser attribution yet |
| Output validity | Documented row/schema, legal class/label mapping, finite scores and coordinates, valid confidence domain and box geometry in the documented frame space | Inspect tensors, preprocessing, parser and writer at the first invalid boundary |
| Detection behavior | Known positive and negative fixture frames, with expected classes/locations and task-appropriate tolerances | Integration remains incomplete until the expectation is met or the fixture is corrected with evidence |

For KITTI output, validate fields using the actual writer's schema, including
optional confidence conventions. Check that coordinates are finite, ordered,
and within the writer's documented clipping/coordinate policy. Reject nonfinite
or malformed values even if detections appear in every frame. Do not prescribe
one universal confidence threshold or infer labels from numeric IDs without the
model's label mapping.

Count **processed frames** independently from files or detections. Some writers
omit empty frames; others write empty files. State that behavior and the
measurement denominator. Missing files are inconclusive until execution,
coverage and writer configuration have been checked.

Record target-bearing fixture frames before assessing detection behavior. A
fixture with targets in 5 of 100 frames can pass when those five frames have the
expected detections and its 95 negative frames remain valid. An empty-room clip
with zero detections can validate negative behavior and output plumbing, but
cannot alone prove positive detector/parser behavior. Unexpected positives on
negative frames must be assessed against the fixture's false-positive tolerance.

Use occupancy or recall thresholds only when the fixture defines target presence
and the expected tolerance. Never replace an established fixture threshold with
an arbitrary 90% detection-bearing-frame requirement. If target annotations are
unavailable, report observed detections and output validity, with positive/negative
behavior unverified; do not mark the whole integration passed.

Retain frame IDs/timestamps, exact source and engine/parser/config provenance,
expected versus observed classes/boxes, and actual failure evidence. Repair the
smallest demonstrated cause; rerun only the affected check and direct consumer.

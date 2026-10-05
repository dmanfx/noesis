# Static reconstruction and the served scene

Status: current authority boundaries, 2026-10-04.

Noesis owns retained static-camera reference evidence and coherent scene
artifacts. Menon consumes verified same-origin scene payloads and canonical
people; it does not reconstruct metric person positions in the browser.

## Independent reference versus PCF

The phone-only DA3/conditioned-MapAnything PCF path keeps static-camera imagery
and calibration as independent registration/validation evidence. A paired
reference binds the exact recording, selected source frames, rectified raster,
camera calibration and target world revision. It is not inserted into phone
inference or adjusted to fit person tracks. See
[PCF workflow](PCF_Workflow.md) and
[phone reconstruction fitting](room_reconstruction_fitting.md).

PCF supplies metric reconstruction and floorplan presentation. Explicit measured
support artifacts separately define where an admitted floor level applies; the
current upper patch retains the owner's measured 0.1905 m parallel rise. Static
background depth cannot measure a person occluding it. Room scale consistency,
current occupied measurement validity and independent physical accuracy are
different checks.

## Current Menon connection

The authenticated `/api/v1/scenes/current/payload` owns the coherent authored
model, dependencies and room artifacts, not independently newest revisions.
Exact bytes and revisions are verified under the
[REST scene contract](api_contracts_rest.md#scene-releases).

Canonical people come from `world_snapshot`, preserving XYZ, covariance,
support geometry, measured/predicted/held state and authoritative continuity.
The calibration bundle's `world_frame_presentations` maps each target revision
once into that actual authored model. Current structural fits are derived from
reviewed PCF/architectural correspondences, not hand-placed camera devices or a
browser offset. The owner-selected all-room fit remains unqualified because
Family/Kitchen opening holdouts fail; see
[structural registration](structural_scene_registration.md).

The `Whole home PCF overlay` uses the RGB tracking-review descriptor and the
same per-room mappings as people. Its temporary standalone viewer retains
original/high-detail meshes. These review surfaces cannot become localization
authority, even when they look convincingly aligned.

## Operations and history

Use the existing configured native Noesis service on loopback REST 8080 and WS
6008, with Menon providing authenticated HTTPS delivery. Do not follow old
split-port or duplicate-runtime recipes. Model/config/calibration selection
comes from the actual service and binding, not an old scene-release date.

The [May/June model-surface builder record](history/experiments/virtual_twin_reconstruction_20260503.md)
preserves ZeroPlane, atlas and initial candidate evidence. It is historical
diagnostic context, not the current PCF workflow or live person-position path.

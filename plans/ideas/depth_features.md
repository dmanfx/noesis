# New Depth Feature Ideas

This document captures a set of “high wow per line of code” ideas that can be built from maps you already have (e.g., depth/height/distance, normals, density/confidence), plus a quick visual explanation of what each idea looks like.

## TL;DR (highest “wow per line of code”)

If you already have **depth + height + distance + normals + density/confidence**, you can cheaply generate:

1. **Plane maps** (floor/wall/table IDs) + clean **floor mask / wall mask** from *normals ∧ depth*
2. **Traversability / free-space** maps (what’s walkable vs obstacle) from *height ∧ slope*
3. **Signed distance field (SDF)** to obstacles (for path planning + “don’t bump into stuff” logic) from an occupancy grid
4. **Change maps** (what moved since yesterday) from *Δdepth vs baseline*
5. **3D object metrics** for detections/tracks (meters, speed, distance-to-camera, clearance) from *depth sampling inside masks*
6. **Candy visuals**: shaded “relief” renders, contour lines, point-cloud/mesh flythroughs

---

# 1) Cool things by **combining** the maps you already have

## A) Plane segmentation: “what surface is this?”

**Inputs:** normals (RGB), depth/distance
**Output:** per-pixel plane labels + confidence

- Cluster normals + depth consistency → label big planes: **floor**, **walls**, **ceiling**, **table tops**, **couch faces**
- This becomes the backbone for *world logic*:
  - “person is on floor plane”
  - “object is on support plane”
  - “door plane moved” (camera drift / scene change)

**Extra bonus:** Once you have planes, you can render a **“cleaned normals”** view (denoise normals within each plane), which makes everything downstream look way more stable.

---

## B) Traversability map: “where could a person walk?”

**Inputs:** height + normals
**Outputs:** walkable mask, obstacle mask, slope map

- A simple rule works surprisingly well:
  - walkable if **height near floor plane** AND **surface is near-horizontal** (normal points “up”)
  - obstacle otherwise
- Huge for Noesis-style logic: make trackers “snap” feet to walkable regions and reject impossible trajectories (teleporting through a couch).

---

## C) Obstacle distance field (SDF): “how close am I to stuff?”

**Inputs:** obstacle mask (from B)
**Output:** SDF / distance-to-nearest-obstacle map (in pixels or meters)

- Take the obstacle mask in BEV or image space, run a distance transform.
- Enables:
  - “keep 0.5m clearance”
  - “find wide-open zones”
  - “auto-inflate obstacles” (safer path planning / zone logic)

---

## D) Change / novelty heatmaps: “what moved since baseline?”

**Inputs:** current depth vs baseline depth (or median-of-last-week)
**Output:** Δdepth map + “changed geometry” mask

- `delta = abs(depth_now - depth_baseline)`
- Threshold + morphology → “something changed here”
- Great for:
  - detecting moved furniture
  - detecting packages on porch / objects appearing
  - spotting camera bump/drift (global shift in depth patterns)

Also doubles as a **camera-health** signal.

---

## E) Depth-aware tracking metrics: “tracks but in meters”

**Inputs:** detections (bboxes/masks) + depth
**Outputs:** per-track distance, height estimate, velocity (m/s), occlusion risk

For each detection mask:

- sample depth at “feet region” or robust median depth in mask
- project to world/BEV (if you have intrinsics/extrinsics)
- compute:
  - **distance-to-camera**
  - **estimated height**
  - **relative speed** (frame-to-frame world displacement)
  - **interaction events** (“within 0.3m of couch edge”)

This is where your *pretty maps* turn into *useful physics*.

---

## F) Occlusion reasoning: “can the camera even see them?”

**Inputs:** depth + normals + your track’s expected world position
**Outputs:** occlusion likelihood map / visibility cones

- Ray-test in image space: if something nearer is in the line of sight, your track is likely occluded (don’t panic-reID).
- Pair with re-ID stabilization: down-weight re-ID switches during predicted occlusion.

---

# 2) Other “maps like these” you can generate (cheap + valuable)

## Geometry derivatives

- **Depth gradient magnitude** (edge strength): highlights object boundaries and depth discontinuities
- **Curvature / Laplacian of depth**: detects corners, sharp shape changes
- **Slope map** (from normals): how tilted each surface is
- **Roughness / planarity map**: local variance of normals or depth after plane fit (good for “is this a flat surface?”)

## Confidence / quality maps

If your “density” view is essentially *where depth is reliable*:

- **Confidence-weighted depth** (filter out low-density areas)
- **Hole / missing-depth map** (useful for debugging + for inpainting)

## “Distance-to-*X*” maps (very useful for rules)

- **Distance-to-floor** (height above floor plane)
- **Distance-to-wall** (once you have wall planes)
- **Distance-to-zone boundary** (for smooth triggers instead of binary zones)

## BEV-specific maps (candy + ops usefulness)

- **Occupancy grid** (2D top-down obstacles)
- **Activity heatmap** (accumulate footpoints over time)
- **Flow map / trail density** (vector field of motion directions)
- **Dwell time map** (where people linger)

---

# 3) Other visuals you can render from MapAnything outputs

## A) “3D reconstruction lite”

Even without full multi-view SLAM:

- **Point cloud view** (project depth into 3D; color by RGB or height)
- **Surfel render** (points with normals shaded = looks shockingly good)
- **Mesh preview** (simple triangulation in image space + depth; not perfect but great for UI)

## B) Cinematic “relief” renders (cheap wow)

- **Hillshade**: pretend there’s a sun; shade by normals/height (makes structure pop)
- **Contour lines**: iso-depth or iso-height lines (topographic map feel)
- **Matcap / normal-shaded render**: single texture lookup using normals → instantly readable shapes

## C) Debug views that actually help

- **Normal-consistency overlay** (where normals are noisy)
- **Plane-ID overlay** (each plane in a flat color)
- **Reprojection error view** (if you fuse multi-cam later)

---

# 4) Tie MapAnything outputs into “something else” (useful + cool)

## A) World logic / automation rules (Noesis core)

- Replace fragile pixel-zone triggers with **world-zone triggers**:
  - “person within 0.5m of stove”
  - “object placed on table plane”
  - “entered hallway corridor region”
- Use **distance-to-zone** maps for smooth hysteresis (less flicker).

## B) Camera drift + recalibration nudges

- Track baseline plane normals (floor/wall). If they rotate/shift over time, likely causes:
  - camera moved
  - calibration stale
  - lens bumped
- Feeds a “self-healing” storyline nicely.

## C) Better crops for SGIEs / ReID

- Use depth to produce **depth-gated crops**:
  - ignore background behind the person
  - tighten crop to a depth band around the person’s median depth
- Often improves embedding stability by removing background texture.

## D) Data products

- Log lightweight metrics per frame:
  - percent walkable area, obstacle area, mean depth
  - per-track distance + speed
  - scene-change score
- These become dashboards, alerts, and “memory palace” primitives.

---

# A practical “next step” recipe (minimum effort, maximum payoff)

If you implement only three new derived products:

1. **Plane segmentation (floor/wall/table)** from normals+depth
2. **Traversability + obstacle mask** from floor plane + slope
3. **Δdepth change map** vs baseline

These unlock: world-space rules, safer tracking, scene memory, and some seriously slick visuals.

---

# Visual explanation of each idea

Below is a quick “what it looks like” for each option mentioned.

## Plane segmentation (floor / wall / table IDs)

- **Flat-color label map** (each plane gets a solid color) overlaid on RGB.
- Optional **edge outlines** where plane boundaries are.
- Optional **confidence alpha** (uncertain pixels fade out).

## Traversability / free-space

- **Green = walkable**, **red = obstacle**, **gray = unknown** overlay on RGB.
- Or rendered as a **top-down BEV** “walkable carpet” silhouette.

## Obstacle distance field (SDF)

- **Heatmap**: dark near obstacles → bright far away (or vice versa).
- Often paired with **contour rings** (“0.5m, 1.0m, 1.5m…” clearance bands).

## Change / novelty heatmap (Δdepth)

- **Heatmap overlay** on RGB showing “moved/changed geometry.”
- Or a **binary mask** + bounding blobs around changed regions.

## Depth-aware tracking metrics (in meters)

- Standard bbox/mask overlay, plus **text HUD** near each track:
  - `dist=3.2m`, `speed=0.7m/s`, `height≈1.75m`
- In BEV: **track dots** with trails, color-coded by speed or distance.

## Occlusion reasoning / visibility

- **Occlusion likelihood overlay**: transparent red where a track is likely hidden.
- Or “visibility cones” in BEV from camera position with blocked areas shaded.

---

## Depth gradient / edge strength

- **Grayscale edge map** (like Sobel/Canny but from depth).
- Or **neon outlines** composited on RGB to show depth discontinuities.

## Curvature / depth Laplacian

- **Heatmap** highlighting corners, ridges, high-curvature surfaces.
- Makes shape detail pop where depth is locally bending.

## Slope map (from normals)

- **Heatmap** of tilt angle (flat surfaces one color, steep another).
- Often combined with traversability: steep → non-walkable.

## Roughness / planarity map

- **Speckle/heatmap** showing where surfaces are “messy” vs flat.
- Great debug overlay: “why is depth garbage on this couch?”

## Confidence / hole map

- **Alpha mask**: reliable depth opaque, unreliable transparent.
- “Holes” show as black/white islands (useful for inpainting diagnostics).

## Distance-to-floor / distance-to-wall

- **Height heatmap** (you already do this) but can be **relative to plane** so it stays stable even if camera pitch changes slightly.
- For walls: **heatmap** that grows with distance from the wall plane.

## Distance-to-zone boundary

- **Soft gradient bands** around a zone (like a fuzzy border).
- Lets you trigger rules smoothly (“warming up” as you approach).

---

## BEV occupancy grid

- **Top-down black/white** (occupied vs free) or colored by class.
- Often rendered with **camera frustum** and **grid lines** for scale.

## Activity heatmap (footpoints)

- **Top-down heatmap** accumulating where people spend time.
- Looks like your “density” view but in world/BEV coordinates.

## Flow / trail density

- **Top-down trails** (comet tails) behind tracks.
- Or a **vector field** (little arrows) showing dominant motion direction.

## Dwell-time map

- Similar to activity heatmap, but weighted by **time stationary**.
- Highlights “standing spots” (sink, fridge, couch corners).

---

## Point cloud view

- A rotatable **3D scatter** of points colored by RGB/height.
- Looks like a “foggy hologram” of the room.

## Surfel render (points + normals shaded)

- Point cloud but each point renders as a tiny disk with lighting → **much cleaner**
- Gives a “3D scan” vibe without full meshing.

## Mesh preview

- A **coarse 3D mesh** of the scene (often blobby) you can orbit around.
- Best shown with **matcap / normal shading** to make geometry legible.

## Hillshade / relief render

- Depth/height lit by a fake sun: **terrain-map look**.
- Very readable structure; great for UI panels.

## Contour lines

- **Topographic lines** on height/depth.
- Looks like a map; excellent for showing shape in a compact way.

## Matcap / normal-shaded render

- Normals mapped through a material sphere texture → **instant 3D feel**
- Often used as a “shape preview” panel.

## Normal-consistency / plane-ID debug overlays

- **Noise map**: highlights unstable normals/depth regions.
- **Plane-ID**: crisp colored segments + boundary outlines.

## Reprojection error view (if multi-cam / fusion later)

- **Heatmap** of mismatch (good = dark, bad = bright).
- Tells you where calibration/fusion is failing.

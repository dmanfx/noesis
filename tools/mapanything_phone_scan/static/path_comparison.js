const PATH_COMPARISON_SCHEMA = "noesis.phone_scan.path_comparison.v1";
const DEFAULT_VIEWBOX = { width: 100, height: 100, padding: 9 };

const SVG_NS = "http://www.w3.org/2000/svg";

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function finiteNumber(value, fallback = null) {
  if (value === null || value === undefined || typeof value === "boolean") return fallback;
  if (typeof value === "string" && value.trim() === "") return fallback;
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function point(value) {
  if (!Array.isArray(value) || value.length < 3) return null;
  const result = value.slice(0, 3).map((item) => finiteNumber(item));
  return result.every((item) => item !== null) ? result : null;
}

function text(value, fallback = "") {
  return value === null || value === undefined ? fallback : String(value);
}

function numberLabel(value, digits = 2, fallback = "—") {
  const number = finiteNumber(value);
  return number === null ? fallback : number.toFixed(digits);
}

function integerLabel(value, fallback = "0") {
  const number = finiteNumber(value);
  return number === null ? fallback : Math.max(0, Math.round(number)).toLocaleString();
}

function statusLabel(value, fallback = "not reported") {
  const raw = text(value, fallback).replaceAll("_", " ").trim();
  return raw ? raw : fallback;
}

function safeUrl(value) {
  if (typeof value !== "string" || !value.trim()) return "";
  try {
    const url = new URL(value, globalThis.location?.origin || "http://roomwalk.invalid");
    if (!['http:', 'https:'].includes(url.protocol)) return "";
    if (globalThis.location?.origin && url.origin !== globalThis.location.origin) return "";
    return `${url.pathname}${url.search}${url.hash}`;
  } catch (_) {
    return "";
  }
}

function firstRecord(...values) {
  return values.find(isRecord) || {};
}

/**
 * Accept the compact public result and the downloaded full comparison without
 * making the UI depend on the path-review wrapper's exact nesting.
 */
export function normalizePathComparisonPayload(value = {}) {
  const root = isRecord(value) ? value : {};
  const results = isRecord(root.results) ? root.results : root;
  const comparison = firstRecord(
    results.path_comparison,
    root.path_comparison,
    results.comparison,
    root.comparison,
    Array.isArray(root.tracks) || isRecord(root.reference) ? root : {},
  );
  const refinement = firstRecord(
    results.path_refinement,
    root.path_refinement,
    results.sensor_refined_path,
    root.sensor_refined_path,
  );
  const artifactUrls = firstRecord(
    results.artifact_urls,
    root.artifact_urls,
    comparison.artifact_urls,
  );
  const reference = firstRecord(comparison.reference, results.reference, root.reference);
  const poses = Array.isArray(reference.poses)
    ? reference.poses.filter((pose) => isRecord(pose))
    : [];
  const tracks = Array.isArray(comparison.tracks)
    ? comparison.tracks.filter((track) => isRecord(track))
    : [];
  return {
    schema: text(comparison.schema || root.schema),
    status: text(comparison.status || root.status),
    reason: text(comparison.reason || root.reason),
    registration: comparison.registration || root.registration || {},
    timing: comparison.timing || root.timing || {},
    accuracyQualified: comparison.accuracy_qualified ?? root.accuracy_qualified ?? false,
    reference: { ...reference, poses },
    tracks,
    pathRefinement: refinement,
    artifactUrls,
  };
}

export function pathComparisonSummary(value = {}) {
  const normalized = normalizePathComparisonPayload(value);
  return {
    schema: normalized.schema,
    status: normalized.status,
    reason: normalized.reason,
    registration: normalized.registration,
    timing: normalized.timing,
    accuracyQualified: normalized.accuracyQualified === true,
    trackCount: normalized.tracks.length,
    matchedCount: normalized.tracks.reduce((total, track) => total + (finiteNumber(track.matched_count, 0) || 0), 0),
    poseCount: normalized.reference.poses.length,
  };
}

function trackKey(track) {
  return text(track.track_key || `${track.tracker_id ?? "?"}/${track.lifecycle_generation ?? "?"}`);
}

function pairMap(track) {
  const map = new Map();
  for (const pair of Array.isArray(track?.pairs) ? track.pairs : []) {
    const index = finiteNumber(pair?.provider_index);
    if (index !== null) map.set(index, pair);
  }
  return map;
}

function splitSegments(rows, coordinateKey, { includeBreaks = true } = {}) {
  const segments = [];
  let current = [];
  for (const row of rows) {
    const coordinates = point(row?.[coordinateKey]);
    if (!coordinates) continue;
    if (includeBreaks && row.break_before && current.length) {
      segments.push(current);
      current = [];
    }
    current.push({ row, point: coordinates });
  }
  if (current.length) segments.push(current);
  return segments;
}

function horizontalPoint(value) {
  const coordinates = point(value);
  return coordinates ? [coordinates[0], coordinates[2]] : null;
}

function boundsFor(points) {
  const valid = points.filter((item) => Array.isArray(item) && item.length >= 2 && item.every(Number.isFinite));
  if (!valid.length) return { minX: -1, maxX: 1, minZ: -1, maxZ: 1 };
  const xs = valid.map((item) => item[0]);
  const zs = valid.map((item) => item[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minZ = Math.min(...zs);
  const maxZ = Math.max(...zs);
  const span = Math.max(maxX - minX, maxZ - minZ, 0.5);
  const centerX = (minX + maxX) / 2;
  const centerZ = (minZ + maxZ) / 2;
  const halfSpan = span / 2 + span * 0.08;
  return {
    minX: centerX - halfSpan,
    maxX: centerX + halfSpan,
    minZ: centerZ - halfSpan,
    maxZ: centerZ + halfSpan,
  };
}

function coordinateMapper(bounds, viewbox = DEFAULT_VIEWBOX) {
  const width = viewbox.width - viewbox.padding * 2;
  const height = viewbox.height - viewbox.padding * 2;
  return ([x, z]) => [
    viewbox.padding + ((x - bounds.minX) / Math.max(bounds.maxX - bounds.minX, Number.EPSILON)) * width,
    viewbox.padding + ((bounds.maxZ - z) / Math.max(bounds.maxZ - bounds.minZ, Number.EPSILON)) * height,
  ];
}

/**
 * Pure geometry used by both the SVG renderer and Node tests. World axes are
 * x/y-up/z, so the top-down projection uses x/z and deliberately ignores y.
 */
export function buildTopdownGeometry(value = {}, selectedTrackKey = "", poseIndex = 0) {
  const comparison = normalizePathComparisonPayload(value);
  const poses = comparison.reference.poses;
  const selectedTrack = comparison.tracks.find((track) => trackKey(track) === selectedTrackKey) || null;
  const selectedPairs = selectedTrack ? selectedTrack.pairs.filter((pair) => isRecord(pair)) : [];
  const points = [];
  for (const pose of poses) {
    const original = horizontalPoint(pose.original_world_m);
    const reference = horizontalPoint(pose.reference_world_m);
    if (original) points.push(original);
    if (reference) points.push(reference);
  }
  for (const pair of selectedPairs) {
    const noesis = horizontalPoint(pair.noesis_world_m);
    if (noesis) points.push(noesis);
  }
  const bounds = boundsFor(points);
  const map = coordinateMapper(bounds);
  const selectedPoseIndex = Math.max(0, Math.min(poses.length - 1, Math.round(finiteNumber(poseIndex, 0) || 0)));
  const pose = poses[selectedPoseIndex] || null;
  const pair = selectedTrack ? selectedPairs.find((item) => finiteNumber(item.provider_index) === selectedPoseIndex) || null : null;
  return {
    comparison,
    bounds,
    map,
    originalSegments: splitSegments(poses, "original_world_m").map((segment) => segment.map((item) => ({ ...item, point: map(horizontalPoint(item.point)) }))),
    referenceSegments: splitSegments(poses, "reference_world_m").map((segment) => segment.map((item) => ({ ...item, point: map(horizontalPoint(item.point)) }))),
    selectedTrack,
    selectedTrackSegments: splitSegments(selectedPairs, "noesis_world_m").map((segment) => segment.map((item) => ({ ...item, point: map(horizontalPoint(item.point)) }))),
    selectedPoseIndex,
    pose,
    pair,
  };
}

export function formatSeparation(value) {
  const number = finiteNumber(value);
  return number === null ? "Separation —" : `Separation ${number.toFixed(3)} m`;
}

function element(document, tag, className = "", content = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== null) node.textContent = content;
  return node;
}

function svgElement(document, tag, attributes = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (value !== undefined && value !== null) node.setAttribute(name, String(value));
  }
  return node;
}

function link(document, url, label) {
  const href = safeUrl(url);
  if (!href) return null;
  const anchor = element(document, "a", "artifact-link", label);
  anchor.href = href;
  anchor.target = "_blank";
  anchor.rel = "noopener";
  anchor.download = "";
  return anchor;
}

function badge(document, label, className = "") {
  return element(document, "span", `status-badge ${className}`.trim(), label);
}

function stat(document, value, label) {
  const wrapper = element(document, "div", "stat");
  wrapper.append(element(document, "b", "", value), element(document, "span", "", label));
  return wrapper;
}

function addTextRow(document, parent, label, value) {
  const row = element(document, "div", "path-comparison-detail-row");
  row.append(element(document, "span", "", label), element(document, "b", "", value));
  parent.append(row);
}

function renderLegend(document) {
  const legend = element(document, "div", "path-comparison-legend");
  const entries = [
    ["path-legend-original", "Original phone path"],
    ["path-legend-reference", "Reference phone path"],
    ["path-legend-track", "Selected Noesis lifecycle"],
    ["path-legend-break", "Gap / segment break"],
  ];
  for (const [className, label] of entries) {
    const item = element(document, "span", "path-comparison-legend-item");
    item.append(element(document, "i", `path-legend-swatch ${className}`), element(document, "span", "", label));
    legend.append(item);
  }
  return legend;
}

function renderPolyline(document, parent, segments, className) {
  for (const segment of segments) {
    if (segment.length < 2) continue;
    const polyline = svgElement(document, "polyline", {
      class: className,
      points: segment.map((item) => item.point.join(",")).join(" "),
    });
    parent.append(polyline);
  }
}

function renderBreaks(document, parent, segments, className) {
  for (const segment of segments) {
    const first = segment[0];
    if (!first?.row?.break_before || !first.point) continue;
    parent.append(svgElement(document, "circle", {
      class: className,
      cx: first.point[0],
      cy: first.point[1],
      r: 1.45,
    }));
  }
}

function renderTopdown(document, parent, state, { onPoseChange, onTrackChange } = {}) {
  const comparison = state.comparison;
  const frame = element(document, "div", "path-comparison-map-frame");
  const svg = svgElement(document, "svg", {
    class: "path-comparison-map",
    viewBox: `0 0 ${DEFAULT_VIEWBOX.width} ${DEFAULT_VIEWBOX.height}`,
    role: "img",
    "aria-label": "Top-down phone path and selected Noesis lifecycle comparison",
  });
  const grid = svgElement(document, "g", { class: "path-comparison-grid" });
  for (const offset of [25, 50, 75]) {
    grid.append(
      svgElement(document, "line", { x1: offset, y1: 4, x2: offset, y2: 96 }),
      svgElement(document, "line", { x1: 4, y1: offset, x2: 96, y2: offset }),
    );
  }
  svg.append(grid);
  renderPolyline(document, svg, state.originalSegments, "path-comparison-line path-line-original");
  renderPolyline(document, svg, state.referenceSegments, "path-comparison-line path-line-reference");
  renderBreaks(document, svg, state.originalSegments, "path-comparison-break path-break-original");
  renderBreaks(document, svg, state.referenceSegments, "path-comparison-break path-break-reference");
  if (state.selectedTrack) {
    renderPolyline(document, svg, state.selectedTrackSegments, "path-comparison-line path-line-track");
    renderBreaks(document, svg, state.selectedTrackSegments, "path-comparison-break path-break-track");
  }
  const selectedOriginal = horizontalPoint(state.pose?.original_world_m);
  const selectedReference = horizontalPoint(state.pose?.reference_world_m);
  const selectedNoesis = horizontalPoint(state.pair?.noesis_world_m);
  for (const [coordinates, className, label] of [
    [selectedOriginal, "path-point-original", "Original phone position"],
    [selectedReference, "path-point-reference", "Reference phone position"],
    [selectedNoesis, "path-point-track", "Selected Noesis position"],
  ]) {
    if (!coordinates) continue;
    const mapped = state.map(coordinates);
    const circle = svgElement(document, "circle", { class: className, cx: mapped[0], cy: mapped[1], r: 2.2 });
    circle.setAttribute("aria-label", label);
    svg.append(circle);
  }
  frame.append(svg);
  parent.append(frame);

  const controls = element(document, "div", "path-comparison-controls");
  const trackField = element(document, "label", "path-comparison-field");
  trackField.append(element(document, "span", "", "Noesis lifecycle track"));
  const trackPicker = element(document, "select", "text-input");
  trackPicker.setAttribute("aria-label", "Noesis lifecycle track");
  const placeholder = element(document, "option", "", "Choose a lifecycle track…");
  placeholder.value = "";
  trackPicker.append(placeholder);
  for (const track of comparison.tracks) {
    const option = element(document, "option", "", `${trackKey(track)} · ${integerLabel(track.matched_count)} matched`);
    option.value = trackKey(track);
    trackPicker.append(option);
  }
  trackPicker.value = state.selectedTrack ? trackKey(state.selectedTrack) : "";
  trackPicker.addEventListener("change", () => onTrackChange?.(trackPicker.value));
  trackField.append(trackPicker);
  controls.append(trackField);

  const sliderField = element(document, "label", "path-comparison-field path-comparison-slider-field");
  const sliderHeading = element(document, "div", "path-comparison-slider-heading");
  sliderHeading.append(element(document, "span", "", "Time selection"));
  const sliderValue = element(document, "b", "path-comparison-slider-value", state.pose ? `${numberLabel(state.pose.phone_time_s, 2)} s` : "—");
  sliderHeading.append(sliderValue);
  const slider = element(document, "input", "path-comparison-slider");
  slider.type = "range";
  slider.min = "0";
  slider.max = String(Math.max(0, comparison.reference.poses.length - 1));
  slider.step = "1";
  slider.value = String(state.selectedPoseIndex);
  slider.disabled = comparison.reference.poses.length < 1;
  slider.setAttribute("aria-label", "Phone path time selection");
  slider.addEventListener("input", () => {
    sliderValue.textContent = comparison.reference.poses[Number(slider.value)]
      ? `${numberLabel(comparison.reference.poses[Number(slider.value)].phone_time_s, 2)} s`
      : "—";
  });
  // Keep the native range input mounted while a finger/mouse/keyboard is
  // moving it. Commit the selected sample only after the interaction settles.
  slider.addEventListener("change", () => onPoseChange?.(Number(slider.value)));
  sliderField.append(sliderHeading, slider);
  controls.append(sliderField);
  parent.append(controls);
}

function renderSelectedPair(document, parent, state) {
  const card = element(document, "div", "path-comparison-selection");
  const pose = state.pose;
  const pair = state.pair;
  card.append(element(document, "div", "eyebrow", "Selected sample"));
  if (!pose) {
    card.append(element(document, "p", "microcopy", "No reference poses are available for time selection."));
    parent.append(card);
    return;
  }
  const time = element(document, "div", "path-comparison-selected-title");
  time.append(element(document, "b", "", `Phone time ${numberLabel(pose.phone_time_s, 3)} s`));
  time.append(element(document, "span", "", pair ? formatSeparation(pair.horizontal_separation_m) : "No matched Noesis observation"));
  card.append(time);
  if (pose.break_before || pair?.break_before) card.append(element(document, "p", "path-comparison-gap-note", "Gap break at this sample; the line is not continuous across it."));
  if (!state.selectedTrack) {
    card.append(element(document, "p", "microcopy", "Choose a Noesis lifecycle track to compare its current observation. No person is selected automatically."));
  } else if (!pair) {
    card.append(element(document, "p", "microcopy", "The selected lifecycle has no matched observation at this phone time."));
  } else {
    addTextRow(document, card, "Reference phone", pose.reference_world_m ? pose.reference_world_m.map((value) => numberLabel(value, 3)).join(", ") : "—");
    addTextRow(document, card, "Noesis ground point", pair.noesis_world_m ? pair.noesis_world_m.map((value) => numberLabel(value, 3)).join(", ") : "—");
    addTextRow(document, card, "Stable ID", pair.stable_id ?? "not reported");
    addTextRow(document, card, "Trail segment", pair.trail_segment_id ?? "not reported");
  }
  parent.append(card);
}

function renderStatusSummary(document, parent, comparison, pathRefinement) {
  const grid = element(document, "div", "stat-grid path-comparison-stats");
  grid.append(
    stat(document, integerLabel(comparison.reference.poses.length), "Reference poses"),
    stat(document, integerLabel(comparison.tracks.length), "Lifecycle tracks"),
    stat(document, integerLabel(comparison.tracks.reduce((total, track) => total + (finiteNumber(track.matched_count, 0) || 0), 0)), "Matched observations"),
    stat(document, statusLabel(comparison.timing?.status, "not reported"), "Timing review"),
  );
  parent.append(grid);
  const badges = element(document, "div", "path-comparison-badges");
  badges.append(badge(document, `Registration: ${statusLabel(comparison.registration?.status || comparison.registration, "not reported")}`, comparison.registration?.status === "registered" ? "aligned-badge" : ""));
  if (comparison.accuracyQualified === true) badges.append(badge(document, "Accuracy qualified", "aligned-badge"));
  else badges.append(badge(document, "Separation diagnostic · not accuracy"));
  if (pathRefinement && Object.keys(pathRefinement).length) {
    const refined = pathRefinement.position_refined === true;
    const rejected = text(pathRefinement.status).toLowerCase() === "rejected";
    const method = text(pathRefinement.method || pathRefinement.estimator || "");
    const label = rejected
      ? "Position refinement rejected"
      : refined
        ? (pathRefinement.fused_vio === true || /fused.?vio/i.test(method) ? "Fused VIO path reported" : "Position-refined path reported")
        : "Position refinement not reported";
    badges.append(badge(document, label, refined ? "aligned-badge" : ""));
    if (rejected) {
      const visualCount = finiteNumber(pathRefinement.accepted_visual ?? pathRefinement.accepted_visual_count ?? pathRefinement.visual_accepted_count);
      const vioCount = finiteNumber(pathRefinement.accepted_vio ?? pathRefinement.accepted_vio_count ?? pathRefinement.vio_accepted_count);
      const visual = visualCount === null
        ? statusLabel(pathRefinement.visual_status || pathRefinement.visual_path?.status || "retained")
        : `${integerLabel(visualCount)} accepted`;
      const vio = vioCount === null
        ? statusLabel(pathRefinement.vio_status || pathRefinement.fused_vio_status || "not accepted")
        : `${integerLabel(vioCount)} accepted`;
      const reason = text(pathRefinement.reason || pathRefinement.rejection_reason || "Backend refinement gate did not pass.");
      const visualLabel = visualCount === null && visual === "retained" ? "Original path retained." : `Visual path: ${visual}.`;
      parent.append(element(document, "p", "microcopy path-refinement-rejection", `Refinement rejected: ${reason} ${visualLabel} VIO correction ${statusLabel(vio)}.`));
    }
  }
  parent.append(badges);
}

function renderEvidenceProblem(document, parent, comparison, summary, hasArtifact) {
  const status = text(comparison.status || summary.status, "not_ready");
  const reason = text(comparison.reason || summary.reason).trim();
  const registrationStatus = text(comparison.registration?.status || summary.registration?.status || summary.registration).toLowerCase();
  const needsAlignment = !registrationStatus || !["registered", "passed", "complete"].includes(registrationStatus);
  const problem = element(document, "div", "path-comparison-evidence status-panel");
  problem.append(element(document, "div", "eyebrow", "Comparison evidence"));
  problem.append(element(document, "h4", "", statusLabel(status, "not ready")));
  problem.append(element(document, "p", "path-comparison-reason", reason || (hasArtifact ? "The comparison artifact did not contain a usable path pair." : "No full comparison artifact was published for this review.")));
  const action = needsAlignment
    ? "Align the visual reconstruction to its selected Noesis camera first, then run Review path again."
    : "Run Review path again after the retained evidence or refinement result changes.";
  problem.append(element(document, "p", "path-comparison-action", action));
  parent.append(problem);
}

function renderArtifactLinks(document, parent, artifactUrls) {
  const row = element(document, "div", "artifact-row path-comparison-artifacts");
  const links = [
    [artifactUrls.comparison, "Full comparison JSON"],
    [artifactUrls.room_reference, "Room reference JSON"],
    [artifactUrls.comparison_csv, "Comparison CSV"],
  ];
  for (const [url, label] of links) {
    const anchor = link(document, url, label);
    if (anchor) row.append(anchor);
  }
  if (row.childElementCount) parent.append(row);
}

function renderInteractive(document, parent, comparison, summary, pathRefinement, artifactUrls, state = { trackKey: "", poseIndex: 0 }, callbacks = {}) {
  parent.replaceChildren();
  const heading = element(document, "div", "subheading path-comparison-heading");
  const copy = element(document, "div");
  copy.append(element(document, "span", "eyebrow", "Path comparison"), element(document, "h3", "", "Phone path against selected Noesis evidence"));
  heading.append(copy, badge(document, statusLabel(comparison.status || summary.status, "review")));
  parent.append(heading);
  const note = element(document, "p", "path-comparison-note", "Review the original and reference phone paths, then choose a Noesis lifecycle. The UI never assigns the phone carrier automatically.");
  parent.append(note);
  renderStatusSummary(document, parent, comparison, pathRefinement);
  const legend = renderLegend(document);
  legend.classList.add("path-comparison-legend-block");
  parent.append(legend);

  const geometry = buildTopdownGeometry(comparison, state.trackKey, state.poseIndex);
  renderTopdown(document, parent, geometry, {
    onTrackChange: (key) => callbacks.onStateChange?.({ trackKey: key, poseIndex: geometry.selectedPoseIndex }),
    onPoseChange: (index) => callbacks.onStateChange?.({ trackKey: geometry.selectedTrack ? trackKey(geometry.selectedTrack) : "", poseIndex: index }),
  });
  renderSelectedPair(document, parent, geometry);
  const timingStatus = statusLabel(comparison.timing?.status, "not reported");
  const timingLimitation = text(comparison.timing?.limitation, "Timing limitations remain attached to the comparison evidence.");
  const limits = element(document, "p", "microcopy path-comparison-accuracy-note", `Phone optical center vs Noesis ground point; separation is a review diagnostic, not person accuracy. Timing: ${timingStatus}. ${timingLimitation}`);
  parent.append(limits);
  renderArtifactLinks(document, parent, artifactUrls);
}

function renderLoading(document, parent) {
  parent.replaceChildren(element(document, "p", "path-comparison-loading", "Loading the full path comparison…"));
}

function renderError(document, parent, error) {
  parent.replaceChildren();
  const panel = element(document, "div", "path-comparison-evidence status-panel");
  panel.append(element(document, "div", "eyebrow", "Comparison unavailable"), element(document, "p", "path-comparison-reason", text(error?.message, "The full comparison could not be loaded.")), element(document, "p", "path-comparison-action", "Keep the review artifact retained and run Review path again after the service is available."));
  parent.append(panel);
}

export function createPathComparisonController({ fetchImpl = globalThis.fetch?.bind(globalThis), documentRef = globalThis.document } = {}) {
  let generation = 0;
  let abortController = null;
  const stateByScan = new Map();
  const cancel = () => {
    generation += 1;
    abortController?.abort();
    abortController = null;
  };
  const render = async ({ host, scanId, summary = {}, pathRefinement = {}, artifactUrls = {} } = {}) => {
    cancel();
    const currentGeneration = generation;
    if (!host || !documentRef || !scanId) return;
    host.dataset.scanId = String(scanId);
    const compact = normalizePathComparisonPayload({ path_comparison: summary, path_refinement: pathRefinement, artifact_urls: artifactUrls });
    renderStatusSummary(documentRef, host, compact, compact.pathRefinement);
    renderArtifactLinks(documentRef, host, artifactUrls);
    const comparisonUrl = safeUrl(artifactUrls.comparison);
    if (!comparisonUrl) {
      renderEvidenceProblem(documentRef, host, compact, summary, false);
      return;
    }
    if (typeof fetchImpl !== "function") {
      renderError(documentRef, host, new Error("The browser fetch API is unavailable."));
      return;
    }
    renderLoading(documentRef, host);
    abortController = typeof AbortController === "function" ? new AbortController() : null;
    try {
      const response = await fetchImpl(comparisonUrl, abortController ? { signal: abortController.signal } : undefined);
      if (!response?.ok) throw new Error(`Comparison artifact returned ${response?.status || "an invalid response"}`);
      const full = normalizePathComparisonPayload(await response.json());
      if (currentGeneration !== generation || host.dataset.scanId !== String(scanId)) return;
      const prior = stateByScan.get(scanId) || { trackKey: "", poseIndex: 0 };
      const availableKey = full.tracks.some((track) => trackKey(track) === prior.trackKey) ? prior.trackKey : "";
      const renderState = (next = {}) => {
        const state = { trackKey: availableKey, poseIndex: 0, ...prior, ...next };
        stateByScan.set(scanId, state);
        if (currentGeneration !== generation || host.dataset.scanId !== String(scanId)) return;
        if (!full.reference.poses.length || !full.tracks.length || !["comparison_ready", "ready", "complete"].includes(text(full.status).toLowerCase())) {
          renderEvidenceProblem(documentRef, host, full, summary, true);
          renderArtifactLinks(documentRef, host, artifactUrls);
          return;
        }
        renderInteractive(documentRef, host, full, summary, pathRefinement, artifactUrls, state, { onStateChange: renderState });
      };
      renderState();
    } catch (error) {
      if (currentGeneration !== generation || error?.name === "AbortError") return;
      renderError(documentRef, host, error);
    } finally {
      if (currentGeneration === generation) abortController = null;
    }
  };
  return { render, cancel };
}

export { safeUrl };

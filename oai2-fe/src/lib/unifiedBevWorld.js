import snapshotSchema from '../../../contracts/schema/world_snapshot.schema.json' with { type: 'json' };
import eventSchema from '../../../contracts/schema/world_event.schema.json' with { type: 'json' };

// Presentation only. No per-camera BEV head, tracker row or PCF sample enters
// person authority. Callers dispatch every canonical envelope in wire order.
export const UNIFIED_BEV_LIMITS = Object.freeze({
  cameras: 32, sources: 64, entities: 256, evidencePerEntity: 32,
  eventsPerCohort: 256, trailPoints: 128, retiredRuns: 4096,
});
const WORLD = 'backend_world_m';
const SCENE = 'menon_scene';
const SHA = /^[0-9a-f]{64}$/;
const RASTER = 'row_zero_max_z_rows_toward_min_z_columns_min_x_to_max_x';
const COHORT = ['source_id', 'frame_id', 'observed_at_us', 'tracking_publication_sequence'];
const record = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const integer = (v, min = 0) => Number.isSafeInteger(v) && v >= min;
const text = (v) => typeof v === 'string' && v.length > 0 && v.length <= 200 && v.trim() === v;
const equal = (a, b) => {
  if (a === b) return true;
  if (Array.isArray(a)) return Array.isArray(b) && a.length === b.length && a.every((v, i) => equal(v, b[i]));
  if (!record(a) || !record(b)) return false;
  const keys = Object.keys(a);
  return keys.length === Object.keys(b).length && keys.every((k) => Object.hasOwn(b, k) && equal(a[k], b[k]));
};
const freeze = (v) => {
  if (v && typeof v === 'object' && !Object.isFrozen(v)) {
    Object.values(v).forEach(freeze);
    Object.freeze(v);
  }
  return v;
};
const copy = (v) => JSON.parse(JSON.stringify(v));
export class UnifiedBevError extends Error {
  constructor(code) { super(code); this.name = 'UnifiedBevError'; this.code = code; }
}
const requireValue = (condition, code) => { if (!condition) throw new UnifiedBevError(code); };

// Use the checked-in canonical schemas, rather than a second field definition.
// This bounded interpreter covers the keywords in these two generated schemas.
function schemaValue(value, schema, root, depth = 0) {
  requireValue(depth <= 16, 'contract_depth_exceeded');
  if (schema.$ref) return schemaValue(value, root.$defs[schema.$ref.split('/').at(-1)], root, depth + 1);
  if (schema.anyOf) {
    requireValue(schema.anyOf.some((branch) => {
      try { schemaValue(value, branch, root, depth + 1); return true; } catch { return false; }
    }), 'contract_shape_invalid');
    return;
  }
  if (Object.hasOwn(schema, 'const')) requireValue(value === schema.const, 'contract_constant_invalid');
  if (schema.enum) requireValue(schema.enum.includes(value), 'contract_enum_invalid');
  if (schema.type === 'null') requireValue(value === null, 'contract_shape_invalid');
  if (schema.type === 'boolean') requireValue(typeof value === 'boolean', 'contract_shape_invalid');
  if (schema.type === 'number') requireValue(Number.isFinite(value), 'contract_number_invalid');
  if (schema.type === 'integer') requireValue(integer(value, schema.minimum ?? 0), 'contract_integer_invalid');
  if (schema.type === 'string') {
    requireValue(typeof value === 'string' && value.trim() === value
      && value.length >= (schema.minLength ?? 0) && value.length <= (schema.maxLength ?? 240), 'contract_string_invalid');
    if (schema.pattern) requireValue(new RegExp(schema.pattern).test(value), 'contract_digest_invalid');
  }
  if (schema.type === 'object') {
    requireValue(record(value) && Object.keys(value).length <= 64, 'contract_shape_invalid');
    for (const key of schema.required ?? []) requireValue(Object.hasOwn(value, key), 'contract_field_missing');
    for (const [key, item] of Object.entries(value)) {
      requireValue(Object.hasOwn(schema.properties, key) || schema.additionalProperties !== false, 'contract_field_unexpected');
      if (schema.properties[key]) schemaValue(item, schema.properties[key], root, depth + 1);
    }
  }
  if (schema.type === 'array') {
    requireValue(Array.isArray(value) && value.length >= (schema.minItems ?? 0)
      && value.length <= (schema.maxItems ?? 256), 'contract_array_invalid');
    value.forEach((item, i) => schemaValue(item, schema.prefixItems?.[i] ?? schema.items ?? {}, root, depth + 1));
  }
}

function affine(value, properSimilarity = false) {
  requireValue(Array.isArray(value) && value.length === 16 && value.every(Number.isFinite)
    && value[3] === 0 && value[7] === 0 && value[11] === 0 && value[15] === 1, 'affine_transform_invalid');
  const a = [[value[0], value[4], value[8]], [value[1], value[5], value[9]], [value[2], value[6], value[10]]];
  const det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
    - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
    + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
  requireValue(Number.isFinite(det) && Math.abs(det) > 1e-12, 'affine_transform_singular');
  if (properSimilarity) {
    const gram = a.map((_, r) => a.map((__, c) => a.reduce((sum, row) => sum + row[r] * row[c], 0)));
    const scale = (gram[0][0] + gram[1][1] + gram[2][2]) / 3;
    requireValue(det > 0 && scale > 1e-12 && gram.every((row, r) => row.every((v, c) => (
      Math.abs(v - (r === c ? scale : 0)) <= Math.max(2e-5, scale * 4e-5)
    ))), 'scene_similarity_invalid');
  }
  return [...value];
}
const point = (m, p) => {
  requireValue(Array.isArray(p) && p.length === 3 && p.every(Number.isFinite), 'point_invalid');
  const out = [0, 1, 2].map((r) => m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r]);
  requireValue(out.every(Number.isFinite), 'point_invalid');
  return out;
};
const multiply = (a, b) => Array.from({ length: 16 }, (_, i) => {
  const r = i % 4; const c = Math.floor(i / 4);
  return [0, 1, 2, 3].reduce((sum, k) => sum + a[k * 4 + r] * b[c * 4 + k], 0);
});
const xyz = (p) => [p.x, p.y, p.z];
const vector = (p) => ({ x: p[0], y: p[1], z: p[2] });

export function createUnifiedBevBindings(input, authoredSceneSha256) {
  const bundle = input?.type === 'calibration-bundle' ? input.data : input;
  requireValue(record(bundle) && SHA.test(authoredSceneSha256 ?? ''), 'authored_scene_binding_missing');
  requireValue(bundle.meta?.authored_scene_sha256 === authoredSceneSha256, 'authored_scene_binding_mismatch');
  const table = bundle.cameras?.frame_bindings;
  const presentations = bundle.world_frame_presentations;
  requireValue(record(table) && record(presentations), 'world_frame_bindings_missing');
  requireValue(Object.keys(table).length > 0 && Object.keys(table).length <= UNIFIED_BEV_LIMITS.cameras
    && Object.keys(presentations).length > 0 && Object.keys(presentations).length <= UNIFIED_BEV_LIMITS.cameras, 'binding_count_invalid');
  for (const [revision, p] of Object.entries(presentations)) {
    requireValue(text(revision) && p?.contract === 'noesis.scene_prior.world_to_scene_presentation'
      && p.contract_version === 1 && p.accepted === true && p.render_frame === SCENE
      && p.world_frame?.frame_id === WORLD && p.world_frame.revision === revision, 'world_presentation_invalid');
    requireValue(p.authored_scene_sha256 === authoredSceneSha256, 'authored_scene_binding_mismatch');
    requireValue(SHA.test(p.world_to_scene_sha256 ?? '') && SHA.test(p.scene_revision_id ?? '')
      && SHA.test(p.registration_sha256 ?? '') && p.registration_sha256 === bundle.meta.registration_sha256
      && p.provenance?.sha256 === p.registration_sha256, 'presentation_digest_invalid');
    requireValue(Array.isArray(p.source_transform_sha256s) && p.source_transform_sha256s.length > 0
      && p.source_transform_sha256s.length <= 256 && p.source_transform_sha256s.every((v) => typeof v === 'string' && SHA.test(v))
      && new Set(p.source_transform_sha256s).size === p.source_transform_sha256s.length, 'source_digest_invalid');
    affine(p.world_to_scene_col_major, true);
  }
  const edges = new Map();
  for (const [camera, b] of Object.entries(table)) {
    requireValue(text(camera) && b?.contract === 'noesis.calibration.frame_binding' && [1, 2].includes(b.contract_version)
      && b.calibration_frame?.frame_id === WORLD && text(b.calibration_frame.revision)
      && b.world_frame?.frame_id === WORLD && text(b.world_frame.revision)
      && SHA.test(b.transform_sha256 ?? ''), 'world_frame_binding_invalid');
    if (b.contract_version === 2) requireValue(b.metric_frame?.accepted === true
      && equal(b.metric_frame.frame, b.world_frame), 'metric_frame_unaccepted');
    const matrix = affine(b.target_from_calibration_col_major);
    const key = `${b.world_frame.revision}:${b.transform_sha256}`;
    requireValue(!edges.has(key) || equal(edges.get(key), matrix), 'world_frame_binding_conflict');
    edges.set(key, matrix);
    requireValue(presentations[b.world_frame.revision]?.source_transform_sha256s.includes(b.transform_sha256), 'world_frame_binding_unavailable');
    if (b.presentation) requireValue(equal(b.presentation.world_to_scene_col_major, presentations[b.world_frame.revision].world_to_scene_col_major)
      && b.presentation.world_to_scene_sha256 === presentations[b.world_frame.revision].world_to_scene_sha256
      && b.presentation.authored_scene_sha256 === authoredSceneSha256, 'world_frame_binding_conflict');
  }
  // Copy only the compact contract tables, never camera images or telemetry.
  const cameras = Object.fromEntries(Object.entries(table).map(([camera, b]) => [camera, {
    contract: b.contract, contract_version: b.contract_version,
    calibration_frame: { frame_id: WORLD, revision: b.calibration_frame.revision },
    world_frame: { frame_id: WORLD, revision: b.world_frame.revision },
    transform_sha256: b.transform_sha256,
    target_from_calibration_col_major: [...b.target_from_calibration_col_major],
  }]));
  const compactPresentations = Object.fromEntries(Object.entries(presentations).map(([revision, p]) => [revision, {
    contract: p.contract, contract_version: 1, accepted: true, render_frame: SCENE,
    world_frame: { frame_id: WORLD, revision },
    world_to_scene_col_major: [...p.world_to_scene_col_major],
    world_to_scene_sha256: p.world_to_scene_sha256, scene_revision_id: p.scene_revision_id,
    source_transform_sha256s: [...p.source_transform_sha256s],
    authored_scene_sha256: authoredSceneSha256, registration_sha256: p.registration_sha256,
    provenance: { sha256: p.registration_sha256 },
  }]));
  return freeze({ coordinateFrame: SCENE, authoredSceneSha256,
    registrationSha256: bundle.meta.registration_sha256,
    cameras, presentations: compactPresentations });
}

function presentationFor(bindings, revision, hashes) {
  const p = bindings?.presentations?.[revision];
  requireValue(p && hashes.length > 0 && hashes.every((h) => p.source_transform_sha256s.includes(h)), 'world_frame_binding_unavailable');
  return p;
}

export function projectUnifiedBevMap(bindings, floorplan) {
  const f = floorplan;
  const cameraId = f?.camera_id;
  const binding = bindings?.cameras?.[cameraId];
  const meta = f?.scene_prior_meta;
  const geometry = f?.scene_prior_diagnostic_meta;
  requireValue(f?.type === 'floorplan_response' && f.frame === 'camera_local_ground_m' && f.units === 'meters'
    && f.world_frame === WORLD && f.scene_prior_only === true && f.display_source === 'pcf'
    && integer(f.ts, 1) && text(f.snapshot_id) && !f.error && binding, 'static_map_frame_invalid');
  requireValue(meta?.contract === 'noesis.scene_prior.floorplan_composite' && meta.contract_version === 1
    && meta.source_frame === WORLD && meta.target_frame === f.frame
    && meta.source_frame_revision === f.world_frame_revision
    && meta.source_frame_revision === binding.world_frame.revision
    && meta.frame_transform_sha256 === binding.transform_sha256
    && meta.camera_geometry_source === 'current_calibrated_extrinsics'
    && meta.prior_id === f.snapshot_id && SHA.test(f.snapshot_content_sha256 ?? '')
    && meta.revision_manifest_sha256 === f.snapshot_content_sha256, 'static_map_binding_mismatch');
  requireValue(geometry?.contract === 'noesis.scene_prior.diagnostic_layers' && geometry.contract_version === 1
    && geometry.raster_orientation === RASTER && meta.raster_orientation === RASTER
    && equal(geometry.bounds, f.bounds), 'static_map_geometry_invalid');
  const bounds = f.bounds;
  requireValue(record(bounds) && ['min_x', 'max_x', 'min_z', 'max_z'].every((k) => Number.isFinite(bounds[k]))
    && bounds.min_x < bounds.max_x && bounds.min_z < bounds.max_z, 'static_map_bounds_invalid');
  const shape = geometry.grid_shape;
  requireValue(Array.isArray(shape) && shape.length === 2 && shape.every((n) => integer(n, 1))
    && shape[0] * shape[1] <= 4_194_304 && Number.isFinite(geometry.resolution_m) && geometry.resolution_m > 0
    && Math.abs(shape[1] * geometry.resolution_m - (bounds.max_x - bounds.min_x)) < 1e-5
    && Math.abs(shape[0] * geometry.resolution_m - (bounds.max_z - bounds.min_z)) < 1e-5, 'static_map_grid_invalid');
  const center = geometry.camera_position_world_m;
  const right = geometry.camera_right_world_xz;
  const forward = geometry.camera_forward_world_xz;
  requireValue(Array.isArray(center) && center.length === 3 && center.every(Number.isFinite)
    && [right, forward].every((v) => Array.isArray(v) && v.length === 2 && v.every(Number.isFinite)
      && Math.abs(Math.hypot(...v) - 1) < 1e-5)
    && Math.abs(right[0] * forward[0] + right[1] * forward[1]) < 1e-5
    && Number.isFinite(meta.floor_y_m), 'static_map_basis_invalid');
  const p = presentationFor(bindings, f.world_frame_revision, [meta.frame_transform_sha256]);
  // Diagnostic camera basis is ALREADY in the target world revision. Do not
  // apply target_from_calibration again. Local Y is height above this floor.
  const worldFromLocal = [right[0], 0, right[1], 0, 0, 1, 0, 0, forward[0], 0, forward[1], 0,
    center[0], meta.floor_y_m, center[2], 1];
  const sceneFromLocalColMajor = affine(multiply(p.world_to_scene_col_major, worldFromLocal));
  const corners = [[bounds.min_x, bounds.min_z], [bounds.max_x, bounds.min_z],
    [bounds.max_x, bounds.max_z], [bounds.min_x, bounds.max_z]];
  return freeze({ cameraId, mapId: f.snapshot_id, coordinateFrame: SCENE,
    sourceFrame: f.frame, worldFrameRevision: f.world_frame_revision,
    worldTransformSha256: meta.frame_transform_sha256, sceneRevisionId: p.scene_revision_id,
    sceneMappingSha256: p.world_to_scene_sha256, authoredSceneSha256: bindings.authoredSceneSha256,
    snapshotContentSha256: f.snapshot_content_sha256, staticObservedAtUs: f.ts,
    positionAuthority: 'static_only', localBounds: { ...bounds }, gridShape: [...shape], rasterOrientation: RASTER,
    sceneFromLocalColMajor, sceneCorners: corners.map(([x, z]) => point(sceneFromLocalColMajor, [x, 0, z])) });
}

export function projectUnifiedBevMapPoint(map, localXZ, heightAboveFloorM = 0) {
  requireValue(map?.positionAuthority === 'static_only' && Array.isArray(localXZ) && localXZ.length === 2, 'static_map_point_invalid');
  return point(map.sceneFromLocalColMajor, [localXZ[0], heightAboveFloorM, localXZ[1]]);
}

function cohortOf(message) {
  const c = message?.cohort;
  requireValue(record(c) && Object.keys(c).length === COHORT.length && COHORT.every((k) => Object.hasOwn(c, k)), 'cohort_invalid');
  requireValue(integer(c.source_id) && c.source_id <= 4096 && integer(c.tracking_publication_sequence)
    && (c.frame_id === null || integer(c.frame_id)) && (c.observed_at_us === null || integer(c.observed_at_us, 1)), 'cohort_invalid');
  requireValue(COHORT.every((k) => Object.hasOwn(message, k) && message[k] === c[k]), 'cohort_mismatch');
  return { ...c };
}
const runKey = (p) => `${p.runtime}:${p.instance_id}:${p.run_id}`;
function validateAttribution(e) {
  const q = e.attribution_quarantine;
  if (q == null) {
    requireValue(!e.subject.subject_id.startsWith('quarantine:'), 'attribution_quarantine_missing');
    return;
  }
  const prefix = `physical:${q.run_id}:`;
  const serial = e.entity_id.startsWith(prefix) ? e.entity_id.slice(prefix.length) : '';
  requireValue(/^[1-9][0-9]*$/.test(serial) && BigInt(serial) < 2n ** 64n
    && q.component_entity_ids.includes(e.entity_id)
    && new Set(q.component_entity_ids).size === q.component_entity_ids.length
    && e.subject.subject_id === `quarantine:${q.run_id}:${serial}`
    && e.subject.kind === 'provisional' && e.subject.generation === 0
    && e.subject.stable_id == null && e.subject.resident_uuid == null && e.subject.display_name == null
    && [...q.owners, ...q.retained_owners].every((c) => c.run_id === q.run_id), 'attribution_quarantine_invalid');
  const claims = [...q.owners, ...q.retained_owners];
  const claimId = (c) => `physical:${c.run_id}:${c.physical_serial}`;
  const claimSubject = (c) => c.household_state === 'resident' ? `resident:${c.household_identity_key}`
    : c.household_state === 'visitor' ? `visitor:${c.run_id}:${c.household_stable_id}:g${c.household_visitor_generation}` : null;
  requireValue(claims.every((c) => c.lifecycle_started_frame <= c.frame_id
    && (!['unknown', 'provisional'].includes(c.household_state) || (c.household_identity_key == null
      && c.household_stable_id == null && c.household_visitor_generation == null))
    && (!['resident', 'visitor', 'handoff'].includes(c.household_state) || text(c.household_identity_key))
    && ((c.household_state === 'visitor') === (c.household_visitor_generation != null))), 'attribution_claim_invalid');
  const subjects = new Map(); const stable = new Map(); const latest = new Map();
  for (const c of q.owners) {
    for (const [map, key] of [[subjects, claimSubject(c)], [stable, c.household_stable_id]]) {
      if (key == null) continue;
      if (!map.has(key)) map.set(key, new Set());
      map.get(key).add(claimId(c));
    }
    const member = JSON.stringify([claimId(c), c.camera_id, c.source_id, c.source_epoch,
      c.raw_id, c.lifecycle_generation]);
    if (!latest.has(member) || c.frame_id >= latest.get(member).frame_id) latest.set(member, c);
  }
  const duplicate = (map) => [...map.entries()].filter(([, ids]) => ids.size > 1).map(([key]) => key);
  const contestedSubjects = new Set(duplicate(subjects)); const contestedStable = new Set(duplicate(stable));
  const groups = new Map(); const divergent = new Set();
  for (const c of latest.values()) {
    if (!['resident', 'visitor', 'handoff'].includes(c.household_state) || c.household_constraint !== 'eligible') continue;
    if (!groups.has(claimId(c))) groups.set(claimId(c), []);
    groups.get(claimId(c)).push(c);
  }
  for (const [id, rows] of groups) {
    const signatures = new Set(rows.map((c) => JSON.stringify([c.household_state,
      c.household_identity_key, c.household_stable_id, c.household_visitor_generation])));
    if (signatures.size < 2) continue;
    divergent.add(id);
    for (const c of rows) {
      if (claimSubject(c) !== null) contestedSubjects.add(claimSubject(c));
      if (c.household_stable_id !== null) contestedStable.add(c.household_stable_id);
    }
  }
  const uniqueClaims = (rows) => new Set(rows.map((c) => JSON.stringify(Object.entries(c).sort()))).size === rows.length;
  requireValue(equal(q.subject_ids, [...contestedSubjects].sort())
    && equal(q.stable_ids, [...contestedStable].sort((a, b) => a - b))
    && uniqueClaims(q.owners) && uniqueClaims(q.retained_owners), 'attribution_claims_incomplete');
  if (q.component_entity_ids.length === 1) {
    requireValue(divergent.has(q.component_entity_ids[0])
      && claims.every((c) => claimId(c) === q.component_entity_ids[0]), 'attribution_singleton_invalid');
  }
  requireValue(q.component_entity_ids.every((id) => claims.some((c) => claimId(c) === id)), 'attribution_component_incomplete');
  const remaining = new Set(q.component_entity_ids);
  const reached = new Set([remaining.values().next().value]);
  remaining.delete([...reached][0]);
  while (remaining.size) {
    const known = claims.filter((c) => reached.has(claimId(c)));
    const added = claims.filter((c) => remaining.has(claimId(c)) && known.some((k) =>
      (claimSubject(c) !== null && claimSubject(c) === claimSubject(k))
      || (c.household_stable_id !== null && c.household_stable_id === k.household_stable_id)));
    requireValue(added.length > 0, 'attribution_component_disconnected');
    for (const c of added) { reached.add(claimId(c)); remaining.delete(claimId(c)); }
  }
}
function validateSnapshot(s) {
  schemaValue(s, snapshotSchema, snapshotSchema);
  requireValue(['ds9', 'replay', 'test'].includes(s.producer.runtime), 'runtime_unsupported');
  requireValue(s.entities.length <= UNIFIED_BEV_LIMITS.entities
    && s.observed_start_us <= s.observed_end_us && s.observed_end_us <= s.published_at_us, 'snapshot_time_invalid');
  const ids = new Set(); const subjects = new Set(); const stableIds = new Set();
  for (const e of s.entities) {
    validateAttribution(e);
    requireValue(!ids.has(e.entity_id) && !subjects.has(e.subject.subject_id)
      && (e.subject.stable_id == null || !stableIds.has(e.subject.stable_id)), 'canonical_identity_duplicate');
    ids.add(e.entity_id); subjects.add(e.subject.subject_id);
    if (e.subject.stable_id != null) stableIds.add(e.subject.stable_id);
    requireValue(e.observed_at_us <= s.published_at_us && e.stale_after_us >= e.observed_at_us
      && e.sources.length <= UNIFIED_BEV_LIMITS.evidencePerEntity, 'entity_time_invalid');
    const sourceIds = new Set();
    for (const source of e.sources) {
      requireValue(source.observed_at_us <= s.published_at_us && !sourceIds.has(source.observation_id), 'source_evidence_invalid');
      sourceIds.add(source.observation_id);
      requireValue(source.zone_authoritative !== true || (source.zone != null && source.zone_source === 'nvdsanalytics_roi'), 'zone_evidence_invalid');
      requireValue(source.zone_source !== 'nvdsanalytics_roi' || source.zone_authoritative === true, 'zone_evidence_invalid');
    }
    const rooms = new Set(e.sources.filter((source) => source.accepted && source.zone_authoritative === true
      && source.zone_source === 'nvdsanalytics_roi').map((source) => source.zone));
    requireValue(rooms.size === 1 ? rooms.has(e.room_id) : e.room_id == null, 'room_evidence_invalid');
    requireValue(rooms.size <= 1 || e.conflict === true, 'room_evidence_invalid');
    requireValue(e.subject.kind !== 'resident' || text(e.subject.resident_uuid), 'subject_invalid');
    requireValue(e.subject.kind === 'resident' || e.subject.resident_uuid == null, 'subject_invalid');
    requireValue(!['unknown', 'provisional'].includes(e.subject.kind) || e.subject.display_name == null, 'subject_invalid');
  }
  for (const e of s.entities) {
    if (e.attribution_quarantine == null) continue;
    for (const peer of s.entities.filter((p) => e.attribution_quarantine.component_entity_ids.includes(p.entity_id))) {
      requireValue(equal(peer.attribution_quarantine, e.attribution_quarantine), 'attribution_component_partial');
    }
  }
  return s;
}

export function createUnifiedBevState({ authoredSceneSha256, maxTrailPoints = UNIFIED_BEV_LIMITS.trailPoints } = {}) {
  requireValue(integer(maxTrailPoints, 1) && maxTrailPoints <= UNIFIED_BEV_LIMITS.trailPoints, 'trail_limit_invalid');
  return freeze({ authoredSceneSha256: authoredSceneSha256 ?? null, maxTrailPoints,
    bindings: null, maps: [], people: [], unavailable: [], deadlines: [],
    cursor: null, sourceCursors: [], pending: [], retiredRuns: [], eventSequence: null,
    lastReceivedAtMs: null, diagnostic: { accepted: true, reason: 'initial', type: null } });
}

function sourceDecision(state, cohort, nextRun, sourceEpoch) {
  const prior = state.sourceCursors.find((c) => c.sourceId === cohort.source_id);
  if (!prior) requireValue(state.sourceCursors.length < UNIFIED_BEV_LIMITS.sources, 'source_limit_exceeded');
  if (prior?.run === nextRun) {
    requireValue(sourceEpoch == null || prior.epoch == null || sourceEpoch >= prior.epoch, 'source_epoch_stale');
    const reset = sourceEpoch != null && prior.epoch != null && sourceEpoch > prior.epoch;
    requireValue(reset || cohort.tracking_publication_sequence > prior.sequence, 'source_sequence_stale');
    if (!reset && cohort.frame_id != null && prior.frameId != null) requireValue(cohort.frame_id >= prior.frameId, 'source_frame_stale');
    return reset;
  }
  return false;
}

function projectPerson(bindings, entity) {
  requireValue(bindings && entity.position && entity.covariance, 'world_point_unavailable');
  requireValue(entity.world_frame === WORLD && entity.position_quantity === 'ground_footprint'
    && text(entity.world_frame_revision) && ['measured', 'predicted', 'held'].includes(entity.position_state), 'world_point_provenance_invalid');
  const accepted = entity.sources.filter((s) => s.accepted);
  requireValue(accepted.length > 0 && accepted.every((s) => s.world_frame === WORLD
    && s.world_frame_revision === entity.world_frame_revision && SHA.test(s.world_transform_sha256 ?? '')
    && bindings.cameras[s.camera_id]?.world_frame.revision === s.world_frame_revision
    && bindings.cameras[s.camera_id]?.transform_sha256 === s.world_transform_sha256
    && (entity.world_transform_sha256 == null || s.world_transform_sha256 === entity.world_transform_sha256)
    && (entity.calibration_revision == null || s.calibration_revision === entity.calibration_revision)), 'source_frame_mismatch');
  const hashes = accepted.map((s) => s.world_transform_sha256);
  if (entity.world_transform_sha256 != null) hashes.push(entity.world_transform_sha256);
  const p = presentationFor(bindings, entity.world_frame_revision, hashes);
  const worldPosition = xyz(entity.position);
  const position = point(p.world_to_scene_col_major, worldPosition);
  const m = p.world_to_scene_col_major;
  const linear = [[m[0], m[4], m[8]], [m[1], m[5], m[9]], [m[2], m[6], m[10]]];
  const worldCovariance = [...entity.covariance.values];
  const sceneCovariance = Array.from({ length: 9 }, (_, i) => {
    const r = Math.floor(i / 3); const c = i % 3;
    return linear.reduce((sum, _, j) => sum + linear.reduce((inner, __, k) => (
      inner + linear[r][j] * worldCovariance[j * 3 + k] * linear[c][k]
    ), 0), 0);
  });
  const worldVelocity = entity.velocity_mps ? xyz(entity.velocity_mps) : null;
  const sceneVelocity = worldVelocity ? linear.map((row) => row.reduce((sum, v, i) => sum + v * worldVelocity[i], 0)) : null;
  requireValue(sceneCovariance.every(Number.isFinite) && (!sceneVelocity || sceneVelocity.every(Number.isFinite)), 'uncertainty_transform_invalid');
  return { entityId: entity.entity_id, subjectId: entity.subject.subject_id,
    stableId: entity.subject.stable_id ?? null, identityKind: entity.subject.kind,
    generation: entity.subject.generation ?? 0,
    label: entity.attribution_quarantine ? 'Person (attribution quarantined)'
      : entity.subject.display_name || (entity.subject.stable_id ? `Person ${entity.subject.stable_id}` : 'Person'),
    attributionQuarantine: entity.attribution_quarantine ? copy(entity.attribution_quarantine) : null,
    worldPosition, position, worldCovariance, sceneCovariance, worldVelocity, sceneVelocity,
    supportState: entity.support_state ?? 'unknown', posture: entity.posture ?? 'unknown',
    supportSurface: entity.support_surface ? copy(entity.support_surface) : null,
    coordinateFrame: SCENE, sceneTransformCount: 1,
    lifecycle: entity.lifecycle, positionState: entity.position_state,
    admission: entity.position_state === 'measured' ? 'accepted' : entity.position_state,
    observedAtUs: entity.observed_at_us, staleAfterUs: entity.stale_after_us,
    continuitySegmentId: entity.continuity_segment_id ?? null,
    worldFrameRevision: entity.world_frame_revision, worldTransformSha256: entity.world_transform_sha256 ?? null,
    calibrationRevision: entity.calibration_revision ?? null,
    sceneRevisionId: p.scene_revision_id, sceneMappingSha256: p.world_to_scene_sha256,
    authoredSceneSha256: bindings.authoredSceneSha256, sourceEvidence: copy(entity.sources),
    conflict: entity.conflict ?? false, roomId: entity.room_id ?? null };
}

function acceptSnapshot(state, message, receivedAtMs) {
  const cohort = cohortOf(message);
  const s = validateSnapshot(message.payload);
  const nextRun = runKey(s.producer);
  const pending = state.pending.find((p) => p.cohort.source_id === cohort.source_id);
  if (pending) {
    requireValue(equal(pending.cohort, cohort), 'cohort_mismatch');
    requireValue(equal(pending.snapshot, s), 'snapshot_mismatch');
    requireValue(!pending.accepted, 'snapshot_duplicate');
  }
  const epoch = pending?.sourceEpoch ?? null;
  const sourceReset = sourceDecision(state, cohort, nextRun, epoch);
  const changedRun = state.cursor !== null && state.cursor.run !== nextRun;
  if (changedRun) {
    requireValue(!state.retiredRuns.includes(nextRun), 'retired_run');
    requireValue(state.retiredRuns.length < UNIFIED_BEV_LIMITS.retiredRuns, 'run_limit_exceeded');
    requireValue(s.published_at_us > state.cursor.publishedAtUs, 'stale_run');
  } else if (state.cursor) {
    requireValue(equal(state.cursor.producer, s.producer), 'producer_changed_within_run');
    requireValue(s.sequence > state.cursor.sequence && s.published_at_us >= state.cursor.publishedAtUs, 'snapshot_sequence_stale');
  }
  const priorSource = state.sourceCursors.find((c) => c.sourceId === cohort.source_id);
  const sequenceGap = !changedRun && ((state.cursor && s.sequence !== state.cursor.sequence + 1)
    || (!sourceReset && priorSource?.run === nextRun
      && cohort.tracking_publication_sequence !== priorSource.sequence + 1));
  const people = []; const unavailable = []; const deadlines = [];
  for (const e of s.entities) {
    if (e.lifecycle === 'lost') continue;
    const oldDeadline = !changedRun && state.deadlines.find((d) => d.entityId === e.entity_id);
    let expiresAtMs = receivedAtMs + (e.stale_after_us - s.published_at_us) / 1000;
    // Repeated holds with unchanged original clocks cannot renew receipt age.
    if (oldDeadline && oldDeadline.observedAtUs === e.observed_at_us && oldDeadline.staleAfterUs === e.stale_after_us) {
      expiresAtMs = Math.min(expiresAtMs, oldDeadline.expiresAtMs);
    }
    deadlines.push({ entityId: e.entity_id, observedAtUs: e.observed_at_us, staleAfterUs: e.stale_after_us, expiresAtMs });
    if (expiresAtMs <= receivedAtMs) continue;
    try {
      const projected = projectPerson(state.bindings, e);
      const old = !changedRun && !sourceReset && !sequenceGap && state.people.find((p) => p.entityId === e.entity_id);
      // GlobalWorldFusion.physical_entity_id emits exactly physical:<run>:<serial>.
      // Its independent, run-owned key + explicit backend segment survives an
      // honest household promotion. Legacy/subject-key IDs still bind generation.
      const prefix = `physical:${s.producer.run_id}:`;
      const serial = e.entity_id.startsWith(prefix) ? e.entity_id.slice(prefix.length) : '';
      const physical = integer(Number(serial), 1) && String(Number(serial)) === serial
        && integer(e.continuity_segment_id);
      const trailKey = JSON.stringify([nextRun, e.entity_id, physical ? null : projected.generation, e.continuity_segment_id,
        e.world_frame_revision, e.world_transform_sha256, e.calibration_revision, projected.sceneMappingSha256]);
      const continuous = old && e.continuity_segment_id != null && old.trailKey === trailKey
        && old.observedAtUs <= e.observed_at_us;
      let trail = continuous ? [...old.trail] : [];
      if (e.position_state === 'measured' && (!trail.length || e.observed_at_us > trail.at(-1).observedAtUs)) {
        trail.push({ position: [...projected.position], worldPosition: [...projected.worldPosition],
          observedAtUs: e.observed_at_us, positionState: e.position_state, cohort: { ...cohort },
          worldFrameRevision: e.world_frame_revision, sceneMappingSha256: projected.sceneMappingSha256 });
        trail = trail.slice(-state.maxTrailPoints);
      }
      people.push({ ...projected, expiresAtMs, trailKey, trail, trailBreak: !continuous, cohort: { ...cohort } });
    } catch (error) {
      unavailable.push({ entityId: e.entity_id, stableId: e.subject.stable_id ?? null,
        reason: error.code ?? 'world_point_unavailable', expiresAtMs });
    }
  }
  const priorCursor = state.sourceCursors.find((c) => c.sourceId === cohort.source_id);
  const sourceCursor = { sourceId: cohort.source_id, run: nextRun, sequence: cohort.tracking_publication_sequence,
    frameId: cohort.frame_id, epoch: epoch ?? (priorCursor?.run === nextRun ? priorCursor.epoch : null) };
  const acceptedPending = { cohort, snapshot: copy(s), events: pending?.events ?? null,
    nextEvent: 0, accepted: true, sourceEpoch: epoch };
  const otherPending = state.pending.filter((p) => p.cohort.source_id !== cohort.source_id);
  // An embedded empty event array closes the exact cohort immediately. A
  // midstream standalone snapshot allows bounded diagnostic events until the
  // next tracking/snapshot for that source; events never create person state.
  const nextPending = pending?.events?.length === 0 ? otherPending : [...otherPending, acceptedPending];
  return { ...state, people, unavailable, deadlines, pending: nextPending,
    sourceCursors: [...state.sourceCursors.filter((c) => c.sourceId !== cohort.source_id), sourceCursor],
    retiredRuns: changedRun ? [...state.retiredRuns, state.cursor.run] : state.retiredRuns,
    eventSequence: changedRun ? null : state.eventSequence,
    diagnostic: { accepted: true, reason: sequenceGap ? 'admitted_after_gap' : 'admitted', type: 'world_snapshot' },
    cursor: { run: nextRun, sequence: s.sequence, snapshotId: s.snapshot_id, producer: copy(s.producer),
      publishedAtUs: s.published_at_us, cohort } };
}

/** Pure reducer: the caller supplies a monotonic local receipt clock in ms. */
export function reduceUnifiedBev(state, message, { receivedAtMs } = {}) {
  let next = state;
  try {
    requireValue(Number.isFinite(receivedAtMs) && receivedAtMs >= 0
      && (state.lastReceivedAtMs == null || receivedAtMs >= state.lastReceivedAtMs), 'receipt_clock_invalid');
    next = advanceUnifiedBev(state, receivedAtMs);
    const type = message?.type;
    let reason = 'admitted';
    if (type === 'calibration-bundle') {
      const bindings = createUnifiedBevBindings(message, state.authoredSceneSha256);
      if (!equal(bindings, state.bindings)) next = { ...next, bindings, maps: [], people: [], unavailable: [] };
    } else if (type === 'floorplan_response') {
      const map = projectUnifiedBevMap(state.bindings, message);
      requireValue(next.maps.some((m) => m.cameraId === map.cameraId) || next.maps.length < UNIFIED_BEV_LIMITS.cameras, 'map_limit_exceeded');
      next = { ...next, maps: [...next.maps.filter((m) => m.cameraId !== map.cameraId), map] };
    } else if (type === 'tracking') {
      const cohort = cohortOf(message);
      const snapshot = message.world_snapshot;
      requireValue(snapshot === null || record(snapshot), 'embedded_snapshot_missing');
      if (snapshot) validateSnapshot(snapshot);
      requireValue(Array.isArray(message.world_events) && message.world_events.length <= UNIFIED_BEV_LIMITS.eventsPerCohort
        && (snapshot !== null || message.world_events.length === 0), 'embedded_events_invalid');
      for (const event of message.world_events) {
        schemaValue(event, eventSchema, eventSchema);
        validateAttribution(event);
      }
      const epoch = message.source_epoch;
      requireValue(integer(epoch), 'source_epoch_invalid');
      const nextRun = snapshot ? runKey(snapshot.producer) : (next.cursor?.run ?? 'unbound');
      sourceDecision(next, cohort, nextRun, epoch);
      const previous = next.pending.find((p) => p.cohort.source_id === cohort.source_id);
      requireValue(!previous || (previous.accepted && (previous.events === null || previous.snapshot === null)), 'incomplete_cohort');
      const other = next.pending.filter((p) => p.cohort.source_id !== cohort.source_id);
      requireValue(other.length < UNIFIED_BEV_LIMITS.sources, 'source_limit_exceeded');
      next = { ...next, pending: [...other, { cohort, snapshot: snapshot ? copy(snapshot) : null,
        events: copy(message.world_events), nextEvent: 0, accepted: false, sourceEpoch: epoch }] };
      if (snapshot === null) {
        const prior = next.sourceCursors.find((c) => c.sourceId === cohort.source_id);
        next = { ...next, pending: [...other, { cohort, snapshot: null, events: [], nextEvent: 0,
          accepted: true, sourceEpoch: epoch }], sourceCursors: [
          ...next.sourceCursors.filter((c) => c.sourceId !== cohort.source_id),
          { sourceId: cohort.source_id, run: nextRun, sequence: cohort.tracking_publication_sequence,
            frameId: cohort.frame_id, epoch: epoch ?? prior?.epoch ?? null },
        ] };
      }
    } else if (type === 'world_snapshot') {
      next = acceptSnapshot(next, message, receivedAtMs);
      reason = next.diagnostic.reason;
    } else if (type === 'world_event') {
      const cohort = cohortOf(message);
      const event = message.payload;
      schemaValue(event, eventSchema, eventSchema);
      validateAttribution(event);
      const p = next.pending.find((item) => item.cohort.source_id === cohort.source_id);
      requireValue(p?.accepted && equal(p.cohort, cohort), 'event_without_exact_snapshot');
      requireValue(equal(event.producer, p.snapshot.producer) && event.published_at_us === p.snapshot.published_at_us
        && event.observed_at_us <= event.published_at_us, 'event_producer_mismatch');
      requireValue(next.eventSequence == null || event.sequence > next.eventSequence, 'event_sequence_stale');
      requireValue(p.nextEvent < UNIFIED_BEV_LIMITS.eventsPerCohort
        && (p.events === null || equal(p.events[p.nextEvent], event)), 'event_mismatch');
      const updated = { ...p, nextEvent: p.nextEvent + 1 };
      next = { ...next, eventSequence: event.sequence,
        pending: next.pending.filter((item) => item !== p).concat(
          p.events !== null && updated.nextEvent === p.events.length ? [] : [updated]) };
    } else {
      throw new UnifiedBevError('noncanonical_input');
    }
    return freeze({ ...next, lastReceivedAtMs: receivedAtMs, diagnostic: { accepted: true, reason, type } });
  } catch (error) {
    // A malformed map/binding must not leave old geometry as current truth.
    const clearBinding = message?.type === 'calibration-bundle';
    const maps = message?.type === 'floorplan_response'
      ? next.maps.filter((m) => m.cameraId !== message.camera_id) : next.maps;
    return freeze({ ...next, ...(clearBinding ? { bindings: null, maps: [], people: [], unavailable: [] } : { maps }),
      diagnostic: { accepted: false, reason: error.code ?? 'invalid_input', type: message?.type ?? null } });
  }
}

export function advanceUnifiedBev(state, nowMs) {
  requireValue(Number.isFinite(nowMs) && nowMs >= 0
    && (state.lastReceivedAtMs == null || nowMs >= state.lastReceivedAtMs), 'receipt_clock_invalid');
  return freeze({ ...state, lastReceivedAtMs: nowMs,
    people: state.people.filter((p) => p.expiresAtMs > nowMs),
    unavailable: state.unavailable.filter((p) => p.expiresAtMs > nowMs) });
}

/** Disconnect clears people/trails but retains replay fences and static maps. */
export function clearUnifiedBevPeople(state, reason = 'disconnected') {
  return freeze({ ...state, people: [], unavailable: [], pending: [],
    diagnostic: { accepted: true, reason, type: null } });
}

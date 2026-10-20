/** Camera recording selection only; never geometry, identity or synchronization authority. */
const escape = (value) => String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const idFor = (camera) => typeof camera === "string" ? camera : camera?.camera_id || camera?.id || "";

export function cameraSelection({ cameras = [], selectedIds = [], primaryId = "", maxCameras = 1, supportedMax = 2, locked = false } = {}) {
  const advertisedMax = Number.isInteger(maxCameras) && maxCameras >= 0 ? maxCameras : 1;
  const maximum = Math.min(advertisedMax, supportedMax, 2); // Existing client/backend contract remains <=2.
  const ids = [...new Set(selectedIds.filter((id) => typeof id === "string" && id))];
  const rows = cameras.map((camera) => ({ id: idFor(camera), label: typeof camera === "string" ? camera : camera?.label || camera?.name || idFor(camera), available: camera?.available === true })).filter((camera) => camera.id);
  // A retained choice disappearing from inventory is visible and removable, never silently lost.
  for (const id of ids) if (!rows.some((row) => row.id === id)) rows.push({ id, label: id, available: false });
  return { cameras: rows, ids, primaryId: ids.includes(primaryId) ? primaryId : ids[0] || "", maximum, advertisedMax, locked: Boolean(locked) };
}

export function changeCameraSelection(state, { cameraId, checked, primaryId } = {}) {
  if (state.locked) throw new Error("Camera selection is locked while recording or waiting for setup acknowledgement.");
  if (primaryId !== undefined) {
    if (!state.ids.includes(primaryId)) throw new Error("The primary reference must be one of the checked cameras.");
    return { ...state, ids: state.ids.slice(), primaryId };
  }
  const row = state.cameras.find((camera) => camera.id === cameraId);
  if (!row) throw new Error("This camera is not in the current inventory.");
  const ids = state.ids.filter((id) => id !== cameraId);
  if (checked) {
    if (!row.available) throw new Error("This camera is unavailable.");
    if (!state.ids.includes(cameraId) && state.ids.length >= state.maximum) throw new Error(`At most ${state.maximum} cameras can be recorded.`);
    ids.push(cameraId);
  }
  return { ...state, ids, primaryId: ids.includes(state.primaryId) ? state.primaryId : ids[0] || "" };
}

export function cameraSelectionMarkup(state, { fieldsetId, primarySelectId }) {
  const rows = state.cameras.map((camera) => {
    const checked = state.ids.includes(camera.id);
    const disabled = state.locked || !checked && (!camera.available || state.ids.length >= state.maximum);
    const reason = !camera.available ? "Unavailable" : !checked && state.ids.length >= state.maximum ? "Recording limit reached" : "";
    return `<label class="camera-recording-option${disabled ? " is-disabled" : ""}"><input type="checkbox" data-camera-id="${escape(camera.id)}" ${checked ? "checked" : ""} ${disabled ? "disabled" : ""}><span>${escape(camera.label)}${reason ? `<small>${escape(reason)}</small>` : ""}</span></label>`;
  }).join("");
  const options = state.ids.length ? state.ids.map((id) => `<option value="${escape(id)}" ${id === state.primaryId ? "selected" : ""}>${escape(state.cameras.find((row) => row.id === id)?.label || id)}</option>`).join("") : '<option value="">No static camera · phone-only reconstruction</option>';
  return `<fieldset id="${escape(fieldsetId)}" class="camera-recording-list"><legend>Static cameras to record</legend><p class="camera-selection-limit">${state.ids.length} checked · server limit ${state.advertisedMax}${state.maximum < state.advertisedMax ? ` · this recorder supports ${state.maximum}` : ""}</p>${rows || '<p class="tool-note">Connect to load camera availability.</p>'}</fieldset><label class="camera-primary-reference" for="${escape(primarySelectId)}">Primary alignment reference<select id="${escape(primarySelectId)}" data-camera-primary class="text-input" ${state.locked || !state.ids.length ? "disabled" : ""}>${options}</select><small>This chooses the reconstruction reference only; it does not change which checked cameras record.</small></label>`;
}

export function nativeCameraSelection(snapshot = {}) {
  const cameras = snapshot.room_cameras || [];
  const primaryId = idFor(cameras[snapshot.room_camera_index]);
  const extraId = idFor(cameras[snapshot.extra_room_camera_index]) || snapshot.requested_extra_room_camera_id || "";
  return cameraSelection({ cameras, selectedIds: [primaryId, extraId].filter(Boolean), primaryId,
    maxCameras: snapshot.max_cameras ?? (snapshot.multi_camera_capture === true ? 2 : 1),
    supportedMax: snapshot.supports_camera_ids === true && snapshot.multi_camera_capture === true ? 2 : 1,
    locked: snapshot.active || snapshot.busy || snapshot.imu_active });
}

export function nativeCameraIndices(state, cameras) {
  if (state.ids.length > state.maximum || state.ids.length > 2) throw new Error("The checked camera set exceeds this recorder's supported limit.");
  if (!state.ids.length) return { room_camera_index: -1, extra_room_camera_index: -1 };
  if (!state.ids.includes(state.primaryId)) throw new Error("Choose a checked camera as the primary reference.");
  const index = (id) => {
    const value = cameras.findIndex((camera) => idFor(camera) === id);
    if (value < 0 || cameras[value].available !== true) throw new Error("A checked camera is unavailable; reconnect it or explicitly uncheck it.");
    return value;
  };
  const extra = state.ids.find((id) => id !== state.primaryId);
  return { room_camera_index: index(state.primaryId), extra_room_camera_index: extra ? index(extra) : -1 };
}

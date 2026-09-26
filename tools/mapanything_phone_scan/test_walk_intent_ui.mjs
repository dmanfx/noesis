import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

const root = new URL(".", import.meta.url);
const readStatic = (name) => readFile(new URL(`./static/${name}`, root), "utf8");

test("RoomWalk presents Capture, Library and Setup as the primary workspace", async () => {
  const html = await readStatic("index.html");
  assert.match(html, /data-workspace-tab="capture"/);
  assert.match(html, /data-workspace-tab="library"/);
  assert.match(html, /data-workspace-tab="setup"/);
  assert.doesNotMatch(html, /data-workspace-tab="calibration"/);
  assert.match(html, /id="capture-mode-reconstruction"/);
  assert.match(html, /id="capture-mode-path"/);
  assert.match(html, /id="path-target-scan"/);
  assert.match(html, /id="path-camera-select"/);
  assert.match(html, /10 cm is a validation target, not proof/);
  assert.match(html, /against your own torso/);
  assert.match(html, /elbows tucked and stable/);
  assert.match(html, /move your whole body with the phone/);
  assert.match(html, /Do not film another person/);
  assert.match(html, /vary phone-to-body distance/);
  assert.match(html, /id="calibration-panel"/);
  assert.match(html, /id="panel-setup"/);
});

test("walk intent and review actions use the shared contract without auto-mixing products", async () => {
  const app = await readStatic("app.js");
  const native = await readStatic("native_capture.js");
  const calibration = await readStatic("calibration.js");
  const comparison = await readStatic("path_comparison.js");
  for (const field of ["schema", "mode", "target_scan_id", "carry_protocol", "accuracy_target_m"]) assert.match(app, new RegExp(`${field}`));
  assert.match(app, /roomwalk\.walk_intent\.v1/);
  assert.match(app, /target_reference/);
  assert.match(app, /pathReferenceForScan/);
  assert.match(app, /reference_pcf_manifest/);
  assert.match(app, /reference_pcf_points/);
  assert.match(app, /reference_pcf_binding/);
  assert.match(app, /\/api\/scans\/\$\{encodeURIComponent\(scanId\)\}\/walk-intent/);
  assert.match(app, /\/supplements\/from-scan\?source_scan_id=/);
  assert.match(app, /\/path-review\$\{targetQuery\}/);
  assert.match(app, /!pathReviewRunning/);
  assert.match(app, /Export path review/);
  assert.match(app, /artifact_urls/);
  assert.match(comparison, /Phone optical center vs Noesis ground point/);
  assert.match(comparison, /UI never assigns the phone carrier automatically/);
  assert.match(native, /mode === "walk" \? "reconstruction"/);
  assert.match(native, /path_refinement/);
  assert.match(native, /supports_target_reference/);
  assert.match(native, /validatePathReferenceSelection/);
  assert.match(native, /against your own torso/);
  assert.match(native, /sweep the phone with your arm/);
  assert.match(native, /walk a nearby body route/);
  assert.match(calibration, /id="calibration-secondary"/);
  assert.match(calibration, /Optional setup/);
});

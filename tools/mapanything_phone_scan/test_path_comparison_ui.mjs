import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import { access } from "node:fs/promises";
import { createRequire } from "node:module";

import {
  buildTopdownGeometry,
  formatSeparation,
  normalizePathComparisonPayload,
  pathComparisonSummary,
} from "./static/path_comparison.js";

let chromium;
try {
  ({ chromium } = createRequire(import.meta.url)(process.env.ROOMWALK_PLAYWRIGHT_MODULE || "playwright"));
} catch (_) {
  // The pure module checks below remain useful without a browser dependency.
}

const smokePath = process.env.ROOMWALK_PATH_COMPARISON_FIXTURE
  || "/tmp/roomwalk-path-light-smoke-5pgz8pf7/path_comparison.json";

async function smokePayload() {
  try {
    await access(smokePath);
    return JSON.parse(await readFile(smokePath, "utf8"));
  } catch (_) {
    return {
      schema: "noesis.phone_scan.path_comparison.v1",
      status: "comparison_ready",
      registration: { status: "registered" },
      timing: { status: "light_cue_review", method: "fixture", limitation: "fixture timing" },
      accuracy_qualified: false,
      reference: { poses: [
        { provider_index: 0, phone_time_s: 0, original_world_m: [0, 0, 0], reference_world_m: [0, 0, 0], break_before: true },
        { provider_index: 1, phone_time_s: 1, original_world_m: [1, 0, 0], reference_world_m: [1.1, 0, 0], break_before: false },
        { provider_index: 2, phone_time_s: 2, original_world_m: [2, 0, 0], reference_world_m: [2.1, 0, 0], break_before: true },
      ] },
      tracks: [{ track_key: "camera:source:7:3", tracker_id: 7, lifecycle_generation: 3, stable_ids: [12, 14], matched_count: 2, current_world_observation_count: 2, invalid_world_count: 0, separation_m: { median: 0.3 }, pairs: [
        { provider_index: 0, phone_time_s: 0, reference_world_m: [0, 0, 0], noesis_world_m: [0.2, 0, 0], horizontal_separation_m: 0.2, original_horizontal_separation_m: 0.2, stable_id: 12, break_before: true, trail_segment_id: "a" },
        { provider_index: 1, phone_time_s: 1, reference_world_m: [1.1, 0, 0], noesis_world_m: [1.4, 0, 0], horizontal_separation_m: 0.3, original_horizontal_separation_m: 0.3, stable_id: 14, break_before: true, trail_segment_id: "b" },
      ] }],
    };
  }
}

test("path comparison normalizes the compact result without assigning a person", async () => {
  const payload = await smokePayload();
  const summary = pathComparisonSummary({
    results: {
      path_comparison: {
        status: payload.status,
        reason: "backend reason",
        registration: payload.registration,
        timing: payload.timing,
        track_count: payload.tracks.length,
        matched_count: payload.tracks.reduce((sum, track) => sum + track.matched_count, 0),
      },
      path_refinement: { status: "available", position_refined: true, method: "fused_vio" },
    },
  });
  assert.equal(summary.status, "comparison_ready");
  assert.equal(summary.trackCount, 0, "compact summaries do not invent full tracks");
  assert.equal(summary.matchedCount, 0, "compact summaries do not invent pair rows");
  const full = normalizePathComparisonPayload(payload);
  assert.equal(full.reference.poses.length > 0, true);
  assert.equal(full.tracks.length > 0, true);
  assert.equal(full.accuracyQualified, false);
  assert.equal(full.pathRefinement.position_refined ?? false, false);
});

test("top-down geometry keeps original/reference paths and lifecycle gap breaks separate", async () => {
  const payload = await smokePayload();
  const trackKey = payload.tracks[0].track_key;
  const selectedIndex = payload.tracks[0].pairs[0].provider_index;
  const geometry = buildTopdownGeometry(payload, trackKey, selectedIndex);
  assert.ok(geometry.originalSegments.length >= 1);
  assert.ok(geometry.referenceSegments.length >= 1);
  assert.equal(geometry.selectedTrack.track_key, trackKey);
  assert.equal(geometry.selectedPoseIndex, selectedIndex);
  assert.ok(geometry.pair);
  assert.match(formatSeparation(geometry.pair.horizontal_separation_m), /Separation .* m/);
  assert.ok(geometry.selectedTrackSegments.length >= 1);
  assert.ok(payload.tracks[0].pairs.some((pair) => pair.break_before), "real smoke data exposes lifecycle/stable-ID breaks");
});

test("null values stay missing and the top-down map uses equal x/z scale", () => {
  const geometry = buildTopdownGeometry({
    status: "comparison_ready",
    reference: { poses: [
      { provider_index: 0, original_world_m: null, reference_world_m: [0, 0, 0], break_before: true },
      { provider_index: 1, original_world_m: [2, 0, 0], reference_world_m: [2, 0, 2], break_before: false },
    ] },
    tracks: [],
  });
  assert.equal(geometry.originalSegments.length, 1);
  assert.equal(formatSeparation(null), "Separation —");
  const origin = geometry.map([0, 0]);
  const oneX = geometry.map([1, 0]);
  const oneZ = geometry.map([0, 1]);
  assert.ok(Math.abs(Math.abs(oneX[0] - origin[0]) - Math.abs(oneZ[1] - origin[1])) < 1e-9);
});

test("path comparison module never uses untrusted HTML interpolation", async () => {
  const source = await readFile(new URL("./static/path_comparison.js", import.meta.url), "utf8");
  assert.doesNotMatch(source, /innerHTML/);
  assert.doesNotMatch(source, /insertAdjacentHTML/);
  const payload = await smokePayload();
  payload.tracks[0].track_key = '<img src=x onerror="alert(1)">';
  const normalized = normalizePathComparisonPayload(payload);
  assert.equal(normalized.tracks[0].track_key, '<img src=x onerror="alert(1)">');
});

test("real path comparison smoke renders a map, leaves track choice empty, and supports selection", { skip: !chromium }, async (t) => {
  const payload = await smokePayload();
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.ROOMWALK_CHROMIUM_EXECUTABLE || chromium.executablePath(),
    args: ["--disable-gpu"],
  });
  t.after(() => browser.close());
  const page = await browser.newPage({ viewport: { width: 412, height: 915 } });
  const moduleSource = await readFile(new URL("./static/path_comparison.js", import.meta.url), "utf8");
  await page.route("**/*", async (route) => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname === "/") return route.fulfill({ contentType: "text/html", body: "<!doctype html><main><div id=host></div></main>" });
    if (pathname === "/static/path_comparison.js") return route.fulfill({ contentType: "text/javascript", body: moduleSource });
    if (pathname === "/path_comparison.json") return route.fulfill({ contentType: "application/json", body: JSON.stringify(payload) });
    return route.abort();
  });
  await page.goto("https://roomwalk.test/");
  const result = await page.evaluate(async () => {
    const { createPathComparisonController } = await import("/static/path_comparison.js");
    const host = document.querySelector("#host");
    const controller = createPathComparisonController();
    await controller.render({
      host,
      scanId: "smoke-scan",
      summary: { status: "comparison_ready", registration: { status: "registered" }, timing: { status: "light_cue_review" } },
      pathRefinement: { status: "rejected", position_refined: false, reason: "heldout gate 0.417 m > 0.35 m", accepted_visual: 10, accepted_vio: 0 },
      artifactUrls: { comparison: "/path_comparison.json" },
    });
    return {
      trackValue: host.querySelector("select")?.value,
      trackOptions: host.querySelectorAll("select option").length,
      trackLinesBeforePick: host.querySelectorAll(".path-line-track").length,
      breakCount: host.querySelectorAll(".path-comparison-break").length,
      text: host.textContent,
      hasMap: Boolean(host.querySelector("svg.path-comparison-map")),
    };
  });
  assert.equal(result.hasMap, true);
  assert.equal(result.trackValue, "", "the UI requires an explicit lifecycle selection");
  assert.ok(result.trackOptions >= 2);
  assert.equal(result.trackLinesBeforePick, 0);
  assert.ok(result.breakCount > 0);
  assert.match(result.text, /Separation|Choose a Noesis lifecycle/);
  assert.match(result.text, /Position refinement rejected/);
  assert.match(result.text, /10 accepted/);
  assert.match(result.text, /0 accepted/);
  await page.locator("select").selectOption({ index: 1 });
  assert.ok(await page.locator(".path-line-track").count() > 0);
  const slider = page.locator("input[type=range]");
  await slider.evaluate((input) => { input.value = input.max; input.dispatchEvent(new Event("input", { bubbles: true })); });
  assert.equal(await slider.count(), 1, "the range input stays mounted during thumb movement");
  await slider.dispatchEvent("change");
  assert.match(await page.locator(".path-comparison-selection").innerText(), /Phone time/);
});

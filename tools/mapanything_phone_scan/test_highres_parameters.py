"""CPU-only coverage for opt-in conditioned MapAnything preprocessing."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from tools.mapanything_phone_scan import run_mapanything_prior_variants as runner
from tools.mapanything_phone_scan.inference import MapAnythingScanError


def _cpu_torch():
    return SimpleNamespace(
        inference_mode=nullcontext,
        cuda=SimpleNamespace(is_available=lambda: False),
    )


def test_preprocessing_options_default_leaves_official_mapping_untouched():
    assert runner._preprocessing_options(None) == {}


@pytest.mark.parametrize("process_res", [294, 308, 504, 756, 994, 1008])
def test_preprocessing_options_accepts_patch_aligned_resolution(process_res):
    assert runner._preprocessing_options(process_res) == {
        "resize_mode": "longest_side",
        "size": process_res,
    }


@pytest.mark.parametrize(
    "process_res",
    [True, False, 756.0, "756", 0, -14, 280, 295, 757, 1009, 1022],
)
def test_preprocessing_options_rejects_invalid_resolution(process_res):
    with pytest.raises(MapAnythingScanError, match="--process-res"):
        runner._preprocessing_options(process_res)


def _prior_fixture(shape_hw):
    height, width = shape_hw
    # Removing one 14-pixel patch loses 14 of the original 500 samples.
    yy, xx = np.indices((height, width + 14))
    reliable = (2.0 + yy * 0.001 + xx * 0.0001).astype(np.float32)
    sparse = np.zeros_like(reliable)
    sparse.flat[:500] = reliable.flat[:500]
    image = np.stack(
        (xx % 251, yy % 253, (xx + yy) % 255), axis=-1
    ).astype(np.uint8)
    raw = []
    for index, depth in enumerate((sparse, reliable)):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [index + 1.0, -2.0, 3.0]
        raw.append({
            "img": (image + index).astype(np.uint8),
            "intrinsics": np.asarray(
                [[width * 1.1, 0.0, (width + 14) / 2.0],
                 [0.0, height * 1.2, height / 2.0],
                 [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            "camera_poses": pose,
            "depth_z": depth.copy(),
        })
    priors = [reliable.copy() for _ in raw]
    # Even fake preprocessing must not mutate the source RGB/cameras/depths.
    for array in [value for row in raw for value in row.values()] + priors:
        array.setflags(write=False)
    return raw, priors


def _crop_inputs(views):
    rows = []
    for view in views:
        intrinsics = view["intrinsics"].copy()
        intrinsics[0, 2] -= 14
        rows.append({
            "img": np.moveaxis(view["img"][:, 14:], -1, 0)[None].copy(),
            "intrinsics": intrinsics,
            "camera_poses": view["camera_poses"].copy(),
            "depth_z": view["depth_z"][None, :, 14:].copy(),
        })
    return rows


def _exercise_top_up(*, overrides, expected_options, shape_hw):
    raw, reliable = _prior_fixture(shape_hw)
    originals = [{key: value.copy() for key, value in row.items()} for row in raw]
    expected = _crop_inputs(raw)
    reliable_processed = _crop_inputs([
        {**row, "depth_z": depth} for row, depth in zip(raw, reliable, strict=True)
    ])
    calls = []

    def record_preprocessing(views, *, verbose, options):
        assert options == expected_options
        rows = _crop_inputs(views)
        calls.append({
            "verbose": verbose,
            "options": options.copy(),
            "counts": [int(np.count_nonzero(row["depth_z"])) for row in rows],
        })
        for row, original in zip(rows, expected, strict=True):
            for key in ("img", "intrinsics", "camera_poses"):
                np.testing.assert_array_equal(row[key], original[key])
        return rows

    if expected_options:
        def preprocess(views, *, verbose=False, **options):
            return record_preprocessing(views, verbose=verbose, options=options)
    else:
        # A strict legacy signature detects any extra options, even None values.
        def preprocess(views, *, verbose=False):
            return record_preprocessing(views, verbose=verbose, options={})

    class Model:
        call_count = 0

        def infer(self, processed, **options):
            self.call_count += 1
            assert options == {
                "memory_efficient_inference": True,
                "use_amp": True,
                "amp_dtype": "bf16",
                "apply_mask": True,
                "mask_edges": True,
                "apply_confidence_mask": False,
                "confidence_percentile": 10,
            }
            assert [np.count_nonzero(row["depth_z"]) for row in processed] == [
                500, shape_hw[0] * shape_hw[1],
            ]
            for row, before, evidence in zip(
                processed, expected, reliable_processed, strict=True
            ):
                for key in ("img", "intrinsics", "camera_poses"):
                    np.testing.assert_array_equal(row[key], before[key])
                existing = before["depth_z"] > 0
                accepted = row["depth_z"] > 0
                np.testing.assert_array_equal(
                    row["depth_z"][existing], before["depth_z"][existing]
                )
                np.testing.assert_array_equal(
                    row["depth_z"][accepted], evidence["depth_z"][accepted]
                )
            return [{"depth_z": row["depth_z"].copy(), "mask": row["depth_z"] > 0}
                    for row in processed]

    model = Model()
    outputs, retained, report = runner._run_conditioned_mapanything_windows(
        model, raw, preprocess_inputs=preprocess, torch=_cpu_torch(),
        amp_dtype="bf16", max_joint_views=16, window_overlap_views=4,
        require_depth_prior=True, reliable_depth_priors=reliable,
        **overrides,
    )
    assert model.call_count == 1
    assert len(outputs) == len(retained) == len(raw) == 2
    assert calls == [
        {"verbose": True, "options": expected_options,
         "counts": [486, shape_hw[0] * shape_hw[1]]},
        {"verbose": False, "options": expected_options,
         "counts": [shape_hw[0] * shape_hw[1]] * 2},
    ]
    for prediction, prior in zip(outputs, retained, strict=True):
        assert np.count_nonzero(prior["depth_z"]) >= 500
        np.testing.assert_array_equal(prediction["depth_z"], prior["depth_z"])
    for row, original in zip(raw, originals, strict=True):
        for key in original:
            np.testing.assert_array_equal(row[key], original[key])
    assert report["preprocessing"] == expected_options
    assert report["processed_shape_hw"] == list(shape_hw)
    assert report["max_joint_views"] == 16
    assert report["window_overlap_views"] == 4
    assert report["window_count"] == 1
    assert report["all_duplicate_depth_gates_passed"] is True
    top_up = report["windows"][0]["processed_sparse_prior_top_up"]
    assert len(top_up) == 1
    assert top_up[0]["global_view_index"] == 0
    assert top_up[0]["samples_before"] == 486
    assert top_up[0]["samples_after"] == 500
    assert top_up[0]["added_reliable_samples"] == 14
    assert top_up[0]["existing_samples_rgb_and_camera_unchanged"] is True


def test_highres_options_reach_initial_and_reliable_replay_before_model():
    _exercise_top_up(
        overrides={"process_res": 756},
        expected_options={"resize_mode": "longest_side", "size": 756},
        shape_hw=(420, 756),
    )


@pytest.mark.parametrize(
    "overrides", [{}, {"process_res": None}], ids=["omitted", "explicit-none"]
)
def test_baseline_preprocessing_and_top_up_receive_no_extra_options(overrides):
    _exercise_top_up(overrides=overrides, expected_options={}, shape_hw=(294, 518))


@pytest.mark.parametrize("process_res", [True, 280, 295, 1022, 756.0, "756"])
def test_invalid_override_fails_before_preprocessing_or_model(process_res):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid resolution must fail before preprocessing/inference")

    with pytest.raises(MapAnythingScanError, match="--process-res"):
        runner._run_conditioned_mapanything_windows(
            SimpleNamespace(infer=unexpected), [{}, {}],
            preprocess_inputs=unexpected, torch=_cpu_torch(), amp_dtype="bf16",
            max_joint_views=16, window_overlap_views=4, require_depth_prior=True,
            process_res=process_res,
        )


@pytest.mark.parametrize("process_res", [None, 756])
def test_cli_process_res_is_opt_in_without_changing_other_defaults(
    monkeypatch, process_res
):
    argv = [
        "prior-variants", "prepared-scan", "--da3-raw", "da3/raw",
        "--output-root", "new-run",
    ]
    if process_res is not None:
        argv.extend(["--process-res", str(process_res)])
    monkeypatch.setattr(runner.sys, "argv", argv)
    args = runner._parse_args()
    assert args.process_res == process_res
    assert args.max_joint_views == 80
    assert args.window_overlap_views == 24
    assert args.consistency_threshold == 0.55
    assert args.boundary_threshold == 0.50
    assert args.sparse_sample_fraction == 0.10
    assert args.amp_dtype == "bf16"
    assert args.model_id == "facebook/map-anything-apache"

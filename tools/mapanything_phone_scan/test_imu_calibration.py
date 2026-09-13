from __future__ import annotations

import hashlib
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from .app import create_app
from .imu_calibration import HEADER, SCHEMA
from .test_browser_capture import _settings


def _bundle(*, status="partial", mutate=None, corrupt=None, extra=None):
    csv = (','.join(HEADER) + '\n' + '\n'.join(
        f'{1_000_000_000 + i * 5_000_000},0.01,0.02,9.8,0,0,0,3,{1_000_100_000 + i * 5_000_000}'
        for i in range(5)) + '\n').encode()
    stream = {"sensor_id": "test-sensor", "uncalibrated": True, "sample_count": 5,
              "first_timestamp_ns": 1_000_000_000, "last_timestamp_ns": 1_020_000_000,
              "maximum_interval_ns": 5_000_000, "nonmonotonic_timestamp_count": 0,
              "unreliable_accuracy_sample_count": 0, "bytes": len(csv), "sha256": hashlib.sha256(csv).hexdigest()}
    manifest = {"schema": SCHEMA, "capture_id": "synthetic-test", "clock": "android.elapsedRealtimeNanos",
                "timestamp_unit": "ns", "axes": "android_device_x_right_y_up_z_out_of_screen",
                "device": {"id": "test", "model": "test", "build_fingerprint": "test", "android_api_level": 36},
                "status": status, "actual_duration_s": 0.02, "expected_duration_s": 10800, "dropped_records": 0,
                "streams": {"accelerometer": {**stream, "file": "accel.csv", "units": "m/s^2"},
                            "gyroscope": {**stream, "file": "gyro.csv", "units": "rad/s"}}}
    if mutate:
        mutate(manifest)
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("imu_capture_manifest.json", json.dumps(manifest))
        archive.writestr("accel.csv", corrupt if corrupt is not None else csv)
        archive.writestr("gyro.csv", csv)
        if extra:
            archive.writestr(extra, b"no")
    return result.getvalue(), csv


def test_partial_imu_capture_is_retained_exactly_without_scan_or_calibration(tmp_path):
    settings = _settings(tmp_path)
    raw, csv = _bundle()
    with TestClient(create_app(settings)) as client:
        for _ in range(2):
            response = client.post('/api/phone-calibration/imu-bundle', content=raw, headers={'Content-Type': 'application/zip'})
            assert response.status_code == 201, response.text
            report = response.json()
            assert report['status'] == 'stored'
            assert report['recording_status'] == 'partial'
            assert report['imu_noise_calibrated'] is False
            assert report['acquisition_issues'] == ['recording_partial', 'less_than_three_hours']
            assert report['streams']['accelerometer']['observed_rate_hz'] == 200
        assert client.get('/api/scans').json() == []
    directory = settings.storage_root / '.imu-calibration' / 'synthetic-test'
    assert (directory / 'capture_bundle.zip').read_bytes() == raw
    assert (directory / 'accel.csv').read_bytes() == csv
    assert len(list(directory.parent.iterdir())) == 1


@pytest.mark.parametrize('mutate', [
    lambda m: m.update(capture_id='../escape'),
    lambda m: m.update(clock='browser.performance'),
    lambda m: m.update(status='recording'),
    lambda m: m['streams']['accelerometer'].update(units='g'),
    lambda m: m['streams']['gyroscope'].update(sample_count=6),
    lambda m: m['streams']['gyroscope'].update(sha256='0' * 64),
    lambda m: m.update(dropped_records=True),
])
def test_invalid_imu_manifest_does_not_persist_capture(tmp_path, mutate):
    settings = _settings(tmp_path)
    raw, _ = _bundle(mutate=mutate)
    with TestClient(create_app(settings)) as client:
        response = client.post('/api/phone-calibration/imu-bundle', content=raw, headers={'Content-Type': 'application/zip'})
        assert response.status_code == 422
    assert list((settings.storage_root / '.imu-calibration').iterdir()) == []


@pytest.mark.parametrize('kwargs', [
    {'corrupt': b'invalid\n'}, {'extra': '../escape'}, {'extra': 'gyro.csv'},
    {'corrupt': (','.join(HEADER) + '\n' + f'{2**64},0,0,9.8,0,0,0,3,{2**64}\n').encode()},
    {'corrupt': (','.join(HEADER) + '\n' + '1,nan,0,9.8,0,0,0,3,2\n').encode()},
])
def test_corrupt_or_extra_archive_member_is_rejected(tmp_path, kwargs):
    settings = _settings(tmp_path)
    raw, _ = _bundle(**kwargs)
    with TestClient(create_app(settings)) as client:
        response = client.post('/api/phone-calibration/imu-bundle', content=raw, headers={'Content-Type': 'application/zip'})
        assert response.status_code == 422
    assert list((settings.storage_root / '.imu-calibration').iterdir()) == []


def test_existing_capture_cannot_be_overwritten(tmp_path):
    settings = _settings(tmp_path)
    original, _ = _bundle()
    different, _ = _bundle(mutate=lambda m: m.update(stop_reason='different retained evidence'))
    with TestClient(create_app(settings)) as client:
        assert client.post('/api/phone-calibration/imu-bundle', content=original, headers={'Content-Type': 'application/zip'}).status_code == 201
        assert client.post('/api/phone-calibration/imu-bundle', content=different, headers={'Content-Type': 'application/zip'}).status_code == 422
    assert (settings.storage_root / '.imu-calibration/synthetic-test/capture_bundle.zip').read_bytes() == original


@pytest.mark.parametrize('content,headers,expected', [
    (b'', {'Content-Type': 'application/zip'}, 400),
    (b'{}', {'Content-Type': 'application/json'}, 415),
    (b'bad', {'Content-Type': 'application/zip'}, 422),
    (b'a', {'Content-Type': 'application/zip', 'Content-Length': str(513 * 1024 * 1024)}, 413),
])
def test_invalid_upload_envelope(tmp_path, content, headers, expected):
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert client.post('/api/phone-calibration/imu-bundle', content=content, headers=headers).status_code == expected

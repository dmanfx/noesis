#!/usr/bin/env python3
"""Simple throughput benchmark for the MapAnything service."""
import argparse
import base64
import statistics
import time
from pathlib import Path

import numpy as np
import requests

from mapanything_config import load_service_config


def encode_dummy_frame(width: int = 640, height: int = 360) -> dict:
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    payload = base64.b64encode(frame.tobytes()).decode('ascii')
    return {
        'cam_id': 'benchmark-cam',
        'img_b64': payload,
        'shape': [height, width, 3]
    }


def main(iterations: int) -> None:
    config = load_service_config()
    url = f"{config.service.base_url}/infer_mono"
    headers = {'X-API-Key': config.service.api_key}
    view = encode_dummy_frame()
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        resp = requests.post(url, json={'view': view}, headers=headers, timeout=10)
        resp.raise_for_status()
        durations.append((time.perf_counter() - start) * 1000.0)
    avg = statistics.mean(durations)
    p90 = statistics.quantiles(durations, n=10)[-1]
    print(f"Ran {iterations} mono inferences. Avg latency: {avg:.2f} ms, P90: {p90:.2f} ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark MapAnything mono inference latency.')
    parser.add_argument('--iterations', type=int, default=10)
    args = parser.parse_args()
    main(args.iterations)

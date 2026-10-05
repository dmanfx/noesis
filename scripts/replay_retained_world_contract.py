#!/usr/bin/env python3
"""Replay admitted retained observations through the current canonical fusion.

Does not launch inference or another runtime. It proves contract changes on
real recorded observations, and reports any coordinate difference explicitly.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from noesis_core.contracts import ObservationEnvelope
from noesis_core.contracts.base import ArtifactFingerprint, ProducerRef
from noesis_core.world import GlobalWorldFusion
from noesis_core.world_service import WorldArtifacts, CanonicalWorldService


def replay(source: Path, output: Path) -> dict:
    fusion = None
    subjects = None
    current = None
    counts = Counter()
    max_delta = 0.0
    mismatches = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open() as incoming, output.open('x') as outgoing:
        for line_number, line in enumerate(incoming, 1):
            if line_number > 100000 or len(line) > 16 * 1024 * 1024:
                raise ValueError('retained cohort exceeds bounded replay')
            row = json.loads(line)
            if row.get('type') == 'tracking':
                old = row['world_snapshot']
                if fusion is None:
                    producer = ProducerRef.model_validate(old['producer'])
                    fusion = GlobalWorldFusion(producer)
                    fingerprint = ArtifactFingerprint(role='replay-placeholder-not-emitted', sha256='0' * 64)
                    subjects = CanonicalWorldService(producer=producer, artifacts=WorldArtifacts(fingerprint, fingerprint, fingerprint))
                tracks = {t['tracker_id']: t for t in row['tracks']}
                for raw in row['observations']:
                    observation = ObservationEnvelope.model_validate(raw)
                    track = tracks[observation.payload.tracklet.tracker_id]
                    fusion.ingest(observation, subjects._subject(observation, track),
                        trail_segment_id=track.get('trail_segment_id'),
                        trail_break_required=track.get('trail_break_required') is True,
                        tracker_lifecycle_generation=track.get('tracker_lifecycle_generation'),
                        source_epoch=track.get('source_epoch', row.get('source_epoch')))
                if not tracks:
                    fusion.clear_source(row['camera_id'], row['source_id'], old['producer']['run_id'])
                current = fusion.snapshot(published_at_us=old['published_at_us']).model_dump(mode='json')
                previous = {e['entity_id']: e for e in old['entities']}
                generated = {e['entity_id']: e for e in current['entities']}
                if set(previous) != set(generated) and len(mismatches) < 10:
                    mismatches.append({'line': line_number, 'kind': 'entity_set', 'before': sorted(previous), 'after': sorted(generated)})
                for identity in set(previous) & set(generated):
                    a, b = previous[identity]['position'], generated[identity]['position']
                    if a and b:
                        delta = math.sqrt(sum((a[k] - b[k]) ** 2 for k in ['x', 'y', 'z']))
                        max_delta = max(max_delta, delta)
                        if delta > 1e-8 and len(mismatches) < 10:
                            mismatches.append({'line': line_number, 'kind': 'position', 'entity': identity, 'delta_m': delta})
                    counts[generated[identity]['position_state'] or 'unknown'] += 1
                if current['sequence'] != old['sequence']:
                    raise ValueError('capture is not a complete contiguous producer sequence')
                row['world_snapshot'] = current
                counts['tracking'] += 1
            elif row.get('type') == 'world_snapshot':
                if current is None or row['payload']['sequence'] != current['sequence']:
                    raise ValueError('standalone snapshot is not its exact tracking cohort')
                row['payload'] = current
            outgoing.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
    return {'source': str(source), 'output': str(output), 'counts': dict(counts),
            'maximum_coordinate_change_m': max_delta, 'mismatches': mismatches,
            'scope': 'Current canonical fusion on retained admitted observation envelopes; no inference rerun, no position fitting.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.source, args.output)
    args.output.with_suffix('.review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if report['mismatches']:
        raise SystemExit('retained fusion differs: inspect evidence before using the review')


if __name__ == '__main__':
    main()

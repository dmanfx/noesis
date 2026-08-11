#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis.virtual_twin.releases import (  # noqa: E402
    SceneReleaseBuildError,
    build_scene_release,
    write_scene_release_bundle,
)
from noesis.virtual_twin.store import VirtualTwinStore  # noqa: E402
from noesis_core.scene_store import (  # noqa: E402
    SceneReleaseStore,
    SceneReleaseStoreError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build one immutable scene release from explicit virtual-twin revisions."
    )
    parser.add_argument("--revision", action="append", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--created-by", required=True)
    parser.add_argument("--authored-scene", type=Path, required=True)
    parser.add_argument("--cohort-max-seconds", type=float, default=120.0)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=REPO_ROOT / "data" / "virtual_twin" / "releases",
    )
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--scene-store", type=Path)
    parser.add_argument("--expected-current-release-id")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.promote and not args.register:
        raise SystemExit("--promote requires --register")
    if args.cohort_max_seconds < 0:
        raise SystemExit("--cohort-max-seconds cannot be negative")
    try:
        virtual_twins = VirtualTwinStore()
        release, validation = build_scene_release(
            virtual_twins=virtual_twins,
            revision_ids=args.revision,
            release_id=args.release_id,
            created_by=args.created_by,
            authored_scene_path=args.authored_scene,
            cohort_max_delta_us=int(args.cohort_max_seconds * 1_000_000),
        )
        release_path, validation_path = write_scene_release_bundle(
            args.output_directory,
            release,
            validation,
        )
        print(f"release={release_path}")
        print(f"validation={validation_path}")
        print(f"validation_sha256={release.validation_report_sha256}")
        if args.register:
            store_path = args.scene_store or (
                Path.home() / ".local" / "state" / "noesis" / "scene_releases.sqlite3"
            )
            scene_store = SceneReleaseStore(
                store_path,
                artifact_root=virtual_twins.revisions_root,
                bundle_root=virtual_twins.root,
            )
            release_sha = scene_store.register(release)
            print(f"registered_sha256={release_sha}")
            if args.promote:
                promotion = scene_store.promote(
                    release.release_id,
                    actor_id=args.created_by,
                    occurred_at_us=release.created_at_us,
                    expected_current_release_id=args.expected_current_release_id,
                )
                print(f"promotion_sequence={promotion.sequence}")
    except (SceneReleaseBuildError, SceneReleaseStoreError) as exc:
        print(f"scene release build failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

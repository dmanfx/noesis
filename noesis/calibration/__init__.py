"""DS8 Calibration module — single owner for K/E/align loading, validation, broadcasting."""

from noesis.calibration.manager import (
    CalibrationManager,
    CalibrationSnapshot,
    CalibrationValidationError,
    create_calibration_manager,
    load_camera_labels,
    streammux_size_from_pipeline_config,
)
from noesis.calibration.geometry import (
    PixelToWorldResult,
    pixel_to_world,
)
from noesis.calibration.pose_v1 import (
    E_col_major_to_pose_v1,
    normalize_pose_v1,
    pose_to_E_col_major,
)
from noesis.calibration.depth_registration import (
    DepthRegistrationBundle,
    DepthRegistrationEntry,
    DepthRegistrationError,
    DepthRegistrationManager,
    calibration_fingerprint,
    calibration_fingerprint_from_snapshot,
    fit_piecewise_registration,
    load_depth_registration,
    model_profile_fingerprint,
    profile_fingerprint,
    write_depth_registration_bundle,
)

__all__ = [
    "CalibrationManager",
    "CalibrationSnapshot",
    "CalibrationValidationError",
    "create_calibration_manager",
    "DepthRegistrationBundle",
    "DepthRegistrationEntry",
    "DepthRegistrationError",
    "DepthRegistrationManager",
    "PixelToWorldResult",
    "E_col_major_to_pose_v1",
    "calibration_fingerprint",
    "calibration_fingerprint_from_snapshot",
    "fit_piecewise_registration",
    "load_depth_registration",
    "load_camera_labels",
    "model_profile_fingerprint",
    "normalize_pose_v1",
    "pixel_to_world",
    "pose_to_E_col_major",
    "profile_fingerprint",
    "streammux_size_from_pipeline_config",
    "write_depth_registration_bundle",
]

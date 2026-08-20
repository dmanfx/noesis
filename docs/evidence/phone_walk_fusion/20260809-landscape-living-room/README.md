# Landscape Living-Room Phone Fusion Evidence

This is the curated image evidence for the selected 2026-08-09 landscape
phone-walk reconstruction documented in
`docs/Phone_Walk_Fusion_Reconstruction.md`.

Source identity:

```text
scan: 20260802-162254-8bcc7dd7
prepared views: 48
prepared orientation: landscape, 1920 x 1080
suite: da3_prior_suite_20260809
selected candidate: prior_conditioned_consensus_da3_carrier
final evaluation: evaluation_static_world_v5_final
coordinate frame: backend_world_m_stream_points
```

The full raw NPZ, GLB, PLY, camera solution, manifests, and evaluation data are
retained in large artifact storage. This directory deliberately keeps only the
compact review evidence suitable for the repository. Machine-readable source
identity, selected metrics, dimensions, sizes, and hashes are recorded in
`evidence_manifest.json`.

## Review set

- `variant_comparison.png` — all candidates under common static-world bounds.
- `selected_diagnostic_layers.png` — Heatmap-style **Diagnostic layers** for
  the selected candidate.
- `selected_point_layers_2p5cm.png` — non-averaged 2.5 cm phone-point layers by
  height band.
- `selected_point_cloud.png` — aligned selected reconstruction point-cloud
  review.
- `selected_mesh_2p5cm.png` — cropped 2.5 cm TSDF mesh review sheet.
- `consensus_collaboration_diagnostics.png` — cross-model agreement,
  reliability, provider selection, rejection, and trajectory diagnostics.
- `selected_fixed_camera_reprojection.jpg` — selected phone reconstruction
  reprojected into the independent fixed-camera view.

## Checksums

| File | SHA-256 |
| --- | --- |
| `variant_comparison.png` | `791a51237c1b312218a01ad12b58e3893482d6affb4254ccc4f69496233ca6b7` |
| `selected_diagnostic_layers.png` | `3418fdae4e8031e36b1319ea3f2404a2e42085d8e91576651fe76e81e4c81625` |
| `selected_point_layers_2p5cm.png` | `a8a6333ac522be208a7b011a65695b9e444b5c7b909d34bc2bb9e9189afcdc44` |
| `selected_point_cloud.png` | `2e4db47769b158bf63247e9a5bfe9cbdf468b9e8ce6d1d17ccf8926d64555ff0` |
| `selected_mesh_2p5cm.png` | `a236f34f8393518dac911ae630d68a653544fb70f28939a3987a059362518b1e` |
| `consensus_collaboration_diagnostics.png` | `03e589951a3060b7f1cc47219bb416380a6d261819e3da98280eecddbde2ea74` |
| `selected_fixed_camera_reprojection.jpg` | `e8a8270833b3a02a19ee7db73e162cc1666335b6104c2cea5ecb46c6667a8992` |

These sealed BEV images predate the canonical camera-ground raster contract and
retain the former Living Room–specific 180-degree review rotation. They are
historical evidence only and must not be used as an orientation template. New
diagnostics use row zero at maximum camera-forward +Z with no room-specific
rotation; neither presentation changes stored backend-world coordinates.

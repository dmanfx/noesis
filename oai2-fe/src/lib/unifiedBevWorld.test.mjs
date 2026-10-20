import test from 'node:test';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import {
  createUnifiedBevBindings, projectUnifiedBevMap, projectUnifiedBevMapPoint,
  createUnifiedBevState, reduceUnifiedBev, advanceUnifiedBev, clearUnifiedBevPeople,
} from './unifiedBevWorld.js';

// Compact original wire fields from the authorized gateway read, 2026-10-13.
// Raster bytes are omitted; these are contract fixtures, not physical accuracy.
const BUNDLE = {"cameras":{"frame_bindings":{"living-room":{"contract":"noesis.calibration.frame_binding","contract_version":1,"calibration_frame":{"frame_id":"backend_world_m","revision":"b73b6a742e2d0d5936876875d904aa74c98996088f06efd04d552dac3f96a214"},"world_frame":{"frame_id":"backend_world_m","revision":"sceneprior_living-room_20260802T202254Z_9380ed8099bb"},"target_from_calibration_col_major":[0.9989850506206234,0.040158882834844686,0.02039933248824016,0,-0.0405045900571707,0.9990376654237877,0.01682620720444276,0,-0.01970397982157794,-0.017635396055760005,0.9996503118516724,0,-0.3223193803386266,-0.6653283676088071,-0.22886856636030564,1],"transform_sha256":"006952a2f835e2cf9f85082eed831f2de466cc0734e5e56cb5d32efbfdfda643","calibration_floor_plane":{"normal":[0.04015888283484472,0.9990376654237886,-0.017635396055760023],"offset_m":-0.6653283676088076},"world_floor_plane":{"normal":[0,1,0],"offset_m":0},"camera_calibration_sha256":"9b34e3ea8a3808475584a55383afcff96373a9e0845d33dc026a6659c975be93","world_alignment_sha256":"88e05c3259bde42265c8bdc97e1c86762ca7540402041ebcceadb407d72df82b","camera_calibration_physical_sha256":null,"world_alignment_physical_sha256":null,"scene_prior_id":"sceneprior_living-room_20260802T202254Z_9380ed8099bb","artifact_revision_id":null,"target_revision_id":"vt_living_room_stream_rgbmesh_20260623T215821_538294633"},"kitchen":{"contract":"noesis.calibration.frame_binding","contract_version":1,"calibration_frame":{"frame_id":"backend_world_m","revision":"b73b6a742e2d0d5936876875d904aa74c98996088f06efd04d552dac3f96a214"},"world_frame":{"frame_id":"backend_world_m","revision":"sceneprior_kitchen_20260814T230412Z_cb6d1e0f8483"},"target_from_calibration_col_major":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],"transform_sha256":"af1290aa956661c7ece73215f4fe3ee0fc25a7fbef0c246fed2de66f12abd844","calibration_floor_plane":{"normal":[0,1,0],"offset_m":0},"world_floor_plane":{"normal":[0,1,0],"offset_m":0},"camera_calibration_sha256":"9b34e3ea8a3808475584a55383afcff96373a9e0845d33dc026a6659c975be93","world_alignment_sha256":"88e05c3259bde42265c8bdc97e1c86762ca7540402041ebcceadb407d72df82b","camera_calibration_physical_sha256":null,"world_alignment_physical_sha256":null,"scene_prior_id":"sceneprior_kitchen_20260814T230412Z_cb6d1e0f8483","artifact_revision_id":null,"target_revision_id":"vt_kitchen_stream_rgbmesh_20260623T215835_609188061"},"family-room":{"contract":"noesis.calibration.frame_binding","contract_version":1,"calibration_frame":{"frame_id":"backend_world_m","revision":"b73b6a742e2d0d5936876875d904aa74c98996088f06efd04d552dac3f96a214"},"world_frame":{"frame_id":"backend_world_m","revision":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4"},"target_from_calibration_col_major":[0.9467470066425219,-0.09059934064225462,-0.30896903548513316,0,0.0649604489274947,0.9936095819714693,-0.09230459733740415,0,0.31535732984752884,0.06731833397872138,0.9465822713434718,0,-3.1508635068958624,0.17446172385443262,5.462715819965061,1],"transform_sha256":"5bec221621ad20d64d5fc8477945b3d33ca90b75264184ea59194575167de217","calibration_floor_plane":{"normal":[-0.09059934064225462,0.9936095819714694,0.06731833397872138],"offset_m":0.17558377759227575},"world_floor_plane":{"normal":[0,1,0],"offset_m":0},"camera_calibration_sha256":"9b34e3ea8a3808475584a55383afcff96373a9e0845d33dc026a6659c975be93","world_alignment_sha256":"88e05c3259bde42265c8bdc97e1c86762ca7540402041ebcceadb407d72df82b","camera_calibration_physical_sha256":null,"world_alignment_physical_sha256":null,"scene_prior_id":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4","artifact_revision_id":null,"target_revision_id":"vt_family_room_stream_rgbmesh_20260623T215850_645134993"}}},"world_frame_presentations":{"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4":{"contract":"noesis.scene_prior.world_to_scene_presentation","contract_version":1,"world_frame":{"frame_id":"backend_world_m","revision":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4"},"render_frame":"menon_scene","world_to_scene_col_major":[88.1076476359557,0,57.27082741225053,0,0,105.08522847872348,0,0,-57.27082741225053,0,88.1076476359557,0,748.6596658465555,0,-823.7160342255914,1],"world_to_scene_sha256":"6b7da51dd05d107b61dfc86a385b6bd672f870f65ede71d76afb008c1466d57c","scene_revision_id":"fb8932ebf348c01e913482bf28e7bb61b0f27e62aa6bba6328b84054afb1bde3","source_transform_sha256s":["5bec221621ad20d64d5fc8477945b3d33ca90b75264184ea59194575167de217"],"provenance":{"role":"world_to_scene_presentation","sha256":"57b72ac1d33860c1e0c075851baed6465e33a2381336b299ab5ba65c7af626cf","version":"noesis.scene_presentation.owner_selected_structural_registration.v1","producer":"noesis.calibration.structural_presentation"},"accepted":true,"authored_scene_sha256":"84bfecdb07904141fcf713d40140a748ec806d432ef89d155b4b8a85ff07d5aa","registration_sha256":"57b72ac1d33860c1e0c075851baed6465e33a2381336b299ab5ba65c7af626cf"},"sceneprior_kitchen_20260814T230412Z_cb6d1e0f8483":{"contract":"noesis.scene_prior.world_to_scene_presentation","contract_version":1,"world_frame":{"frame_id":"backend_world_m","revision":"sceneprior_kitchen_20260814T230412Z_cb6d1e0f8483"},"render_frame":"menon_scene","world_to_scene_col_major":[94.63018872884415,0,29.566094606454183,0,0,99.14144728182285,0,0,-29.566094606454183,0,94.63018872884415,0,-32.13162859054481,0,-379.3407705184826,1],"world_to_scene_sha256":"49215ad07c6b16a53b8117e2b80b2425bf483999c88de0afe9cafa907143aab2","scene_revision_id":"62a434029f087e45ad1a975df19ccae4dbf887934202572044dd3a98d0c05af9","source_transform_sha256s":["af1290aa956661c7ece73215f4fe3ee0fc25a7fbef0c246fed2de66f12abd844"],"provenance":{"role":"world_to_scene_presentation","sha256":"57b72ac1d33860c1e0c075851baed6465e33a2381336b299ab5ba65c7af626cf","version":"noesis.scene_presentation.owner_selected_structural_registration.v1","producer":"noesis.calibration.structural_presentation"},"accepted":true,"authored_scene_sha256":"84bfecdb07904141fcf713d40140a748ec806d432ef89d155b4b8a85ff07d5aa","registration_sha256":"57b72ac1d33860c1e0c075851baed6465e33a2381336b299ab5ba65c7af626cf"},"sceneprior_living-room_20260802T202254Z_9380ed8099bb":{"contract":"noesis.scene_prior.world_to_scene_presentation","contract_version":1,"world_frame":{"frame_id":"backend_world_m","revision":"sceneprior_living-room_20260802T202254Z_9380ed8099bb"},"render_frame":"menon_scene","world_to_scene_col_major":[107.65497447441778,0,10.026641370447265,0,0,108.12089097976907,0,0,-10.026641370447265,0,107.65497447441778,0,-64.16366691341689,0,-91.58286698099874,1],"world_to_scene_sha256":"ee0e569b4c8d39be7ee9a2bba5970050819922537e56c3e7456c96818100f6e1","scene_revision_id":"aa2fbb94ad746f23111533798b429bd2ce353e770b0cca873e2aa9310e193b0d","source_transform_sha256s":["006952a2f835e2cf9f85082eed831f2de466cc0734e5e56cb5d32efbfdfda643"],"provenance":{"role":"world_to_scene_presentation","sha256":"57b72ac1d33860c1e0c075851baed6465e33a2381336b299ab5ba65c7af626cf","version":"noesis.scene_presentation.owner_selected_structural_registration.v1","producer":"noesis.calibration.structural_presentation"},"accepted":true,"authored_scene_sha256":"84bfecdb07904141fcf713d40140a748ec806d432ef89d155b4b8a85ff07d5aa","registration_sha256":"57b72ac1d33860c1e0c075851baed6465e33a2381336b299ab5ba65c7af626cf"}},"meta":{"version":3,"conventions":{"E":"world→camera","handedness":"RH","up":"Y"},"coord_space":"backend_world_m","units":"meters","world_frame":"backend_world_m","world_frame_contract":"revision_bound_per_camera_v1","cameras_E_semantics":"camera_from_calibration_frame_raw","scene_per_m":1,"world_frame_presentation_contract":"owner_selected_structural_scene_presentation_v1","structurally_qualified":false,"scene_presentation_selection_id":"owner-all-room-structural-20261004","authored_scene_sha256":"84bfecdb07904141fcf713d40140a748ec806d432ef89d155b4b8a85ff07d5aa","registration_sha256":"57b72ac1d33860c1e0c075851baed6465e33a2381336b299ab5ba65c7af626cf"}};
const MAPS = [{"type":"floorplan_response","request_id":"bev-wc1-read-living-room","camera_id":"living-room","cache_only":true,"scene_prior_only":true,"served_from_cache":true,"point_count":148768,"display_source":"pcf","ts":1786411643649927,"snapshot_id":"sceneprior_living-room_20260802T202254Z_9380ed8099bb","snapshot_content_sha256":"43aa8f768cff159b211de0cf4433b57ce23de54af75d861acaeb618d058ccc3e","world_frame":"backend_world_m","world_frame_revision":"sceneprior_living-room_20260802T202254Z_9380ed8099bb","frame":"camera_local_ground_m","units":"meters","bounds":{"min_x":-5.7250000000000005,"max_x":5.925000000000001,"min_z":-0.6000000000000001,"max_z":9.15},"scale_m_per_px":0.025,"scene_prior_meta":{"contract":"noesis.scene_prior.floorplan_composite","contract_version":1,"status":"pcf","prior_id":"sceneprior_living-room_20260802T202254Z_9380ed8099bb","revision_manifest_path":"revisions/sceneprior_living-room_20260802T202254Z_9380ed8099bb/manifest.json","revision_manifest_sha256":"43aa8f768cff159b211de0cf4433b57ce23de54af75d861acaeb618d058ccc3e","space_id":"living-room","mode":"shadow","source_type":"conditioned_multimodel_room_walk","source_model":"facebook/map-anything-apache + depth-anything/DA3Metric-Large + prior_conditioned_consensus_da3_carrier","source_frame":"backend_world_m","source_frame_revision":"sceneprior_living-room_20260802T202254Z_9380ed8099bb","frame_transform_sha256":"006952a2f835e2cf9f85082eed831f2de466cc0734e5e56cb5d32efbfdfda643","target_revision_id":"vt_living_room_stream_rgbmesh_20260623T215821_538294633","target_frame":"camera_local_ground_m","camera_geometry_source":"current_calibrated_extrinsics","display_source":"pcf","raster_orientation":"row_zero_max_z_rows_toward_min_z_columns_min_x_to_max_x","floor_y_m":0,"composition_policy":"pcf_only","live_observed_cells":0,"static_available_cells":32521,"static_fill_cells":32521,"composite_observed_cells":32521,"quality":{"passed":true,"source_point_count":150901,"selected_point_count":148768,"authored_cell_count":62252,"observed_cell_count":39065,"floor_supported_cell_count":22733,"obstacle_cell_count":9350,"authored_observed_fraction":0.6275300391955279,"authored_floor_supported_fraction":0.36517702242498234,"alignment_status":"passed"},"reason":"canonical_pcf_presentation_source"},"scene_prior_diagnostic_meta":{"contract":"noesis.scene_prior.diagnostic_layers","contract_version":1,"raster_encoding_version":1,"source":"facebook/map-anything-apache + depth-anything/DA3Metric-Large + prior_conditioned_consensus_da3_carrier","derivation":"prior_conditioned_fusion_points_and_grid","derived_at_catalog_load":false,"mapanything_inference_triggered":false,"registration_triggered":false,"bounds":{"min_x":-5.7250000000000005,"max_x":5.925000000000001,"min_z":-0.6000000000000001,"max_z":9.15},"grid_shape":[390,466],"resolution_m":0.025,"camera_position_world_m":[6.166371349363727,2.0868796323415535,0.11426735421942202],"camera_right_world_xz":[-0.9720468275897367,-0.23478706304379027],"camera_forward_world_xz":[-0.23478706304379024,0.9720468275897366],"raster_orientation":"row_zero_max_z_rows_toward_min_z_columns_min_x_to_max_x","point_count":148768},"calibration_fingerprint":"fada0ac04cc0e33b92e47a22a770c14e41d1243725d5f9c5dc57fc17dcd215b4"},{"type":"floorplan_response","request_id":"bev-wc1-read-kitchen","camera_id":"kitchen","cache_only":true,"scene_prior_only":true,"served_from_cache":true,"point_count":167338,"display_source":"pcf","ts":1786767243655236,"snapshot_id":"sceneprior_kitchen_20260814T230412Z_cb6d1e0f8483","snapshot_content_sha256":"f3c1e6c6b79f2a2437a7d7b34145d4833bbdf88699cd7e07c3059ad4daf1e198","world_frame":"backend_world_m","world_frame_revision":"sceneprior_kitchen_20260814T230412Z_cb6d1e0f8483","frame":"camera_local_ground_m","units":"meters","bounds":{"min_x":-7.15,"max_x":10.3,"min_z":-1.2000000000000002,"max_z":10},"scale_m_per_px":0.025,"scene_prior_meta":{"contract":"noesis.scene_prior.floorplan_composite","contract_version":1,"status":"pcf","prior_id":"sceneprior_kitchen_20260814T230412Z_cb6d1e0f8483","revision_manifest_path":"revisions/sceneprior_kitchen_20260814T230412Z_cb6d1e0f8483/manifest.json","revision_manifest_sha256":"f3c1e6c6b79f2a2437a7d7b34145d4833bbdf88699cd7e07c3059ad4daf1e198","space_id":"kitchen","mode":"shadow","source_type":"conditioned_multimodel_room_walk","source_model":"facebook/map-anything-apache + depth-anything/DA3Metric-Large + prior_conditioned_consensus_da3_carrier","source_frame":"backend_world_m","source_frame_revision":"sceneprior_kitchen_20260814T230412Z_cb6d1e0f8483","frame_transform_sha256":"af1290aa956661c7ece73215f4fe3ee0fc25a7fbef0c246fed2de66f12abd844","target_revision_id":"vt_kitchen_stream_rgbmesh_20260623T215835_609188061","target_frame":"camera_local_ground_m","camera_geometry_source":"current_calibrated_extrinsics","display_source":"pcf","raster_orientation":"row_zero_max_z_rows_toward_min_z_columns_min_x_to_max_x","floor_y_m":0,"composition_policy":"pcf_only","live_observed_cells":0,"static_available_cells":28524,"static_fill_cells":28524,"composite_observed_cells":28524,"quality":{"passed":true,"source_point_count":168795,"selected_point_count":167338,"authored_cell_count":72850,"observed_cell_count":36107,"floor_supported_cell_count":26353,"obstacle_cell_count":6038,"authored_observed_fraction":0.49563486616334934,"authored_floor_supported_fraction":0.3617433081674674,"alignment_status":"passed"},"reason":"canonical_pcf_presentation_source"},"scene_prior_diagnostic_meta":{"contract":"noesis.scene_prior.diagnostic_layers","contract_version":1,"raster_encoding_version":1,"source":"facebook/map-anything-apache + depth-anything/DA3Metric-Large + prior_conditioned_consensus_da3_carrier","derivation":"prior_conditioned_fusion_points_and_grid","derived_at_catalog_load":false,"mapanything_inference_triggered":false,"registration_triggered":false,"bounds":{"min_x":-7.15,"max_x":10.3,"min_z":-1.2000000000000002,"max_z":10},"grid_shape":[448,698],"resolution_m":0.025,"camera_position_world_m":[13.98107375623384,2.2651528326027273,0.1786232979176076],"camera_right_world_xz":[-0.9945218953682734,0.1045284632676535],"camera_forward_world_xz":[0.10452846326765348,0.9945218953682733],"raster_orientation":"row_zero_max_z_rows_toward_min_z_columns_min_x_to_max_x","point_count":167338},"calibration_fingerprint":"fa6f48ab2eea585a978dc94489acfbae9ccecc85162e200fd810b3639e53216d"},{"type":"floorplan_response","request_id":"bev-wc1-read-family-room","camera_id":"family-room","cache_only":true,"scene_prior_only":true,"served_from_cache":true,"point_count":118819,"display_source":"pcf","ts":1786419994545133,"snapshot_id":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4","snapshot_content_sha256":"4710d7ce658be69b33d809b6ab1d8614e5d06ffc015f26aba20e9990a18192ac","world_frame":"backend_world_m","world_frame_revision":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4","frame":"camera_local_ground_m","units":"meters","bounds":{"min_x":-7.550000000000001,"max_x":4.4,"min_z":-0.4,"max_z":8.725},"scale_m_per_px":0.025,"scene_prior_meta":{"contract":"noesis.scene_prior.floorplan_composite","contract_version":1,"status":"pcf","prior_id":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4","revision_manifest_path":"revisions/sceneprior_family-room_20260811T015847Z_8b80dc69a7c4/manifest.json","revision_manifest_sha256":"4710d7ce658be69b33d809b6ab1d8614e5d06ffc015f26aba20e9990a18192ac","space_id":"family-room","mode":"shadow","source_type":"conditioned_multimodel_room_walk","source_model":"facebook/map-anything-apache + depth-anything/DA3Metric-Large + prior_conditioned_consensus_da3_carrier","source_frame":"backend_world_m","source_frame_revision":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4","frame_transform_sha256":"5bec221621ad20d64d5fc8477945b3d33ca90b75264184ea59194575167de217","target_revision_id":"vt_family_room_stream_rgbmesh_20260623T215850_645134993","target_frame":"camera_local_ground_m","camera_geometry_source":"current_calibrated_extrinsics","display_source":"pcf","raster_orientation":"row_zero_max_z_rows_toward_min_z_columns_min_x_to_max_x","floor_y_m":0,"composition_policy":"pcf_only","live_observed_cells":0,"static_available_cells":21810,"static_fill_cells":21810,"composite_observed_cells":21810,"quality":{"passed":true,"source_point_count":119554,"selected_point_count":118819,"authored_cell_count":49762,"observed_cell_count":24960,"floor_supported_cell_count":11920,"obstacle_cell_count":8588,"authored_observed_fraction":0.5015875567702263,"authored_floor_supported_fraction":0.23954021140629395,"alignment_status":"passed"},"reason":"canonical_pcf_presentation_source"},"scene_prior_diagnostic_meta":{"contract":"noesis.scene_prior.diagnostic_layers","contract_version":1,"raster_encoding_version":1,"source":"facebook/map-anything-apache + depth-anything/DA3Metric-Large + prior_conditioned_consensus_da3_carrier","derivation":"prior_conditioned_fusion_points_and_grid","derived_at_catalog_load":false,"mapanything_inference_triggered":false,"registration_triggered":false,"bounds":{"min_x":-7.550000000000001,"max_x":4.4,"min_z":-0.4,"max_z":8.725},"grid_shape":[365,478],"resolution_m":0.025,"camera_position_world_m":[15.419562592882803,1.629763670010806,12.53002550330233],"camera_right_world_xz":[0.9982833503226408,0.05856921092693333],"camera_forward_world_xz":[0.058569210926933346,-0.9982833503226409],"raster_orientation":"row_zero_max_z_rows_toward_min_z_columns_min_x_to_max_x","point_count":118819},"calibration_fingerprint":"a521addf352a137af9080aa461409a367abd611337c2fd1ab2c780927e08dd90"}];
const HELD = {"entity_id":"visitor:ff74349a-cf7c-486b-9ddf-015d5034d30d:1016:g56","subject":{"subject_id":"visitor:ff74349a-cf7c-486b-9ddf-015d5034d30d:1016:g56","kind":"visitor","generation":56,"resident_uuid":null,"display_name":null,"stable_id":1016},"lifecycle":"present","position":{"x":17.72216123704632,"y":0,"z":9.709995417401142},"covariance":{"values":[0.64,0,0,0,0.64,0,0,0,0.64]},"position_state":"held","continuity_segment_id":2,"world_frame":"backend_world_m","world_frame_revision":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4","world_transform_sha256":"5bec221621ad20d64d5fc8477945b3d33ca90b75264184ea59194575167de217","calibration_revision":"unknown","position_quantity":"ground_footprint","support_state":"unknown","support_surface":null,"posture":"unknown","velocity_mps":null,"room_id":null,"observed_at_us":1791857148136654,"stale_after_us":1791857150136654,"sources":[{"observation_id":"ff74349a-cf7c-486b-9ddf-015d5034d30d:2:1212592:13366","camera_id":"family-room","zone":"Family Room","zone_source":"camera_default","zone_authoritative":false,"observed_at_us":1791857148136654,"position":{"x":17.72216123704632,"y":0,"z":9.709995417401142},"covariance":{"values":[0.64,0,0,0,0.64,0,0,0,0.64]},"world_frame":"backend_world_m","world_frame_revision":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4","world_transform_sha256":"5bec221621ad20d64d5fc8477945b3d33ca90b75264184ea59194575167de217","calibration_revision":"unknown","support_surface":null,"position_source":"anchor_hold","position_quality":"held","position_state":"held","accepted":true,"rejection_reason":null}],"conflict":false,"conflict_reason":null};
// Original sparse canonical journal payloads from 02:10–02:12 UTC. The unit
// envelope() helper below is SYNTHETIC: these are not retained wire cohorts.
const JOURNAL_SNAPSHOTS = [{"contract":"noesis.world.snapshot","contract_version":1,"entities":[{"calibration_revision":"unknown","conflict":false,"conflict_reason":null,"continuity_segment_id":3,"covariance":{"values":[0.005430768405881018,0,0.0036593339310500425,0,0.03998978126430081,0,0.0036593339310500425,0,0.018764304446586333]},"entity_id":"visitor:ff74349a-cf7c-486b-9ddf-015d5034d30d:1001:g56","lifecycle":"present","observed_at_us":1791857399994987,"position":{"x":7.290092882423013,"y":0.1905,"z":4.609715644558751},"position_quantity":"ground_footprint","position_state":"measured","posture":"standing","room_id":null,"sources":[{"accepted":true,"calibration_revision":"unknown","camera_id":"living-room","covariance":{"values":[0.005430768405881,1.6906330046585975e-19,0.0036593339310500425,1.6906330046585975e-19,0.03998978126430079,-1.810107513617141e-17,0.0036593339310500425,-1.810107513617141e-17,0.018764304446586316]},"observation_id":"ff74349a-cf7c-486b-9ddf-015d5034d30d:0:33862:13380","observed_at_us":1791857399994987,"position":{"x":7.290092882423013,"y":0.1905,"z":4.609715644558751},"position_quality":"estimated","position_source":"person_anchor_floor_only","position_state":"measured","rejection_reason":null,"support_surface":{"extent_xz":[[6.355887911888553,8.720242938196536],[5.714492312291988,4.4409155409119565],[7.5651657717555185,4.332387358586472],[8.465699999124784,8.495734174333105]],"normal_world":[0,1,0],"offset_m":-0.1905,"revision":"692b982cf2973d3dca6bca9f69ff99f8461c81dd3dc8103237d7c0611bb50d60","surface_id":"foyer-upper-floor"},"world_frame":"backend_world_m","world_frame_revision":"sceneprior_living-room_20260802T202254Z_9380ed8099bb","world_transform_sha256":"006952a2f835e2cf9f85082eed831f2de466cc0734e5e56cb5d32efbfdfda643","zone":"Living Room","zone_authoritative":false,"zone_source":"camera_default"}],"stale_after_us":1791857401994987,"subject":{"display_name":null,"generation":56,"kind":"visitor","resident_uuid":null,"stable_id":1001,"subject_id":"visitor:ff74349a-cf7c-486b-9ddf-015d5034d30d:1001:g56"},"support_state":"floor","support_surface":{"extent_xz":[[6.355887911888553,8.720242938196536],[5.714492312291988,4.4409155409119565],[7.5651657717555185,4.332387358586472],[8.465699999124784,8.495734174333105]],"normal_world":[0,1,0],"offset_m":-0.1905,"revision":"692b982cf2973d3dca6bca9f69ff99f8461c81dd3dc8103237d7c0611bb50d60","surface_id":"foyer-upper-floor"},"velocity_mps":{"x":-0.2007576390645314,"y":0,"z":-0.0013327531617096983},"world_frame":"backend_world_m","world_frame_revision":"sceneprior_living-room_20260802T202254Z_9380ed8099bb","world_transform_sha256":"006952a2f835e2cf9f85082eed831f2de466cc0734e5e56cb5d32efbfdfda643"},{"calibration_revision":"unknown","conflict":false,"conflict_reason":null,"continuity_segment_id":1,"covariance":{"values":[0.64,0,0,0,0.64,0,0,0,0.64]},"entity_id":"visitor:ff74349a-cf7c-486b-9ddf-015d5034d30d:1020:g52","lifecycle":"present","observed_at_us":1791857399909697,"position":{"x":18.07010788364527,"y":0,"z":9.197075593698562},"position_quantity":"ground_footprint","position_state":"held","posture":"unknown","room_id":null,"sources":[{"accepted":true,"calibration_revision":"unknown","camera_id":"family-room","covariance":{"values":[0.64,0,0,0,0.64,0,0,0,0.64]},"observation_id":"ff74349a-cf7c-486b-9ddf-015d5034d30d:2:1215323:13307","observed_at_us":1791857399909697,"position":{"x":18.07010788364527,"y":0,"z":9.197075593698562},"position_quality":"held","position_source":"anchor_hold","position_state":"held","rejection_reason":null,"support_surface":null,"world_frame":"backend_world_m","world_frame_revision":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4","world_transform_sha256":"5bec221621ad20d64d5fc8477945b3d33ca90b75264184ea59194575167de217","zone":"Family Room","zone_authoritative":false,"zone_source":"camera_default"}],"stale_after_us":1791857401909697,"subject":{"display_name":null,"generation":52,"kind":"visitor","resident_uuid":null,"stable_id":1020,"subject_id":"visitor:ff74349a-cf7c-486b-9ddf-015d5034d30d:1020:g52"},"support_state":"unknown","support_surface":null,"velocity_mps":null,"world_frame":"backend_world_m","world_frame_revision":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4","world_transform_sha256":"5bec221621ad20d64d5fc8477945b3d33ca90b75264184ea59194575167de217"}],"frame":"backend_world_m","observed_end_us":1791857399994987,"observed_start_us":1791857399909697,"producer":{"instance_id":"TauntonMainframe","run_id":"ff74349a-cf7c-486b-9ddf-015d5034d30d","runtime":"ds9","software_revision":"5d77c658e64152e95be12cd5a20597a8f25952a2"},"published_at_us":1791857400016101,"sequence":1517096,"snapshot_id":"ff74349a-cf7c-486b-9ddf-015d5034d30d:1517096","units":"meters"},{"contract":"noesis.world.snapshot","contract_version":1,"entities":[{"calibration_revision":"unknown","conflict":false,"conflict_reason":null,"continuity_segment_id":6,"covariance":{"values":[0.64,0,0,0,0.64,0,0,0,0.64]},"entity_id":"visitor:ff74349a-cf7c-486b-9ddf-015d5034d30d:1011:g50","lifecycle":"present","observed_at_us":1791857444992779,"position":{"x":17.664880354907563,"y":0,"z":6.2832407821527285},"position_quantity":"ground_footprint","position_state":"predicted","posture":"standing","room_id":null,"sources":[{"accepted":true,"calibration_revision":"unknown","camera_id":"kitchen","covariance":{"values":[0.64,0,0,0,0.64,0,0,0,0.64]},"observation_id":"ff74349a-cf7c-486b-9ddf-015d5034d30d:1:554662:13390","observed_at_us":1791857444992779,"position":{"x":17.664880354907563,"y":0,"z":6.2832407821527285},"position_quality":"held","position_source":"image_motion_prediction","position_state":"predicted","rejection_reason":null,"support_surface":null,"world_frame":"backend_world_m","world_frame_revision":"sceneprior_kitchen_20260814T230412Z_cb6d1e0f8483","world_transform_sha256":"af1290aa956661c7ece73215f4fe3ee0fc25a7fbef0c246fed2de66f12abd844","zone":"Kitchen","zone_authoritative":false,"zone_source":"camera_default"}],"stale_after_us":1791857446992779,"subject":{"display_name":null,"generation":50,"kind":"visitor","resident_uuid":null,"stable_id":1011,"subject_id":"visitor:ff74349a-cf7c-486b-9ddf-015d5034d30d:1011:g50"},"support_state":"unknown","support_surface":null,"velocity_mps":null,"world_frame":"backend_world_m","world_frame_revision":"sceneprior_kitchen_20260814T230412Z_cb6d1e0f8483","world_transform_sha256":"af1290aa956661c7ece73215f4fe3ee0fc25a7fbef0c246fed2de66f12abd844"},{"calibration_revision":"unknown","conflict":false,"conflict_reason":null,"continuity_segment_id":1,"covariance":{"values":[0.64,0,0,0,0.64,0,0,0,0.64]},"entity_id":"visitor:ff74349a-cf7c-486b-9ddf-015d5034d30d:1020:g52","lifecycle":"present","observed_at_us":1791857444899183,"position":{"x":18.07010788364527,"y":0,"z":9.197075593698562},"position_quantity":"ground_footprint","position_state":"held","posture":"unknown","room_id":null,"sources":[{"accepted":true,"calibration_revision":"unknown","camera_id":"family-room","covariance":{"values":[0.64,0,0,0,0.64,0,0,0,0.64]},"observation_id":"ff74349a-cf7c-486b-9ddf-015d5034d30d:2:1216131:13307","observed_at_us":1791857444899183,"position":{"x":18.07010788364527,"y":0,"z":9.197075593698562},"position_quality":"held","position_source":"anchor_hold","position_state":"held","rejection_reason":null,"support_surface":null,"world_frame":"backend_world_m","world_frame_revision":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4","world_transform_sha256":"5bec221621ad20d64d5fc8477945b3d33ca90b75264184ea59194575167de217","zone":"Family Room","zone_authoritative":false,"zone_source":"camera_default"}],"stale_after_us":1791857446899183,"subject":{"display_name":null,"generation":52,"kind":"visitor","resident_uuid":null,"stable_id":1020,"subject_id":"visitor:ff74349a-cf7c-486b-9ddf-015d5034d30d:1020:g52"},"support_state":"unknown","support_surface":null,"velocity_mps":null,"world_frame":"backend_world_m","world_frame_revision":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4","world_transform_sha256":"5bec221621ad20d64d5fc8477945b3d33ca90b75264184ea59194575167de217"},{"calibration_revision":"unknown","conflict":false,"conflict_reason":null,"continuity_segment_id":23,"covariance":{"values":[0.0025,8.092765130783517e-19,0.003272123160471687,8.092765130783517e-19,0.03671497208230367,-1.4082052561745353e-17,0.003272123160471687,-1.4082052561745353e-17,0.008757788348023392]},"entity_id":"visitor:ff74349a-cf7c-486b-9ddf-015d5034d30d:1026:g49","lifecycle":"present","observed_at_us":1791857444899183,"position":{"x":13.62200819919124,"y":0,"z":8.571677204046875},"position_quantity":"ground_footprint","position_state":"measured","posture":"standing","room_id":null,"sources":[{"accepted":true,"calibration_revision":"unknown","camera_id":"family-room","covariance":{"values":[0.0019270876713314013,8.092765130783517e-19,0.003272123160471687,8.092765130783517e-19,0.03614205975363507,-1.4082052561745353e-17,0.003272123160471687,-1.4082052561745353e-17,0.008184876019354793]},"observation_id":"ff74349a-cf7c-486b-9ddf-015d5034d30d:2:1216130:13382","observed_at_us":1791857444899183,"position":{"x":13.62200819919124,"y":0,"z":8.571677204046875},"position_quality":"estimated","position_source":"pose_floor_only","position_state":"measured","rejection_reason":null,"support_surface":null,"world_frame":"backend_world_m","world_frame_revision":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4","world_transform_sha256":"5bec221621ad20d64d5fc8477945b3d33ca90b75264184ea59194575167de217","zone":"Family Room","zone_authoritative":false,"zone_source":"camera_default"}],"stale_after_us":1791857446899183,"subject":{"display_name":null,"generation":49,"kind":"visitor","resident_uuid":null,"stable_id":1026,"subject_id":"visitor:ff74349a-cf7c-486b-9ddf-015d5034d30d:1026:g49"},"support_state":"floor","support_surface":null,"velocity_mps":null,"world_frame":"backend_world_m","world_frame_revision":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4","world_transform_sha256":"5bec221621ad20d64d5fc8477945b3d33ca90b75264184ea59194575167de217"}],"frame":"backend_world_m","observed_end_us":1791857444992779,"observed_start_us":1791857444899183,"producer":{"instance_id":"TauntonMainframe","run_id":"ff74349a-cf7c-486b-9ddf-015d5034d30d","runtime":"ds9","software_revision":"5d77c658e64152e95be12cd5a20597a8f25952a2"},"published_at_us":1791857445000486,"sequence":1517911,"snapshot_id":"ff74349a-cf7c-486b-9ddf-015d5034d30d:1517911","units":"meters"},{"contract":"noesis.world.snapshot","contract_version":1,"entities":[{"calibration_revision":"unknown","conflict":false,"conflict_reason":null,"continuity_segment_id":1,"covariance":{"values":[0.1329081546384232,-6.269421963682345e-18,-0.24759925975749375,-6.269421963682345e-18,0.04000000000000002,2.011661950098173e-16,-0.24759925975749375,2.011661950098173e-16,0.7601722750735866]},"entity_id":"visitor:ff74349a-cf7c-486b-9ddf-015d5034d30d:1001:g56","lifecycle":"present","observed_at_us":1791857498668343,"position":{"x":16.414825006867318,"y":0,"z":9.716063517394169},"position_quantity":"ground_footprint","position_state":"measured","posture":"standing","room_id":null,"sources":[{"accepted":true,"calibration_revision":"unknown","camera_id":"family-room","covariance":{"values":[0.1329081546384232,-6.269421963682345e-18,-0.24759925975749375,-6.269421963682345e-18,0.04000000000000002,2.011661950098173e-16,-0.24759925975749375,2.011661950098173e-16,0.7601722750735866]},"observation_id":"ff74349a-cf7c-486b-9ddf-015d5034d30d:2:1217064:13404","observed_at_us":1791857498668343,"position":{"x":16.414825006867318,"y":0,"z":9.716063517394169},"position_quality":"estimated","position_source":"gravity_drop","position_state":"measured","rejection_reason":null,"support_surface":null,"world_frame":"backend_world_m","world_frame_revision":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4","world_transform_sha256":"5bec221621ad20d64d5fc8477945b3d33ca90b75264184ea59194575167de217","zone":"Family Room","zone_authoritative":false,"zone_source":"camera_default"}],"stale_after_us":1791857500668343,"subject":{"display_name":null,"generation":56,"kind":"visitor","resident_uuid":null,"stable_id":1001,"subject_id":"visitor:ff74349a-cf7c-486b-9ddf-015d5034d30d:1001:g56"},"support_state":"unknown","support_surface":null,"velocity_mps":null,"world_frame":"backend_world_m","world_frame_revision":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4","world_transform_sha256":"5bec221621ad20d64d5fc8477945b3d33ca90b75264184ea59194575167de217"}],"frame":"backend_world_m","observed_end_us":1791857498668343,"observed_start_us":1791857498668343,"producer":{"instance_id":"TauntonMainframe","run_id":"ff74349a-cf7c-486b-9ddf-015d5034d30d","runtime":"ds9","software_revision":"5d77c658e64152e95be12cd5a20597a8f25952a2"},"published_at_us":1791857498679072,"sequence":1518698,"snapshot_id":"ff74349a-cf7c-486b-9ddf-015d5034d30d:1518698","units":"meters"},{"contract":"noesis.world.snapshot","contract_version":1,"entities":[{"calibration_revision":"unknown","conflict":false,"conflict_reason":null,"continuity_segment_id":6,"covariance":{"values":[0.01221730770477989,0,0.004901177276601238,0,0.03246208846049882,0,0.004901177276601238,0,0.002987453728255083]},"entity_id":"visitor:ff74349a-cf7c-486b-9ddf-015d5034d30d:1001:g56","lifecycle":"present","observed_at_us":1791857504999488,"position":{"x":14.0931134280357,"y":0,"z":9.552901508127198},"position_quantity":"ground_footprint","position_state":"measured","posture":"standing","room_id":null,"sources":[{"accepted":true,"calibration_revision":"unknown","camera_id":"family-room","covariance":{"values":[0.01221730770477989,0,0.004901177276601238,0,0.03246208846049882,0,0.004901177276601238,0,0.002987453728255083]},"observation_id":"ff74349a-cf7c-486b-9ddf-015d5034d30d:2:1217133:13404","observed_at_us":1791857504999488,"position":{"x":14.0931134280357,"y":0,"z":9.552901508127198},"position_quality":"estimated","position_source":"pose_floor_only","position_state":"measured","rejection_reason":null,"support_surface":null,"world_frame":"backend_world_m","world_frame_revision":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4","world_transform_sha256":"5bec221621ad20d64d5fc8477945b3d33ca90b75264184ea59194575167de217","zone":"Family Room","zone_authoritative":false,"zone_source":"camera_default"}],"stale_after_us":1791857506999488,"subject":{"display_name":null,"generation":56,"kind":"visitor","resident_uuid":null,"stable_id":1001,"subject_id":"visitor:ff74349a-cf7c-486b-9ddf-015d5034d30d:1001:g56"},"support_state":"floor","support_surface":null,"velocity_mps":{"x":-1.4690906580082281,"y":0,"z":-0.15433318198131446},"world_frame":"backend_world_m","world_frame_revision":"sceneprior_family-room_20260811T015847Z_8b80dc69a7c4","world_transform_sha256":"5bec221621ad20d64d5fc8477945b3d33ca90b75264184ea59194575167de217"}],"frame":"backend_world_m","observed_end_us":1791857504999488,"observed_start_us":1791857504999488,"producer":{"instance_id":"TauntonMainframe","run_id":"ff74349a-cf7c-486b-9ddf-015d5034d30d","runtime":"ds9","software_revision":"5d77c658e64152e95be12cd5a20597a8f25952a2"},"published_at_us":1791857505021755,"sequence":1518796,"snapshot_id":"ff74349a-cf7c-486b-9ddf-015d5034d30d:1518796","units":"meters"}];
const clone = (v) => JSON.parse(JSON.stringify(v));
const DIGEST = BUNDLE.meta.authored_scene_sha256;
const apply = (state, message, ms = 1000) => reduceUnifiedBev(state, message, { receivedAtMs: ms });
const initialized = () => apply(createUnifiedBevState({ authoredSceneSha256: DIGEST }),
  { type: 'calibration-bundle', data: BUNDLE }, 0);
function entity(time = 1_000_000, changes = {}) {
  const e = clone(HELD);
  Object.assign(e, { entity_id: 'visitor:test:7:g1',
    subject: { subject_id: 'visitor:test:7:g1', kind: 'visitor', generation: 1, stable_id: 7, resident_uuid: null, display_name: null },
    position_state: 'measured', observed_at_us: time, stale_after_us: time + 2_000_000,
    continuity_segment_id: 1 }, changes);
  e.sources[0].observed_at_us = time;
  e.sources[0].position_state = e.position_state;
  return e;
}
function snapshot(sequence = 1, entities = [entity()], changes = {}) {
  const time = 1_000_000 + sequence * 10_000;
  return { contract: 'noesis.world.snapshot', contract_version: 1,
    snapshot_id: `run:${sequence}`, producer: { runtime: 'test', instance_id: 'fixture', run_id: 'run', software_revision: 'r1' },
    sequence, observed_start_us: Math.min(time, ...entities.map(e => e.observed_at_us)),
    observed_end_us: Math.max(time, ...entities.map(e => e.observed_at_us)),
    published_at_us: time, frame: 'backend_world_m', units: 'meters', entities, ...changes };
}
function envelope(s, sequence = s.sequence, changes = {}) {
  const cohort = { source_id: 2, frame_id: sequence, observed_at_us: s.published_at_us, tracking_publication_sequence: sequence };
  return { type: 'world_snapshot', payload: s, cohort, ...cohort, ...changes };
}
function tracking(s, events = [], changes = {}) {
  const { payload, ...wire } = envelope(s);
  return { ...wire, type: 'tracking', camera_id: 'family-room', source_epoch: 0, world_snapshot: payload, world_events: events, ...changes };
}
function feed(state, s, ms = 1000, t = {}) {
  const staged = apply(state, tracking(s, [], t), ms);
  assert.equal(staged.diagnostic.accepted, true, staged.diagnostic.reason);
  const wire = envelope(s);
  if (t.cohort) Object.assign(wire, t.cohort, { cohort: t.cohort });
  const next = apply(staged, wire, ms);
  assert.equal(next.diagnostic.accepted, true, next.diagnostic.reason);
  return next;
}
const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-8, `${a} != ${b}`);
function expectedPoint(matrix, p) {
  return [
    matrix[0] * p[0] + matrix[4] * p[1] + matrix[8] * p[2] + matrix[12],
    matrix[1] * p[0] + matrix[5] * p[1] + matrix[9] * p[2] + matrix[13],
    matrix[2] * p[0] + matrix[6] * p[1] + matrix[10] * p[2] + matrix[14],
  ];
}

test('actual three camera binding shape admits one authored space, with distinct scene revisions', () => {
  const bindings = createUnifiedBevBindings(BUNDLE, DIGEST);
  assert.equal(Object.keys(bindings.cameras).length, 3);
  assert.equal(new Set(Object.values(bindings.presentations).map(p => p.authored_scene_sha256)).size, 1);
  assert.equal(new Set(Object.values(bindings.presentations).map(p => p.scene_revision_id)).size, 3);
  assert.ok(Object.isFrozen(bindings));
  const extra = clone(BUNDLE);
  extra.cameras.frame_bindings['family-room'].unrelated_large_payload = ['not copied'];
  extra.world_frame_presentations[HELD.world_frame_revision].unrelated_large_payload = ['not copied'];
  assert.deepEqual(createUnifiedBevBindings(extra, DIGEST), bindings);
});

test('all actual PCF footprints use the published local basis and target mapping exactly once', () => {
  const bindings = createUnifiedBevBindings(BUNDLE, DIGEST);
  for (const f of MAPS) {
    const map = projectUnifiedBevMap(bindings, f);
    const g = f.scene_prior_diagnostic_meta;
    const [x, z] = [f.bounds.min_x, f.bounds.max_z];
    const world = [g.camera_position_world_m[0] + x * g.camera_right_world_xz[0] + z * g.camera_forward_world_xz[0],
      f.scene_prior_meta.floor_y_m,
      g.camera_position_world_m[2] + x * g.camera_right_world_xz[1] + z * g.camera_forward_world_xz[1]];
    const expected = expectedPoint(BUNDLE.world_frame_presentations[f.world_frame_revision].world_to_scene_col_major, world);
    const actual = projectUnifiedBevMapPoint(map, [x, z]);
    actual.forEach((v, i) => near(v, expected[i]));
    assert.equal(map.positionAuthority, 'static_only');
    assert.equal(map.staticObservedAtUs, f.ts);
    assert.equal(map.gridShape[0], g.grid_shape[0]);
  }
});

test('captured held person has transform parity; static maps do not change its world position', () => {
  const e = entity(1_000_000, { position_state: 'held' });
  let state = feed(initialized(), snapshot(1, [e]));
  const person = state.people[0];
  const expected = expectedPoint(BUNDLE.world_frame_presentations[e.world_frame_revision].world_to_scene_col_major,
    [e.position.x, e.position.y, e.position.z]);
  person.position.forEach((v, i) => near(v, expected[i]));
  assert.equal(person.sceneTransformCount, 1);
  assert.equal(person.admission, 'held');
  assert.equal(person.trail.length, 0);
  assert.deepEqual(person.sourceEvidence, e.sources);
  for (const f of MAPS) state = apply(state, f);
  assert.deepEqual(state.people[0].position, person.position);
  assert.deepEqual(state.people[0].worldPosition, person.worldPosition);
});

test('occupied journal payloads retain original clocks, StableIDs and measured/predicted/held projection parity', () => {
  const seen = new Set();
  for (const s of JOURNAL_SNAPSHOTS) {
    // Isolate sparse records; never reconstruct a continuous path between them.
    const state = feed(initialized(), s);
    assert.equal(state.cursor.snapshotId, s.snapshot_id);
    assert.equal(state.cursor.publishedAtUs, s.published_at_us);
    assert.equal(state.unavailable.length, 0);
    assert.equal(state.people.length, s.entities.length);
    for (const e of s.entities) {
      const p = state.people.find(person => person.entityId === e.entity_id);
      const world = [e.position.x, e.position.y, e.position.z];
      const expected = expectedPoint(BUNDLE.world_frame_presentations[e.world_frame_revision].world_to_scene_col_major, world);
      p.position.forEach((v, i) => near(v, expected[i]));
      assert.deepEqual(p.worldPosition, world);
      assert.equal(p.stableId, e.subject.stable_id);
      assert.equal(p.observedAtUs, e.observed_at_us);
      assert.equal(p.staleAfterUs, e.stale_after_us);
      assert.deepEqual(p.sourceEvidence, e.sources);
      assert.equal(p.admission, e.position_state === 'measured' ? 'accepted' : e.position_state);
      assert.equal(p.trail.length, e.position_state === 'measured' ? 1 : 0);
      assert.equal(p.sceneTransformCount, 1);
      seen.add(e.position_state);
    }
  }
  assert.deepEqual([...seen].sort(), ['held', 'measured', 'predicted']);
});

test('velocity and covariance use only the linear mapping', () => {
  const e = entity(1_000_000, { velocity_mps: { x: 1, y: 0, z: 0 } });
  const p = feed(initialized(), snapshot(1, [e])).people[0];
  const m = BUNDLE.world_frame_presentations[e.world_frame_revision].world_to_scene_col_major;
  p.sceneVelocity.forEach((v, i) => near(v, m[i]));
  const scale = Math.hypot(m[0], m[1], m[2]);
  near(p.sceneCovariance[0], 0.64 * scale * scale);
  near(p.sceneCovariance[4], 0.64 * scale * scale);
  near(p.sceneCovariance[8], 0.64 * scale * scale);
});

test('invalid affine, singular or nonuniform mappings, revisions, hash allowlists and target bindings fail closed', () => {
  const mutations = [
    b => { Object.values(b.world_frame_presentations)[0].world_to_scene_col_major[3] = 0.1; },
    b => { Object.values(b.world_frame_presentations)[0].world_to_scene_col_major[0] = Infinity; },
    b => { b.cameras.frame_bindings['family-room'].target_from_calibration_col_major.fill(0); },
    b => { Object.values(b.world_frame_presentations)[0].world_to_scene_col_major[5] *= 2; },
    b => { Object.values(b.world_frame_presentations)[0].world_frame.revision = 'other'; },
    b => { b.cameras.frame_bindings['family-room'].transform_sha256 = 'a'.repeat(64); },
    b => { Object.values(b.world_frame_presentations)[0].world_to_scene_sha256 = 'bad'; },
    b => { Object.values(b.world_frame_presentations)[0].authored_scene_sha256 = 'b'.repeat(64); },
  ];
  for (const mutate of mutations) {
    const b = clone(BUNDLE); mutate(b);
    assert.throws(() => createUnifiedBevBindings(b, DIGEST));
  }
  assert.throws(() => createUnifiedBevBindings(BUNDLE, 'b'.repeat(64)), /authored_scene_binding_mismatch/);
});

test('version 2 metric acceptance and conflicting same-edge bindings are checked', () => {
  const b = clone(BUNDLE); const camera = b.cameras.frame_bindings['family-room'];
  camera.contract_version = 2; camera.metric_frame = { frame: clone(camera.world_frame), accepted: true };
  assert.doesNotThrow(() => createUnifiedBevBindings(b, DIGEST));
  camera.metric_frame.accepted = false;
  assert.throws(() => createUnifiedBevBindings(b, DIGEST), /metric_frame_unaccepted/);
  const duplicate = clone(BUNDLE);
  duplicate.cameras.frame_bindings.alias = clone(duplicate.cameras.frame_bindings['family-room']);
  duplicate.cameras.frame_bindings.alias.target_from_calibration_col_major[12] += 1;
  assert.throws(() => createUnifiedBevBindings(duplicate, DIGEST), /world_frame_binding_conflict/);
});

test('static geometry refuses wrong revision, digest, basis, snapshot, grid and manifest-preview source', () => {
  for (const invalid of [undefined, NaN, Infinity, -1]) {
    assert.throws(() => projectUnifiedBevMap(createUnifiedBevBindings(BUNDLE, DIGEST),
      { ...MAPS[0], ts: invalid }), /static_map_frame_invalid/);
  }
  assert.throws(() => projectUnifiedBevMap(createUnifiedBevBindings(BUNDLE, DIGEST),
    { ...MAPS[0], snapshot_id: null }), /static_map_frame_invalid/);
  const bindings = createUnifiedBevBindings(BUNDLE, DIGEST);
  const mutations = [
    f => { f.world_frame_revision = 'wrong'; },
    f => { f.scene_prior_meta.frame_transform_sha256 = 'a'.repeat(64); },
    f => { f.scene_prior_diagnostic_meta.camera_right_world_xz = [1, 1]; },
    f => { f.snapshot_content_sha256 = 'a'.repeat(64); },
    f => { f.scene_prior_diagnostic_meta.grid_shape[0] += 1; },
    f => { f.scene_prior_meta.camera_geometry_source = 'manifest_preview'; },
  ];
  for (const mutate of mutations) {
    const f = clone(MAPS[0]); mutate(f);
    assert.throws(() => projectUnifiedBevMap(bindings, f));
  }
});

test('nested tracking stages exact data; standalone snapshot is authoritative, with complete cohort parity', () => {
  const s = snapshot();
  const staged = apply(initialized(), tracking(s));
  assert.equal(staged.people.length, 0);
  assert.equal(staged.pending.length, 1);
  const mismatch = envelope(clone(s)); mismatch.payload.entities[0].position.x += 1;
  assert.equal(apply(staged, mismatch).diagnostic.reason, 'snapshot_mismatch');
  const cohortMismatch = envelope(s); cohortMismatch.frame_id += 1;
  assert.equal(apply(staged, cohortMismatch).diagnostic.reason, 'cohort_mismatch');
  assert.equal(apply(staged, envelope(s)).people.length, 1);
});

test('strict schema refuses unknown fields, frame changes and duplicate entity/subject/StableIDs', () => {
  const duplicateEntity = entity(); const duplicateSubject = entity(); const duplicateStable = entity();
  duplicateSubject.entity_id = 'second';
  duplicateStable.entity_id = 'second'; duplicateStable.subject.subject_id = 'second';
  for (const entities of [[entity(), duplicateEntity], [entity(), duplicateSubject], [entity(), duplicateStable]]) {
    assert.equal(apply(initialized(), envelope(snapshot(1, entities))).diagnostic.reason, 'canonical_identity_duplicate');
  }
  const extra = snapshot(); extra.entities[0].sceneTransformCount = 1;
  assert.equal(apply(initialized(), envelope(extra)).diagnostic.reason, 'contract_field_unexpected');
  const wrongFrame = snapshot(); wrongFrame.frame = 'menon_scene';
  assert.equal(apply(initialized(), envelope(wrongFrame)).diagnostic.accepted, false);
});

test('missing point, wrong source revision/hash/camera and missing canonical provenance stay unavailable', () => {
  const mutations = [
    e => { e.position = null; e.covariance = null; },
    e => { e.world_frame_revision = 'unadmitted'; },
    e => { e.sources[0].world_transform_sha256 = 'a'.repeat(64); },
    e => { e.sources[0].camera_id = 'unknown-camera'; },
    e => { e.world_frame_revision = null; e.world_transform_sha256 = null; },
  ];
  for (const mutate of mutations) {
    const e = entity(); mutate(e);
    const state = feed(initialized(), snapshot(1, [e]));
    assert.equal(state.people.length, 0);
    assert.equal(state.unavailable.length, 1);
  }
});

test('telemetry-only identity and per-camera BEV cannot create a person', () => {
  const wire = tracking(snapshot());
  wire.world_snapshot = null; wire.world_events = []; wire.tracks = [{ stable_id: 99, world: { x: 1, y: 0, z: 2 } }];
  const state = apply(initialized(), wire);
  assert.equal(state.people.length, 0);
  assert.equal(apply(state, envelope(snapshot())).diagnostic.accepted, false);
  assert.equal(apply(initialized(), { type: 'bev-frame', footpoints: [{ stableId: 99, x: 1, y: 2 }] }).diagnostic.reason, 'noncanonical_input');
});

test('strict snapshot sequence rejects replay, but regressed observation extent is allowed', () => {
  let state = feed(initialized(), snapshot());
  assert.equal(apply(state, envelope(snapshot())).diagnostic.accepted, false);
  const s2 = snapshot(2); s2.observed_end_us = 1_000_000; s2.observed_start_us = 1_000_000;
  state = feed(state, s2, 1010);
  assert.equal(state.cursor.sequence, 2);
  assert.equal(state.people.length, 1);
});

test('source epoch reset breaks trails and old epochs cannot return', () => {
  let state = feed(initialized(), snapshot());
  const s2 = snapshot(2, [entity(1_015_000)]);
  const c = { ...envelope(s2).cohort, frame_id: 0, tracking_publication_sequence: 0 };
  state = feed(state, s2, 1010, { source_epoch: 1, cohort: c, ...c });
  assert.equal(state.people[0].trailBreak, true);
  assert.equal(state.people[0].trail.length, 1);
  const rejected = apply(state, tracking(snapshot(3), [], { source_epoch: 0 }), 1020);
  assert.equal(rejected.diagnostic.reason, 'source_epoch_stale');
});

test('missing canonical/source sequence breaks trails instead of silently bridging a cohort gap', () => {
  let state = feed(initialized(), snapshot());
  state = feed(state, snapshot(3, [entity(1_025_000)]), 1020);
  assert.equal(state.diagnostic.reason, 'admitted_after_gap');
  assert.equal(state.people[0].trailBreak, true);
  assert.equal(state.people[0].trail.length, 1);
  const s4 = snapshot(4, [entity(1_035_000)]);
  const c = { ...envelope(s4).cohort, tracking_publication_sequence: 6 };
  state = feed(state, s4, 1030, { cohort: c, ...c });
  assert.equal(state.diagnostic.reason, 'admitted_after_gap');
  assert.equal(state.people[0].trail.length, 1);
});

test('producer run changes clear trails and retire the old run, including after disconnect', () => {
  let state = feed(initialized(), snapshot());
  const changed = snapshot(0, [entity(1_020_000)], { producer: { ...snapshot().producer, run_id: 'new' },
    published_at_us: 1_030_000, observed_end_us: 1_020_000 });
  state = feed(state, changed, 1020);
  assert.equal(state.people[0].trail.length, 1);
  assert.equal(state.retiredRuns.length, 1);
  state = clearUnifiedBevPeople(state);
  assert.equal(state.people.length, 0);
  assert.equal(apply(state, envelope(snapshot(100, [entity(2_000_000)])), 1030).diagnostic.reason, 'retired_run');
});

test('measured trails continue within one canonical segment; prediction/hold do not append', () => {
  let state = feed(initialized(), snapshot());
  state = feed(state, snapshot(2, [entity(1_015_000)]), 1010);
  assert.equal(state.people[0].trail.length, 2);
  assert.equal(state.people[0].trailBreak, false);
  for (const [seq, position_state] of [[3, 'predicted'], [4, 'held']]) {
    state = feed(state, snapshot(seq, [entity(1_015_000, { position_state })]), 1000 + seq * 10);
    assert.equal(state.people[0].trail.length, 2);
    assert.equal(state.people[0].admission, position_state);
  }
  state = feed(state, snapshot(5, [entity(1_040_000, { continuity_segment_id: 2 })]), 1050);
  assert.equal(state.people[0].trailBreak, true);
  assert.equal(state.people[0].trail.length, 1);
});

test('frame/calibration revision changes break trails; unavailable membership cannot reuse old trails', () => {
  let state = feed(initialized(), snapshot());
  const changed = entity(1_015_000, { calibration_revision: 'new-calibration' });
  changed.sources[0].calibration_revision = 'new-calibration';
  state = feed(state, snapshot(2, [changed]), 1010);
  assert.equal(state.people[0].trail.length, 1);
  assert.equal(state.people[0].trailBreak, true);
  state = feed(state, snapshot(3, [entity(1_020_000, { position: null, covariance: null })]), 1020);
  state = feed(state, snapshot(4, [entity(1_030_000)]), 1030);
  assert.equal(state.people[0].trail.length, 1);
});

test('authoritative absence/lost and receipt-anchored expiry remove head and trail', () => {
  const state = feed(initialized(), snapshot());
  assert.equal(feed(state, snapshot(2, []), 1010).people.length, 0);
  assert.equal(feed(state, snapshot(2, [entity(1_015_000, { lifecycle: 'lost' })]), 1010).people.length, 0);
  const deadline = state.people[0].expiresAtMs;
  assert.equal(advanceUnifiedBev(state, deadline - 0.1).people.length, 1);
  assert.equal(advanceUnifiedBev(state, deadline).people.length, 0);
});

test('same original held clocks never renew the deadline or resurrect an expired entity', () => {
  const e = entity(1_000_000, { position_state: 'held' });
  let state = feed(initialized(), snapshot(1, [e]));
  const deadline = state.people[0].expiresAtMs;
  state = feed(state, snapshot(2, [e]), 2000);
  assert.equal(state.people[0].expiresAtMs, deadline);
  state = advanceUnifiedBev(state, deadline);
  state = feed(state, snapshot(3, [e]), deadline + 100);
  assert.equal(state.people.length, 0);
});

test('rejected input still expires people at a valid receipt time; invalid clocks cannot advance state', () => {
  const state = feed(initialized(), snapshot());
  const deadline = state.people[0].expiresAtMs;
  const expired = apply(state, { type: 'bev-frame' }, deadline);
  assert.equal(expired.diagnostic.accepted, false);
  assert.equal(expired.people.length, 0);
  assert.equal(expired.lastReceivedAtMs, deadline);
  const invalid = apply(state, { type: 'bev-frame' }, 0);
  assert.equal(invalid.diagnostic.reason, 'receipt_clock_invalid');
  assert.deepEqual(invalid.people, state.people);
  assert.equal(invalid.lastReceivedAtMs, state.lastReceivedAtMs);
});

test('standalone world events must match their accepted exact snapshot and embedded order', () => {
  const s = snapshot();
  const event = { contract: 'noesis.world.event', contract_version: 1, event_id: 'event:1',
    producer: s.producer, sequence: 1, event_type: 'appeared', entity_id: s.entities[0].entity_id,
    subject: s.entities[0].subject, observed_at_us: s.entities[0].observed_at_us,
    published_at_us: s.published_at_us, frame: 'backend_world_m', units: 'meters', position: s.entities[0].position };
  let state = apply(initialized(), tracking(s, [event]));
  state = apply(state, envelope(s));
  const wire = { ...envelope(s), type: 'world_event', payload: event };
  const wrong = clone(wire); wrong.payload.position.x += 1;
  assert.equal(apply(state, wrong).diagnostic.reason, 'event_mismatch');
  const accepted = apply(state, wire);
  assert.equal(accepted.diagnostic.accepted, true);
  assert.equal(accepted.pending.length, 0);
  assert.deepEqual(accepted.people, state.people);
  assert.equal(apply(initialized(), wire).diagnostic.reason, 'event_without_exact_snapshot');
});

test('resource bounds cap trails and evidence; failed map/bundle update clears stale geometry', () => {
  let state = createUnifiedBevState({ authoredSceneSha256: DIGEST, maxTrailPoints: 2 });
  state = apply(state, { type: 'calibration-bundle', data: BUNDLE }, 0);
  for (let seq = 1; seq <= 5; seq++) state = feed(state, snapshot(seq, [entity(1_000_000 + seq * 1000)]), 1000 + seq);
  assert.equal(state.people[0].trail.length, 2);
  assert.throws(() => createUnifiedBevState({ maxTrailPoints: 129 }), /trail_limit_invalid/);
  const oversized = entity(); oversized.sources = Array.from({ length: 33 }, (_, i) => ({ ...oversized.sources[0], observation_id: `source-${i}` }));
  assert.equal(apply(initialized(), envelope(snapshot(1, [oversized]))).diagnostic.accepted, false);
  state = apply(state, MAPS[0], 1010);
  state = apply(state, { ...MAPS[0], error: 'unavailable' }, 1011);
  assert.equal(state.maps.length, 0);
  state = apply(state, { type: 'calibration-bundle', data: {} }, 1012);
  assert.equal(state.people.length, 0); assert.equal(state.bindings, null);
});

test('all reducer outputs are immutable and do not mutate their inputs', () => {
  const s = snapshot(); const before = clone(s);
  const state = feed(initialized(), s);
  assert.deepEqual(s, before);
  assert.ok(Object.isFrozen(state.people[0].sourceEvidence));
  assert.ok(Object.isFrozen(state.people[0].trail));
  assert.throws(() => state.people[0].position.push(9), TypeError);
});

// Synthetic component inputs. The Python XC3D fixture separately exercises
// both REAL World publications/events through the actual publisher and Node.
const physical = (time = 1_000_000, changes = {}) => entity(time, {
  entity_id: 'physical:run:1',
  subject: { subject_id: 'unknown:run:1', kind: 'unknown', generation: 0,
    stable_id: null, resident_uuid: null, display_name: null }, ...changes,
});
const promotedSubject = { subject_id: 'visitor:run:17:g4', kind: 'visitor', generation: 4,
  stable_id: 17, resident_uuid: null, display_name: null };

test('XC3D T6 physical promotion keeps truthful generation and original measured trail', () => {
  const first = feed(initialized(), snapshot(1, [physical()]));
  const second = feed(first, snapshot(2, [physical(1_015_000, { subject: promotedSubject })]), 1010);
  assert.equal(second.people[0].generation, 4);
  assert.equal(second.people[0].stableId, 17);
  assert.equal(second.people[0].trail.length, 2);
  assert.equal(second.people[0].trailBreak, false);
  assert.deepEqual(second.people[0].trail[0], first.people[0].trail[0]);
});

test('XC3D T6 legacy malformed other-run or absent-segment IDs retain generation/reuse safety', () => {
  for (const id of ['legacy:fixed', 'physical:other-run:1', 'physical:run:01', 'physical:run:0', 'physical:run:not-a-key']) {
    const first = feed(initialized(), snapshot(1, [physical(1_000_000, { entity_id: id })]));
    const second = feed(first, snapshot(2, [physical(1_015_000, { entity_id: id, subject: promotedSubject })]), 1010);
    assert.equal(second.people[0].generation, 4);
    assert.equal(second.people[0].trail.length, 1, id);
    assert.equal(second.people[0].trailBreak, true, id);
  }
  const noSegment = feed(initialized(), snapshot(1, [physical(1_000_000, { continuity_segment_id: null })]));
  assert.equal(feed(noSegment, snapshot(2, [physical(1_015_000,
    { continuity_segment_id: null, subject: promotedSubject })]), 1010).people[0].trail.length, 1);
  const old = feed(initialized(), snapshot(1, [physical(1_000_000, { subject: promotedSubject })]));
  const reused = feed(old, snapshot(2, [physical(1_015_000, { entity_id: 'physical:run:2',
    subject: { ...promotedSubject, generation: 5, subject_id: 'visitor:run:17:g5' } })]), 1010);
  assert.equal(reused.people[0].trail.length, 1);
  assert.equal(reused.people[0].generation, 5);
});

test('XC3D T6 physical epoch run absence sequence and segment changes still break trails', () => {
  for (const change of ['epoch', 'run', 'absence', 'sequence', 'segment']) {
    let first = feed(initialized(), snapshot(1, [physical()]));
    let sequence = change === 'sequence' ? 3 : 2;
    const e = physical(1_015_000, { subject: promotedSubject,
      continuity_segment_id: change === 'segment' ? 2 : 1 });
    const options = change === 'run' ? { producer: { ...snapshot().producer, run_id: 'next-run' } } : {};
    if (change === 'run') e.entity_id = 'physical:next-run:1';
    if (change === 'absence') {
      first = feed(first, snapshot(2, []), 1005);
      sequence = 3;
    }
    const second = feed(first, snapshot(sequence, [e], options), 1010,
      change === 'epoch' ? { source_epoch: 1 } : {});
    assert.equal(second.people[0].generation, 4, change);
    assert.equal(second.people[0].trail.length, 1, change);
    assert.equal(second.people[0].trailBreak, true, change);
  }
});

test('XC3D T6 physical frame transform calibration and scene mapping revisions still break trails', () => {
  for (const change of ['frame', 'transform', 'calibration', 'mapping']) {
    let first = feed(initialized(), snapshot(1, [physical()]));
    const e = physical(1_015_000, { subject: promotedSubject });
    if (change === 'calibration') {
      e.calibration_revision = 'new-exact-calibration';
      e.sources[0].calibration_revision = e.calibration_revision;
    } else {
      const b = clone(BUNDLE), frame = b.cameras.frame_bindings['family-room'];
      const oldRevision = frame.world_frame.revision;
      const presentation = b.world_frame_presentations[oldRevision];
      if (change === 'frame') {
        const revision = 'synthetic-next-world-revision';
        frame.world_frame.revision = revision;
        presentation.world_frame.revision = revision;
        delete b.world_frame_presentations[oldRevision];
        b.world_frame_presentations[revision] = presentation;
        e.world_frame_revision = revision;
        e.sources[0].world_frame_revision = revision;
      } else if (change === 'transform') {
        const digest = 'a'.repeat(64);
        frame.transform_sha256 = digest;
        presentation.source_transform_sha256s = [digest];
        e.world_transform_sha256 = digest;
        e.sources[0].world_transform_sha256 = digest;
      } else {
        presentation.world_to_scene_sha256 = 'b'.repeat(64);
        presentation.scene_revision_id = 'a'.repeat(64);
      }
      first = apply(first, { type: 'calibration-bundle', data: b }, 1005);
      assert.equal(first.diagnostic.accepted, true, first.diagnostic.reason);
    }
    const second = feed(first, snapshot(2, [e]), 1010);
    assert.equal(second.people[0].generation, 4, change);
    assert.equal(second.people[0].trail.length, 1, change);
    assert.equal(second.people[0].trailBreak, true, change);
  }
});

test('XC3D T6 promotion in held or predicted state never appends or renews old age', () => {
  let state = feed(initialized(), snapshot(1, [physical()]));
  const original = clone(state.people[0].trail);
  const deadline = state.people[0].expiresAtMs;
  for (const [sequence, position_state] of [[2, 'held'], [3, 'predicted']]) {
    state = feed(state, snapshot(sequence, [physical(1_000_000,
      { subject: promotedSubject, position_state })]), 1000 + sequence * 10);
    assert.equal(state.people[0].generation, 4);
    assert.deepEqual(state.people[0].trail, original);
    assert.equal(state.people[0].expiresAtMs, deadline);
  }
});

function attributionR4Fixture(time = 1_000_000) {
  const owners = [1, 2].map((serial) => ({ run_id: 'run', physical_serial: serial,
    camera_id: 'family-room', source_id: 2, source_epoch: 0, raw_id: serial, lifecycle_generation: 1,
    lifecycle_started_frame: 1, frame_id: 2, observation_id: `synthetic:${serial}:2`, acquired_at_s: 10.2,
    clock_domain: 'synthetic', media_time_s: .2, clock_basis: 'synthetic', household_state: 'resident',
    household_identity_key: 'synthetic-same', household_stable_id: 9,
    household_visitor_generation: null, household_constraint: 'eligible' }));
  const payload = { status: 'quarantined', purpose: 'physical_only_quarantine',
    reason: 'settled_household_claim_conflict', run_id: 'run', association_revision: 3,
    subject_ids: ['resident:synthetic-same'], stable_ids: [9], owners, retained_owners: [],
    component_entity_ids: ['physical:run:1', 'physical:run:2'] };
  const sorted = (v) => Array.isArray(v) ? v.map(sorted) : v && typeof v === 'object'
    ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, sorted(v[k])])) : v;
  const q = { ...payload, conflict_id: createHash('sha256').update(JSON.stringify(sorted(payload))).digest('hex') };
  return [1, 2].map((serial) => physical(time, { entity_id: `physical:run:${serial}`,
    subject: { subject_id: `quarantine:run:${serial}`, kind: 'provisional', generation: 0,
      stable_id: null, resident_uuid: null, display_name: null }, attribution_quarantine: q }));
}

test('XC3D R4 explicit quarantine preserves two physical trails and truthful label', () => {
  let state = feed(initialized(), snapshot(1, attributionR4Fixture()));
  assert.equal(state.diagnostic.accepted, true);
  assert.equal(state.people.length, 2);
  state = feed(state, snapshot(2, attributionR4Fixture(1_015_000)), 1010);
  for (const p of state.people) {
    assert.equal(p.label, 'Person (attribution quarantined)');
    assert.equal(p.stableId, null);
    assert.equal(p.identityKind, 'provisional');
    assert.equal(p.trail.length, 2);
    assert.equal(p.attributionQuarantine.status, 'quarantined');
    assert.ok(Object.isFrozen(p.attributionQuarantine.owners));
  }
});

function attributionR19Fixture(time = 1_000_000) {
  const row = attributionR4Fixture(time)[0]; const q = row.attribution_quarantine;
  q.component_entity_ids = ['physical:run:1'];
  q.owners[1] = { ...q.owners[1], physical_serial: 1, camera_id: 'kitchen', source_id: 1,
    household_identity_key: 'synthetic-other', household_stable_id: 10 };
  q.subject_ids = ['resident:synthetic-other', 'resident:synthetic-same']; q.stable_ids = [9, 10];
  const { conflict_id: ignored, ...payload } = q;
  const sorted = (v) => Array.isArray(v) ? v.map(sorted) : v && typeof v === 'object'
    ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, sorted(v[k])])) : v;
  q.conflict_id = createHash('sha256').update(JSON.stringify(sorted(payload))).digest('hex');
  return row;
}
test('XC3D R19 one physical person keeps both divergent claims without label or trail loss', () => {
  let state = feed(initialized(), snapshot(1, [attributionR19Fixture()]));
  assert.equal(state.diagnostic.accepted, true); assert.equal(state.people.length, 1);
  state = feed(state, snapshot(2, [attributionR19Fixture(1_015_000)]), 1010);
  assert.equal(state.diagnostic.accepted, true);
  assert.equal(state.people[0].trail.length, 2); assert.equal(state.people[0].stableId, null);
  assert.equal(state.people[0].attributionQuarantine.owners.length, 2);
});
for (const defect of ['same-descriptor', 'same-member', 'unresolved', 'partial', 'extra-entity']) {
  test(`XC3D R19 refuses invalid singleton ${defect} atomically`, () => {
    const state = feed(initialized(), snapshot(1, [attributionR19Fixture()]));
    const row = attributionR19Fixture(1_015_000); const q = row.attribution_quarantine;
    if (defect === 'same-descriptor') Object.assign(q.owners[1], { household_identity_key: 'synthetic-same', household_stable_id: 9 });
    else if (defect === 'same-member') Object.assign(q.owners[1], { camera_id:q.owners[0].camera_id, source_id:q.owners[0].source_id, raw_id:q.owners[0].raw_id, frame_id:3 });
    else if (defect === 'unresolved') q.owners[1].household_constraint = 'unresolved';
    else if (defect === 'partial') q.stable_ids = [9];
    else q.retained_owners = [{ ...q.owners[0], physical_serial:3 }];
    const rejected = apply(state, envelope(snapshot(2, [row])), 1010);
    assert.equal(rejected.diagnostic.accepted, false); assert.deepEqual(rejected.people, state.people);
  });
}

for (const defect of ['missing', 'settled', 'partial', 'component', 'purpose', 'claims', 'duplicate-owner']) {
  test(`XC3D R4 rejects ${defect} quarantine without replacing prior heads`, () => {
    const state = feed(initialized(), snapshot(1, attributionR4Fixture()));
    const rows = attributionR4Fixture(1_015_000);
    if (defect === 'missing') rows[0].attribution_quarantine = null;
    else if (defect === 'settled') rows[0].subject.stable_id = 9;
    else if (defect === 'partial') {
      rows[1].attribution_quarantine = null;
      rows[1].subject = { subject_id: 'resident:synthetic-same', kind: 'resident', generation: 0,
        resident_uuid: 'synthetic-same', stable_id: 9, display_name: null };
    } else if (defect === 'component') rows[0].attribution_quarantine.component_entity_ids.push('physical:run:99');
    else if (defect === 'purpose') rows[0].attribution_quarantine.purpose = 'strict';
    else if (defect === 'claims') rows[0].attribution_quarantine.subject_ids = [];
    else rows[0].attribution_quarantine.owners[1] = clone(rows[0].attribution_quarantine.owners[0]);
    const rejected = apply(state, envelope(snapshot(2, rows)), 1010);
    assert.equal(rejected.diagnostic.accepted, false);
    assert.deepEqual(rejected.people, state.people);
  });
}

test('XC3D T6 duplicate identity refusal never zeroes prior legitimate state', () => {
  const state = feed(initialized(), snapshot(1, [physical()]));
  const second = physical(1_015_000, { entity_id: 'physical:run:2' });
  const rejected = apply(state, envelope(snapshot(2, [physical(), second])), 1010);
  assert.equal(rejected.diagnostic.reason, 'canonical_identity_duplicate');
  assert.deepEqual(rejected.people, state.people);
  assert.deepEqual(rejected.cursor, state.cursor);
});

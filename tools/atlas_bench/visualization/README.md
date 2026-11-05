# Local review outputs

Successful runs save a contact sheet with returned RGB, a labeled per-view
5th–95th percentile depth preview, and confidence when present. Black invalid
depth stays unknown. The EXRs, cameras and source RGB remain unchanged.

Later experiments should put side-by-side real/generated views, matched depth
residuals, visibility masks and registered point views under their separate
external derived/report directories. Match grids and exposure explicitly;
preview normalization cannot establish geometric accuracy. No viewer writes
back into PCF or runtime presentation.

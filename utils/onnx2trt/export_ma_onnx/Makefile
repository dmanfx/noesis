# Absolute workspace path
WORKSPACE = /home/mayor/Noesis_Devel
# MapAnything repo path
REPO_PATH = $(WORKSPACE)/external/map-anything
# Python executable (system)
PY = python3
# Output dir for ONNX files
OUTDIR ?= ma_onnx_out
# Image dimensions
IMG_H ?= 512
IMG_W ?= 512
# ONNX opset
OPSET ?= 18
# Optional checkpoint path
CHECKPOINT_PATH ?=
# Repo metadata for report
REPO_URL ?= https://github.com/facebookresearch/map-anything
REPO_BRANCH ?= main

.PHONY: export clean report

export:
	$(PY) $(WORKSPACE)/export_ma_onnx/export_to_onnx.py \
		--repo $(REPO_PATH) \
		--outdir $(OUTDIR) \
		--h $(IMG_H) \
		--w $(IMG_W) \
		--opset $(OPSET) \
		--ckpt "$(CHECKPOINT_PATH)" \
		--repo-url $(REPO_URL) \
		--repo-branch $(REPO_BRANCH)

report:
	@cat $(OUTDIR)/export_report.txt || echo "No report found; run 'make export' first."

clean:
	rm -rf $(OUTDIR)
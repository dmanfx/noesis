# Rejected review artifact

Status: rejected 2026-08-15 after physical-layout review.

The images and `alignment_report.json` in this directory projected backend-world
X directly to screen. That made Kitchen camera-left appear on the wrong side and
produced a misleading footprint even though the stored room geometry had not
been moved.

Do not use this directory as Kitchen/Family Room alignment evidence. The
corrected review holds the Family Room camera-oriented presentation fixed,
projects the Kitchen/Family overlap through the static Kitchen camera, and
scores discrete Kitchen orientation hypotheses:

```text
artifacts/pcf_multiroom/kitchen_family_corrected_20260815/
```

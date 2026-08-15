# Validation report examples

A concise report is enough:

```text
scope: tracking -> BEV -> Menon for living-room
mode: live native DS9.1, 20 seconds
inputs: runtime run ID, camera ID, calibration/scene revision
result: pass
observed: source frames advanced; stable_id and world sequence paired;
          Menon point/trail advanced in the declared scene frame
limits: no labeled identity-accuracy claim
```

For a blocked result, name the exact missing required input and stop. Do not
replace it with a broader or different validation mode.

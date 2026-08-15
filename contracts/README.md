# Noesis product contracts

`schema/` and `typescript/noesis-contracts.ts` are generated from strict,
runtime-neutral Pydantic models under `noesis_core/contracts/`.

```bash
python3 scripts/export_noesis_core_schemas.py
python3 scripts/export_noesis_core_schemas.py --check
```

Wire payloads carry `contract` and `contract_version`; consumers reject missing
or unsupported versions. `fixtures/v1/` contains non-secret characterization
inputs. Deterministic runtime evidence uses the validated, hash-chained writers
under `noesis_core`.

The canonical native DeepStream 9.1 application consumes these shared product
contracts through `DS9/noesis/`. Older appliance selector and
`noesis-runtime-v1` checkout-identity material remains implemented for
historical compatibility but is not part of ordinary runtime activation or
development. Its former documentation is archived under
`docs/history/ds9_container/`.

For current wire behavior, use `docs/api_contracts_ws.md`,
`docs/api_contracts_rest.md`, and `docs/metadata_contracts.md`.

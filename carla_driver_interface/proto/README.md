# Protocol definitions

| Directory | What | Source |
| --- | --- | --- |
| `driver_extension/v0/` | This project's extension messages, carried inside alpasim's `bytes` extension points | This repository |
| `alpasim_grpc/v0/` | alpasim's `egodriver` contract and what it imports (`common`, `sensorsim`, `runtime`) | [NVlabs/alpasim](https://github.com/NVlabs/alpasim) `68709245a5dc0f2eda4f8cb2c3aa8cbdfa913043`, `src/grpc/alpasim_grpc/v0/` (`alpasim_grpc` 0.55.0), Apache-2.0 (`LICENSE.alpasim`) |

## Why alpasim's are vendored

`alpasim-grpc` declares `requires-python = ">=3.11,<3.13"`. Autoware's
environment is Python 3.10, and the scenario framework this package lives
beside supports it, so depending on `alpasim-grpc` would drop 3.10. The files are
copied **verbatim** and never edited: package names, field numbers and the
`alpasim_grpc/v0/...` import paths are upstream's, which is what keeps the
messages wire-compatible with an alpasim runtime or driver.

`scripts/compile_protos.py` compiles both sets into
`src/carla_driver_interface/grpc_api/_proto/`, with the cross-imports made
relative so nothing claims the top-level `alpasim_grpc` name. Regenerate after
any change here:

```console
$ uv run python scripts/compile_protos.py
```

`tests/test_proto_compat.py` fails when the committed modules drift from the
protos.

## Moving to a newer alpasim

1. Copy `common.proto`, `sensorsim.proto`, `egodriver.proto` and `runtime.proto`
   from the new revision into `alpasim_grpc/v0/`.
2. Update the revision here, `ALPASIM_GRPC_REV` in
   `src/carla_driver_interface/__init__.py`, and `API_VERSION_MESSAGE` in
   `src/carla_driver_interface/grpc_api/__init__.py` (the `alpasim-grpc`
   version at that revision).
3. Regenerate and run the tests.

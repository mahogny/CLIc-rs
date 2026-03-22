# clesperanto-rs

A pure Rust rewrite of [CLIc](https://github.com/clEsperanto/CLIc) — the GPU-accelerated image processing backend of the [clEsperanto](https://github.com/clEsperanto) ecosystem (pyclesperanto, clesperantoj, Fiji/clij3).

No C++ FFI. OpenCL via the [`opencl3`](https://crates.io/crates/opencl3) crate.

This code was generated through automatic translation. The C++ version of CLIc is authorative and improvements
should first be made to CLIc and clEsperanto. Improvements to CLIc-rs should primarily focus on translation errors
or making the API easier to use.

## Status

| Tier | Functions | Tests |
|------|-----------|-------|
| tier1 | copy, gaussian_blur, add/subtract/multiply/divide scalars & images, absolute, power, projections, filters | 11 GPU |
| tier2 | absolute_difference, add/subtract images, clip, difference_of_gaussian, morphological ops (opening, closing, top-hat, bottom-hat, std-dev), global reductions | 15 GPU |
| tier3 | mean_of_all_pixels, gamma_correction | 4 GPU |
| tier4 | mean_squared_error | (included above) |
| tier5 | array_equal | 3 GPU |
| tier7 | translate, scale (affine transforms via `affine_transform.cl`) | 5 GPU |

## Usage

```rust
use clic_rs::{BackendManager, array::{push, pull}, tier1};

let device = BackendManager::get().get_device("", "gpu").unwrap();
let src = push(&vec![1.0f32; 100], 10, 10, 1, &device).unwrap();
let dst = tier1::gaussian_blur(&device, &src, None, 1.0, 1.0, 0.0).unwrap();
let result: Vec<f32> = pull(&dst).unwrap();
```

## Building

Requires an OpenCL runtime (e.g. from your GPU driver or [PoCL](https://portablecl.org) for CPU fallback).

```bash
cargo build
cargo test                        # unit tests (no GPU required)
cargo test --features gpu-tests   # integration tests (requires OpenCL device)
```

## Architecture

```
clic-rs/src/
  execution.rs        # generate_defines() + execute() — core kernel dispatch
  array.rs            # Arc<Mutex<Array>> (ArrayPtr) — GPU memory lifecycle
  backend.rs          # OpenCL backend: allocate, read, write, execute kernels
  device.rs           # OpenCLDevice — context, queue, program cache
  cache.rs            # LRU program cache + SHA-256 disk cache (~/.cache/clesperanto/)
  tier0.rs            # Array creation helpers (create_like, create_one, …)
  tier1/              # Elementary ops: math, filters, projections, blur
  tier2/              # Compositions: morphology, reductions, clipping
  tier3/ … tier7/     # Higher-level compositions
clic-rs/kernels/      # Vendored .cl files from clij-opencl-kernels 3.5.3
CLIc/                 # C++ reference implementation (read-only, for comparison)
```

The execution model mirrors CLIc: `generate_defines()` builds a `#define` preamble encoding array dimensions and data-type macros, which is prepended to the kernel source before compilation. Compiled programs are cached in memory (LRU, 128 entries) and on disk (SHA-256 keyed).

## License

Same as CLIc — see [CLIc/LICENSE](CLIc/LICENSE).

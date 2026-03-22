# clic-rs

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

## Benchmarks

Measured on Apple Silicon (Intel GPU, macOS 15.7, OpenCL). Both implementations synchronize the GPU after each operation (`clFinish`). clic-rs benefits from an in-memory LRU kernel program cache, avoiding recompilation on repeated calls.

| Operation | Image size | CLIc (C++) | clic-rs (Rust) |
|-----------|------------|------------|----------------|
| `gaussian_blur` | 64×64 | 3768 µs | 1293 µs |
| `gaussian_blur` | 256×256 | 3321 µs | 1579 µs |
| `gaussian_blur` | 512×512 | 5272 µs | 2833 µs |
| `add_images_weighted` | 64×64 | 1232 µs | 783 µs |
| `add_images_weighted` | 256×256 | 1200 µs | 952 µs |
| `add_images_weighted` | 512×512 | 1456 µs | 1386 µs |
| `mean_of_all_pixels` | 64×64 | 3939 µs | 1116 µs |
| `mean_of_all_pixels` | 256×256 | 2561 µs | 1272 µs |
| `mean_of_all_pixels` | 512×512 | 3505 µs | 1544 µs |

Run benchmarks:

```bash
bash benchmark/run.sh          # compare C++ CLIc vs clic-rs side by side
cargo bench --bench gpu        # Rust only (Criterion HTML report)
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

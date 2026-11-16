# CS465 Milestone 1 — Matrix Multiplication Optimization

Course project for **CS 465 Computer Systems Architecture**: optimizing a 1024×1024
matrix multiplication on contemporary platforms, per the project handout.

**Team:** Biruk Temesgen, Mubashar Ikram, Prafula Tamang, Hamza Khan, Michael Hoang, Michael Sun

| Thread | Members | Folder |
|---|---|---|
| Memory hierarchy (tiling/blocking, Zeus) | Biruk Temesgen, Mubashar Ikram | `memory/` (to be added) |
| Multicore (pthreads + OpenMP, Zeus) | Hamza Khan, Michael Hoang | `multicore/` (to be added) |
| **GPU (CUDA, with and without tiling)** | Prafula Tamang, Michael Sun | [`gpu/`](gpu/) |

This repository currently contains the **GPU thread** results for Milestone 1.
Other threads will be added by their owners.

## GPU thread summary

- **Hardware:** NVIDIA GeForce RTX 4070 (12 GB, Ada Lovelace, compute capability 8.9), driver 610.88
- **Toolchain:** CUDA toolkit 13.3 (`nvcc`), gcc 13.3, Ubuntu 24.04 under WSL2
- **Problem:** C = A × B, 1024×1024 floats, `srand(10)` then `(float)(rand() % 10)`
  (values 0–9 keep every partial sum ≤ 1024×81 = 82,944, exactly representable in
  float, so GPU and CPU results match bit-for-bit). Two CUDA versions, per the
  handout: with and without tiling/blocking (shared-memory tiles).
- **Timing:** `clock_gettime(CLOCK_MONOTONIC)` around the kernel launch +
  `cudaDeviceSynchronize()` only — matrix initialization and PCIe copies are
  excluded, per the handout. A non-timed warm-up launch precedes each timed launch.
- **Correctness:** every variant verified element-by-element against a CPU
  recomputation, and all 30 timed runs printed the identical checksum
  **21733638588**.

### Results (kernel time, average of 5 runs)

| Block size | Without tiling/blocking | With tiling/blocking | Speedup |
|---|---|---|---|
| 8×8 | 1.956 ms | 1.336 ms | 1.46× |
| 16×16 | **1.387 ms** | **1.185 ms** | 1.17× |
| 32×32 | 1.484 ms | 1.356 ms | 1.09× |

**Answer to handout Q4 (this GPU):** the optimal block size is **16×16** for both
the version without tiling/blocking and the version with it; tiling/blocking gives
~1.1–1.5×. The gain is smaller than the classic textbook 2–4× because both input
matrices (8 MB each) fit entirely in the RTX 4070's 32 MB L2 cache, so even the
untiled kernel's reads mostly hit fast cache. PCIe copies (untimed): ~2.1 ms HtoD +
~3.5 ms DtoH — together larger than the kernel itself.

## Build and run (Linux / WSL2)

```bash
cd gpu
bash run_all.sh          # builds BS=8/16/32, verifies vs CPU, runs 30 timed runs
```

Manual build of a single configuration:

```bash
nvcc -arch=native -DBS=16 mm.cu -o mm16
./mm16 0                 # mode 0 = naive
./mm16 1                 # mode 1 = tiled
./mm16 1 1               # extra arg = also verify against CPU
```

## Files

| File | Purpose |
|---|---|
| `gpu/mm.cu` | Both kernels, warm-up, `clock_gettime` timing, checksum, CPU verify |
| `gpu/run_all.sh` | Build + verify + 30 timed runs (results collector) |
| `gpu/make_chart.py` | Parses results, prints averages table, draws the bar chart |
| `gpu/gpu_chart.png` | Chart used on the slides |
| `gpu/results_NVIDIA_GeForce_RTX_4070.txt` | Raw data, reported session |
| `gpu/results_NVIDIA_GeForce_RTX_4070_run1.txt` | Raw data, first session (reproducibility) |
| `gpu/mm_windows_test.cu` | Native Windows cross-check build (QueryPerformanceCounter shim) |
| `gpu/build_windows.bat` | Build recipe for the native Windows cross-check |

## Notes

- The handout's `-O0` rule applies to Zeus compilations; GPU code is compiled with
  `nvcc -arch=native` (device code optimization is controlled by nvcc, and no
  host-side timing loops exist to optimize).
- A native Windows (MSVC + nvcc 13.3) build gives the same kernel times; its
  checksum differs (21742160178) because MSVC's `rand()` differs from glibc's
  for the same seed — different inputs, still a correct product. All reported
  numbers come from the Linux build.

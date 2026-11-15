# Team C (GPU thread) — Prafula's notes

## Email to send the instructor (send today)
Subject: [CS465 Fall 2026 Project] GPU thread — float input question

Hi Professor,

For the GPU thread, `rand()` returns integers, so we plan to populate the input
matrices with `(float)(rand() % 10)` after `srand(10)` — values 0–9 stored as
floats. This keeps every intermediate sum <= 1024*81 = 82,944, exactly
representable in float, so GPU and CPU results match bit-for-bit. Is this
acceptable? Also, should both A and B be populated from the same `rand()`
stream (A filled first, then B), as we assumed?

Thanks,
Team C (Prafula Tamang, Michael Sun)

## Environment used
- GPU: NVIDIA GeForce RTX 4070 (12 GB, laptop/desktop, Ada Lovelace, compute capability 8.9)
- Driver: 610.88, CUDA 13.3 runtime
- Toolchain (results reported above): WSL2 Ubuntu 24.04 + CUDA toolkit 13.3 (nvcc), gcc 13.3
- Timing: clock_gettime(CLOCK_MONOTONIC) around kernel launch + cudaDeviceSynchronize, per handout
- Warm-up launch before every timed launch so one-time GPU startup is not counted

## Native Windows toolchain (also installed 2026-11-15, verified working)
- VS Build Tools 2026 (MSVC 19.50) + CUDA Toolkit 13.3 for Windows at
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3 (nvcc on user PATH).
- Build recipe: gpu\build_windows.bat (calls vcvars64, then nvcc -arch=native).
- mm_windows_test.cu = mm.cu with a Windows timing shim (QueryPerformanceCounter
  instead of POSIX clock_gettime, which MSVC lacks). Graded mm.cu is unchanged.
- IMPORTANT: a native Windows run prints checksum 21742160178, not 21733638588,
  because MSVC's rand() differs from glibc's rand() (same srand(10), different
  sequences). Both are internally correct (verify: OK); all reported numbers
  come from the WSL/Linux build for consistency with the handout's rules.

## Challenges encountered (for slide 5 — GPU bullets)
- `nvcc` was not available on Windows; no Visual Studio installed. Fixed by
  using WSL2 (Ubuntu 24.04) with the CUDA toolkit — the RTX 4070 is passed
  through to WSL, so kernels run on the real GPU.
- Mid-install disk failure: the C: drive filled up during the CUDA toolkit
  install, which corrupted the first WSL distro (dpkg I/O errors, distro would
  not start). Fixed by cleaning temp files and creating a fresh distro on the
  E: drive with plenty of space, then reinstalling the toolkit there.
- Float inputs need care: `rand()` returns ints. We used `(float)(rand()%10)`
  so every partial sum stays exactly representable and the checksum matches
  the CPU verification (pending instructor confirmation).
- GPU timing trap: without `cudaDeviceSynchronize()` after the launch,
  `clock_gettime` measures only the async launch (~microseconds). We
  synchronize before stopping the clock.
- First launch is slower (GPU startup); a warm-up launch before the timed one
  removes that from the measurement.

## Results — NVIDIA GeForce RTX 4070, nvcc 13.3, -arch=native (sm_89), 5-run averages
(Rerun 2026-11-15 in Ubuntu-24.04 after distro repair; results file:
results_NVIDIA_GeForce_RTX_4070.txt, earlier session kept as *_run1.txt)

| Block size | Naive (s) | Tiled (s) | Speedup |
|---|---|---|---|
| 8x8 | 0.001956 | 0.001336 | 1.46x |
| 16x16 | 0.001387 | 0.001185 | 1.17x |
| 32x32 | 0.001484 | 0.001356 | 1.09x |

- Best naive block size: 16x16 (1.387 ms)
- Best tiled block size: 16x16 (1.185 ms) -> ~1.17x faster than best naive
- All 6 variants verified bit-exact against a CPU check, and every one of the
  30 runs printed checksum=21733638588 (single distinct value, identical to
  the first run - same GPU, same inputs, two different WSL distros).
- Reproducibility: first run (in the cuda-dev distro) gave best tiled 32x32
  at 0.999 ms; this run gave 16x16 at 1.185 ms. Both runs agree that naive is
  fastest at 16x16 and that tiling helps; the tiled 16-vs-32 winner flips
  within ~0.2 ms of run-to-run noise (GPU also drives the display while
  browser/Discord run). Safe presentation line: tiling gives ~1.1-1.5x, best
  block size 16x16 (naive) / 16-32 tiled.
- Copy times (not part of the kernel): h2d ~2.1 ms, d2h ~3.3 ms average —
  together they cost more than the kernel itself, worth one slide bullet.

Notes for slide 3:
- kernel time only (h2d/d2h copy times reported separately)
- checksum identical across all 6 variants (proof of same output)
- Question 4 asks "various Nvidia GPU platforms" — Michael Sun can run the
  same mm.cu on Google Colab's T4 for a second data point.

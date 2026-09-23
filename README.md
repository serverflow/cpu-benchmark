# SFBench for Intel Xeon Phi (K1OM)

**Frozen video edition: beta 0.48. This branch will not be updated.**

This branch contains the Xeon Phi Knights Corner build used for the video. It is
separate from `main`, which targets ordinary CPUs. The prebuilt `artifacts/cpu_benchmark`
is the preserved video binary, not a fresh or approximate rebuild. It was built
with K1OM GCC 5.1.1 and reports `SFBench beta 0.48`.

## Contents

- `artifacts/cpu_benchmark`: K1OM benchmark executable from the video backup.
- `artifacts/mic_showcase`: companion K1OM showcase executable.
- `src/`, `cmake/`, `CMakeLists.txt`: source snapshot for the beta 0.48 K1OM build.
- `video_cases/01_sfbench_fp64.sh`: FP64 scaling and native FP64/FP32 throughput.
- `video_cases/07_live_thread_monitor.sh`: live usage of all 240 logical CPUs.

The source comes from the archived beta 0.48 production build. Only version
metadata (`MINOR=48`), the K1OM-only CMake guard and the benchmark script's
optional installation-path override were changed for this frozen branch. These
changes do **not** change the preserved executable in `artifacts/`.

## Run on Phi

Copy the two executables and `video_cases/` to the already configured Phi,
for example to `/home/testuser/sfbench/`. Do not try to execute K1OM binaries
on the x86-64 host. On `mic0`:

```sh
cd /home/testuser/sfbench
chmod +x cpu_benchmark mic_showcase video_cases/*.sh
./cpu_benchmark --version
./video_cases/01_sfbench_fp64.sh
```

In a second SSH terminal, start the live thread monitor before the benchmark:

```sh
cd /home/testuser/sfbench
./video_cases/07_live_thread_monitor.sh
```

The benchmark script saves JSON and stderr per run under `results/fairness_*`.
Its default root is `/home/testuser/sfbench`; set `SFBENCH_ROOT` to use a
different installation directory.

## Rebuild

Requires the Intel MPSS 3.8.6 K1OM cross-toolchain and a compatible Linux build
host. This repository does not include Intel's SDK or MPSS packages.

```sh
cmake -S . -B build-k1om \
  -DCMAKE_TOOLCHAIN_FILE=cmake/k1om-gcc.cmake \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-k1om -j4
```

The branch rejects a non-K1OM CMake target. Rebuilding can produce different
binary bytes; use `artifacts/cpu_benchmark` to reproduce the video edition.

## Measurement notes

The compute score uses a one-lane FP64 scoring path for cross-architecture
comparison. The precision test separately reports full-width native IMCI
FP64 and FP32 throughput. Do not infer the score from the native IMCI GFLOPS.
The 60/120/180/240-thread runs use respectively 1/2/3/4 hardware threads per
physical core. Scores and throughput depend on clock, thermals and system load.

SHA-256 checksums are in [`artifacts/SHA256SUMS.txt`](artifacts/SHA256SUMS.txt).

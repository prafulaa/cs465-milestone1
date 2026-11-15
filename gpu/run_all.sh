#!/bin/bash
# Team C (GPU) build + verify + timing runs, per the guide.
# Run inside WSL:  bash run_all.sh
set -e
cd "$(dirname "$0")"

NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
ARCH=${ARCH:--arch=native}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')

echo "== nvcc =="
$NVCC --version | tail -2
echo "== GPU: $GPU_NAME =="

echo "== building BS=8/16/32 =="
for bs in 8 16 32; do
  $NVCC $ARCH -DBS=$bs mm.cu -o mm_bs$bs
done

echo "== correctness check (naive + tiled, once per build) =="
./mm_bs8  0 1
./mm_bs8  1 1
./mm_bs16 0 1
./mm_bs16 1 1
./mm_bs32 0 1
./mm_bs32 1 1

echo "== timing: 5 runs x {8,16,32} x {naive,tiled} =="
OUT=results_${GPU_NAME}.txt
{
  echo "# GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
  echo "# nvcc: $($NVCC --version | tail -1)"
  for bs in 8 16 32; do
    for mode in 0 1; do
      for r in 1 2 3 4 5; do ./mm_bs$bs $mode; done
    done
  done
} | tee "$OUT"

echo "== checksums (must all match) =="
grep -o 'checksum=[0-9]*' "$OUT" | sort -u
echo "DONE"

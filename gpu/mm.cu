#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 1024
#ifndef BS
#define BS 16            // block size AND tile size; set with -DBS=8/16/32
#endif

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); \
  exit(1); } } while (0)

static double now(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

// ---- Mode 0: naive. One thread computes one cell of C. ----
__global__ void matmul_naive(const float *A, const float *B, float *C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < N) {
    float s = 0.0f;
    for (int k = 0; k < N; k++) s += A[row * N + k] * B[k * N + col];
    C[row * N + col] = s;
  }
}

// ---- Mode 1: tiled. The block loads BSxBS pieces into fast shared memory. ----
__global__ void matmul_tiled(const float *A, const float *B, float *C) {
  __shared__ float As[BS][BS];
  __shared__ float Bs[BS][BS];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * BS + ty;
  int col = blockIdx.x * BS + tx;
  float s = 0.0f;
  for (int t = 0; t < N / BS; t++) {          // N is divisible by 8, 16, 32
    As[ty][tx] = A[row * N + t * BS + tx];
    Bs[ty][tx] = B[(t * BS + ty) * N + col];
    __syncthreads();                           // wait until the whole tile is loaded
    for (int k = 0; k < BS; k++) s += As[ty][k] * Bs[k][tx];
    __syncthreads();                           // wait before overwriting the tile
  }
  C[row * N + col] = s;
}

int main(int argc, char **argv) {
  int mode = (argc > 1) ? atoi(argv[1]) : 0;   // 0 = naive, 1 = tiled
  int verify = (argc > 2) ? atoi(argv[2]) : 0; // 1 = also check against CPU

  size_t bytes = (size_t)N * N * sizeof(float);
  float *hA = (float *)malloc(bytes), *hB = (float *)malloc(bytes), *hC = (float *)malloc(bytes);

  srand(10);
  for (int i = 0; i < N * N; i++) hA[i] = (float)(rand() % 10);
  for (int i = 0; i < N * N; i++) hB[i] = (float)(rand() % 10);

  float *dA, *dB, *dC;
  CHECK(cudaMalloc(&dA, bytes)); CHECK(cudaMalloc(&dB, bytes)); CHECK(cudaMalloc(&dC, bytes));

  double t0 = now();
  CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
  double t_h2d = now() - t0;

  dim3 block(BS, BS);
  dim3 grid(N / BS, N / BS);

  // Warm-up launch (not timed): the first launch pays one-time GPU startup cost.
  if (mode == 0) matmul_naive<<<grid, block>>>(dA, dB, dC);
  else           matmul_tiled<<<grid, block>>>(dA, dB, dC);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  // Timed run: just before launch, just after the GPU finishes.
  double t1 = now();
  if (mode == 0) matmul_naive<<<grid, block>>>(dA, dB, dC);
  else           matmul_tiled<<<grid, block>>>(dA, dB, dC);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());          // without this you time only the launch
  double t_kernel = now() - t1;

  t0 = now();
  CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
  double t_d2h = now() - t0;

  double sum = 0.0;
  for (int i = 0; i < N * N; i++) sum += hC[i];

  printf("mode=%s BS=%d kernel=%.6f s  h2d=%.6f s  d2h=%.6f s  checksum=%.0f\n",
         mode ? "tiled" : "naive", BS, t_kernel, t_h2d, t_d2h, sum);

  if (verify) {   // slow on CPU (a few seconds); run it once per build, not every time
    int bad = 0;
    for (int i = 0; i < N && !bad; i++)
      for (int j = 0; j < N; j++) {
        float s = 0.0f;
        for (int k = 0; k < N; k++) s += hA[i * N + k] * hB[k * N + j];
        if (s != hC[i * N + j]) { bad = 1; break; }
      }
    printf("verify: %s\n", bad ? "MISMATCH" : "OK");
  }

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  free(hA); free(hB); free(hC);
  return 0;
}

// Solution stub for "mse-loss".
// TODO: implement the body below. The signature is derived from
//       kernel-harnesses/mse-loss.cu and must stay in sync with it.
//
// Build it (the build system auto-picks this file):
//   make build/bin/mse-loss.exe
//   ./build/bin/mse-loss.exe
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

#define BLOCK_SIZE 512

__global__ void _kernel(const float* predictions, const float* targets, float* output, size_t n, float inv_n) {

  int base = (threadIdx.x + blockIdx.x * blockDim.x) * 8;

  // Scalar tail for the last partial group of 8 so the tail matches the
  // vectorized path exactly.
  if (base + 7 >= n) {
    for (int i = base; i < n; ++i) {
      float x = predictions[i];
      float y = targets[i];
      float a = fabsf(x - y);

      float r = (a < 1) ? (0.5f * a * a) : (a - 0.5f);
      r *= inv_n;
      atomicAdd(output, r);
    }
  } else {

    float4 x0 = *reinterpret_cast<const float4*>(predictions + base);
    float4 x1 = *reinterpret_cast<const float4*>(predictions + base + 4);

    float4 y0 = *reinterpret_cast<const float4*>(targets + base);
    float4 y1 = *reinterpret_cast<const float4*>(targets + base + 4);

    float4 a0 = {
      (x0.x - y0.x),
      (x0.y - y0.y),
      (x0.z - y0.z),
      (x0.w - y0.w),
    };

    float4 a1 = {
      (x1.x - y1.x),
      (x1.y - y1.y),
      (x1.z - y1.z),
      (x1.w - y1.w),
    };

    float acc0 = 
      a0.x * a0.x +
      a0.y * a0.y +
      a0.z * a0.z +
      a0.w * a0.w;
    float acc1 = 
      a1.x * a1.x +
      a1.y * a1.y +
      a1.z * a1.z +
      a1.w * a1.w;
    float r = acc0 + acc1;

#if 1
    const int tid = threadIdx.x;
    __shared__ float merge_buf[BLOCK_SIZE];

#pragma unroll
    for (int i = 1; i < BLOCK_SIZE; i *= 2) {
      merge_buf[tid] = r;
      __syncthreads();
      r += merge_buf[(tid + i) % BLOCK_SIZE];
      __syncthreads();
    }

    if (tid != 0)
      return;
#endif

    r *= inv_n;
    atomicAdd(output, r);
  }
}


// Note: all pointer arguments are device pointers.
extern "C" void solution(const float* predictions, const float* targets, float* output, const size_t* shape, size_t ndim) {
    // TODO: implement mse-loss
  int n = ndim;
  int threads_needed = (n + 7) / 8;
  int grid = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
  float inv_n = 1.0f / static_cast<float>(n);

  _kernel<<<grid, BLOCK_SIZE>>>(predictions, targets, output, n, inv_n);
}

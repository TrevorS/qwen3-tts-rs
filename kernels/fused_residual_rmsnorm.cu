// Fused residual-add + RMSNorm CUDA kernel.
//
// Combines two operations into one kernel launch:
//   sum[i] = x[i] + residual[i]
//   out[i] = sum[i] * weight[col] / sqrt(mean(sum^2) + eps)
//
// Output layout: [normed (rows*cols) | sum (rows*cols)]
// This allows returning two tensors from a single CustomOp2.
//
// Adapted from candle-kernels' rmsnorm (ggml heritage).

#include <stdint.h>

#if __CUDA_ARCH__ >= 530
#include <cuda_fp16.h>
#endif
#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#define WARP_SIZE 32

template <typename T>
__device__ float to_float(T v) { return static_cast<float>(v); }

template <typename T>
__device__ T from_float(float v) { return static_cast<T>(v); }

__device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// One block per row. Each thread handles a strided subset of columns.
// Output: first rows*cols elements = normed, next rows*cols = sum
template <typename T>
__device__ void fused_residual_rmsnorm(
    const T * __restrict__ x,
    const T * __restrict__ residual,
    const T * __restrict__ weight,
    T * __restrict__ dst,
    const int ncols,
    const int block_size,
    const float eps
) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;
    const int nrows = gridDim.x;
    const int row_offset = row * ncols;

    // Pointers into the two output regions
    T * __restrict__ dst_normed = dst;
    T * __restrict__ dst_sum = dst + nrows * ncols;

    // Pass 1: compute sum = x + residual, accumulate sum^2
    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        float xi = to_float(x[row_offset + col]);
        float ri = to_float(residual[row_offset + col]);
        float si = xi + ri;
        // Store sum immediately (will be read again in pass 2)
        dst_sum[row_offset + col] = from_float<T>(si);
        tmp += si * si;
    }

    // Warp reduction
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = (lane_id < (block_size / WARP_SIZE)) ? s_sum[lane_id] : 0.0f;
        tmp = warp_reduce_sum(tmp);
    }

    const float scale = rsqrtf(tmp / ncols + eps);

    // Pass 2: normalize and scale
    for (int col = tid; col < ncols; col += block_size) {
        float si = to_float(dst_sum[row_offset + col]);
        float w = to_float(weight[col]);
        dst_normed[row_offset + col] = from_float<T>(si * scale * w);
    }
}

#define FUSED_RESIDUAL_RMSNORM_OP(TYPENAME, FN_NAME) \
  extern "C" __global__ void FN_NAME(                \
      const TYPENAME *x,                             \
      const TYPENAME *residual,                      \
      const TYPENAME *weight,                        \
      TYPENAME *dst,                                 \
      const int n_cols,                              \
      const int block_size,                          \
      const float eps) {                             \
    fused_residual_rmsnorm<TYPENAME>(                \
        x, residual, weight, dst, n_cols, block_size, eps); \
  }

#if __CUDA_ARCH__ >= 800
FUSED_RESIDUAL_RMSNORM_OP(__nv_bfloat16, fused_residual_rmsnorm_bf16)
#endif
#if __CUDA_ARCH__ >= 530
FUSED_RESIDUAL_RMSNORM_OP(__half, fused_residual_rmsnorm_f16)
#endif
FUSED_RESIDUAL_RMSNORM_OP(float, fused_residual_rmsnorm_f32)
FUSED_RESIDUAL_RMSNORM_OP(double, fused_residual_rmsnorm_f64)

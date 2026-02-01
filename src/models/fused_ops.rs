//! Fused CUDA kernels with CPU fallbacks.
//!
//! On CUDA (with `cuda` feature), launches a custom PTX kernel for fused residual + RMSNorm.
//! On CPU/Metal, falls back to sequential operations with identical results.

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{rms_norm, RmsNorm, VarBuilder};

/// RMSNorm that supports fused residual addition on CUDA.
///
/// Drop-in replacement for `candle_nn::RmsNorm` in decoder layers.
/// Call [`FusedRmsNorm::forward_residual`] to fuse `rms_norm(x + residual)` into
/// a single kernel launch (CUDA only). Falls back to two separate ops on CPU.
pub struct FusedRmsNorm {
    inner: RmsNorm,
    #[cfg(feature = "cuda")]
    weight: Tensor,
    #[cfg(feature = "cuda")]
    eps: f32,
}

impl FusedRmsNorm {
    /// Load from a VarBuilder path (same weight names as candle_nn::rms_norm).
    pub fn load(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let inner = rms_norm(hidden_size, eps, vb.clone())?;
        #[cfg(feature = "cuda")]
        let weight = vb.get(hidden_size, "weight")?;
        Ok(Self {
            inner,
            #[cfg(feature = "cuda")]
            weight,
            #[cfg(feature = "cuda")]
            eps: eps as f32,
        })
    }

    /// Standard forward (no residual fusion). Delegates to inner RmsNorm.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.inner.forward(x)?)
    }

    /// Fused residual-add + RMSNorm.
    ///
    /// Returns `(rms_norm(x + residual), x + residual)`.
    ///
    /// On CUDA: single kernel launch via custom PTX.
    /// On CPU/Metal: sequential `add` then `rms_norm`.
    pub fn forward_residual(&self, x: &Tensor, residual: &Tensor) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        {
            if x.device().is_cuda() {
                return self.forward_residual_fused(x, residual);
            }
        }
        self.forward_residual_sequential(x, residual)
    }

    fn forward_residual_sequential(
        &self,
        x: &Tensor,
        residual: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let sum = (x + residual)?;
        let normed = self.inner.forward(&sum)?;
        Ok((normed, sum))
    }

    #[cfg(feature = "cuda")]
    fn forward_residual_fused(&self, x: &Tensor, residual: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = x.contiguous()?;
        let residual = residual.contiguous()?;

        let shape = x.shape().clone();
        let dims = shape.dims();
        let hidden = *dims.last().unwrap();
        let el = shape.elem_count();
        let n_rows = el / hidden;

        let op = FusedResidualRmsNormOp {
            weight: self.weight.clone(),
            eps: self.eps,
            n_rows,
            n_cols: hidden,
        };

        // CustomOp2 returns a single tensor of shape [2*rows, cols] containing
        // [normed | sum] concatenated along the row dimension.
        let combined = x.apply_op2_no_bwd(&residual, &op)?;

        // Split back into two tensors with the original shape.
        let normed = combined.narrow(0, 0, n_rows)?.reshape(dims)?;
        let sum = combined.narrow(0, n_rows, n_rows)?.reshape(dims)?;
        Ok((normed, sum))
    }
}

#[cfg(feature = "cuda")]
static FUSED_RESIDUAL_RMSNORM_PTX: &str = include_str!("../../kernels/fused_residual_rmsnorm.ptx");

#[cfg(feature = "cuda")]
struct FusedResidualRmsNormOp {
    weight: Tensor,
    eps: f32,
    n_rows: usize,
    n_cols: usize,
}

#[cfg(feature = "cuda")]
impl FusedResidualRmsNormOp {
    fn launch_kernel<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::WithDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CudaStorage,
        l2: &candle_core::Layout,
    ) -> candle_core::Result<candle_core::cuda_backend::CudaStorageSlice> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;
        use candle_core::DType;

        let dev = s1.device();

        let x = s1.as_cuda_slice::<T>()?;
        let x = match l1.contiguous_offsets() {
            None => candle_core::bail!("x must be contiguous"),
            Some((o1, o2)) => x.slice(o1..o2),
        };
        let r = s2.as_cuda_slice::<T>()?;
        let r = match l2.contiguous_offsets() {
            None => candle_core::bail!("residual must be contiguous"),
            Some((o1, o2)) => r.slice(o1..o2),
        };

        let kernel_name = match T::DTYPE {
            DType::BF16 => "fused_residual_rmsnorm_bf16",
            DType::F16 => "fused_residual_rmsnorm_f16",
            DType::F32 => "fused_residual_rmsnorm_f32",
            DType::F64 => "fused_residual_rmsnorm_f64",
            dt => candle_core::bail!("fused-residual-rmsnorm unsupported dtype {dt:?}"),
        };

        let func = dev.get_or_load_custom_func(
            kernel_name,
            "fused_residual_rmsnorm",
            FUSED_RESIDUAL_RMSNORM_PTX,
        )?;

        let weight_guard = self.weight.storage_and_layout();
        let weight_cuda = match &*weight_guard.0 {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("weight must be on CUDA"),
        };
        let w = weight_cuda.as_cuda_slice::<T>()?;

        let block_size: u32 = if self.n_cols < 1024 { 32 } else { 1024 };
        let cfg = LaunchConfig {
            grid_dim: (self.n_rows as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let out_el = 2 * self.n_rows * self.n_cols;
        // SAFETY: Filled by kernel before read.
        let dst = unsafe { dev.alloc::<T>(out_el)? };

        let mut builder = func.builder();
        builder.arg(&x);
        builder.arg(&r);
        builder.arg(w);
        builder.arg(&dst);
        candle_core::builder_arg!(builder, self.n_cols as i32, block_size as i32, self.eps);
        // SAFETY: kernel launch
        unsafe { builder.launch(cfg) }.w()?;

        Ok(T::wrap_cuda_slice(dst, dev.clone()).slice)
    }
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp2 for FusedResidualRmsNormOp {
    fn name(&self) -> &'static str {
        "fused-residual-rmsnorm"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle_core::CpuStorage,
        _l1: &candle_core::Layout,
        _s2: &candle_core::CpuStorage,
        _l2: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("fused-residual-rmsnorm is CUDA-only; use forward_residual_sequential")
    }

    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CudaStorage,
        l2: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::Shape;

        let dev = s1.device();
        let out_shape = Shape::from_dims(&[2 * self.n_rows, self.n_cols]);

        let slice = match s1.dtype() {
            candle_core::DType::BF16 => self.launch_kernel::<half::bf16>(s1, l1, s2, l2)?,
            candle_core::DType::F16 => self.launch_kernel::<half::f16>(s1, l1, s2, l2)?,
            candle_core::DType::F32 => self.launch_kernel::<f32>(s1, l1, s2, l2)?,
            candle_core::DType::F64 => self.launch_kernel::<f64>(s1, l1, s2, l2)?,
            dt => candle_core::bail!("fused-residual-rmsnorm unsupported dtype {dt:?}"),
        };

        let dst = candle_core::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, out_shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn create_fused_rms_norm(hidden_size: usize, eps: f64, device: &Device) -> FusedRmsNorm {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        FusedRmsNorm::load(hidden_size, eps, vb).unwrap()
    }

    #[test]
    fn test_fused_rmsnorm_forward_matches_standard() {
        let device = Device::Cpu;
        let hidden = 64;
        let eps = 1e-6;

        let norm = create_fused_rms_norm(hidden, eps, &device);

        let x = Tensor::randn(0.0f32, 1.0, (2, 10, hidden), &device).unwrap();
        let out_standard = norm.forward(&x).unwrap();
        let out_inner = norm.inner.forward(&x).unwrap();

        let diff = (&out_standard - &out_inner)
            .unwrap()
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-5, "forward mismatch: max diff = {diff}");
    }

    #[test]
    fn test_fused_residual_rmsnorm_sequential() {
        let device = Device::Cpu;
        let hidden = 64;
        let eps = 1e-6;

        let norm = create_fused_rms_norm(hidden, eps, &device);

        let x = Tensor::randn(0.0f32, 1.0, (2, 10, hidden), &device).unwrap();
        let residual = Tensor::randn(0.0f32, 1.0, (2, 10, hidden), &device).unwrap();

        let (normed, sum) = norm.forward_residual(&x, &residual).unwrap();

        // Verify sum = x + residual
        let expected_sum = (&x + &residual).unwrap();
        let sum_diff = (&sum - &expected_sum)
            .unwrap()
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(sum_diff < 1e-6, "sum mismatch: {sum_diff}");

        // Verify normed = rms_norm(sum)
        let expected_normed = norm.inner.forward(&expected_sum).unwrap();
        let norm_diff = (&normed - &expected_normed)
            .unwrap()
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(norm_diff < 1e-5, "norm mismatch: {norm_diff}");
    }

    #[test]
    fn test_fused_rmsnorm_shapes() {
        let device = Device::Cpu;
        let hidden = 64;
        let norm = create_fused_rms_norm(hidden, 1e-6, &device);

        let x = Tensor::randn(0.0f32, 1.0, (1, 5, hidden), &device).unwrap();
        let residual = Tensor::randn(0.0f32, 1.0, (1, 5, hidden), &device).unwrap();

        let (normed, sum) = norm.forward_residual(&x, &residual).unwrap();
        assert_eq!(normed.dims(), &[1, 5, hidden]);
        assert_eq!(sum.dims(), &[1, 5, hidden]);
    }
}

//! KV cache implementations for autoregressive generation.
//!
//! Provides two variants:
//! - [`KVCache`]: Concatenation-based (works on all backends).
//! - [`PreAllocKVCache`]: Pre-allocated fixed-size buffer with in-place writes.
//!   Uses `InplaceOp2` + `copy2d` on CUDA, `slice_set` on Metal/CPU.
//!   Zero allocation during generation on CUDA and Metal.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

#[cfg(feature = "cuda")]
use candle_core::{backend::BackendStorage, CudaStorage, InplaceOp2, Layout};

// ─── Concat-based KVCache (original) ────────────────────────────────────────

/// KV cache for efficient autoregressive generation (concat-based).
pub struct KVCache {
    pub(crate) k: Option<Tensor>,
    pub(crate) v: Option<Tensor>,
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

impl KVCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn update_k(&mut self, k: &Tensor) -> Result<Tensor> {
        let k = if let Some(prev_k) = &self.k {
            Tensor::cat(&[prev_k, k], 2)?
        } else {
            k.clone()
        };
        self.k = Some(k.clone());
        Ok(k)
    }

    pub fn update_v(&mut self, v: &Tensor) -> Result<Tensor> {
        let v = if let Some(prev_v) = &self.v {
            Tensor::cat(&[prev_v, v], 2)?
        } else {
            v.clone()
        };
        self.v = Some(v.clone());
        Ok(v)
    }

    /// Get K cache sum for debugging. Returns 0.0 if cache is empty.
    pub fn k_sum(&self) -> Result<f32> {
        match &self.k {
            Some(k) => {
                let vals: Vec<f32> = k
                    .to_dtype(candle_core::DType::F32)?
                    .flatten_all()?
                    .to_vec1()?;
                Ok(vals.iter().sum())
            }
            None => Ok(0.0),
        }
    }

    /// Get V cache sum for debugging. Returns 0.0 if cache is empty.
    pub fn v_sum(&self) -> Result<f32> {
        match &self.v {
            Some(v) => {
                let vals: Vec<f32> = v
                    .to_dtype(candle_core::DType::F32)?
                    .flatten_all()?
                    .to_vec1()?;
                Ok(vals.iter().sum())
            }
            None => Ok(0.0),
        }
    }

    /// Get K cache shape for debugging.
    pub fn k_shape(&self) -> Option<Vec<usize>> {
        self.k.as_ref().map(|k| k.dims().to_vec())
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}

// ─── Unified cache enum ─────────────────────────────────────────────────────

/// Unified KV cache type: either concat-based or pre-allocated.
pub enum AnyKVCache {
    Concat(KVCache),
    PreAlloc(PreAllocKVCache),
}

impl AnyKVCache {
    /// Update cache with new K/V values and return the full K/V sequences.
    pub fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            AnyKVCache::Concat(cache) => {
                let k = cache.update_k(k)?;
                let v = cache.update_v(v)?;
                Ok((k, v))
            }
            AnyKVCache::PreAlloc(cache) => cache.update(k, v),
        }
    }

    pub fn reset(&mut self) {
        match self {
            AnyKVCache::Concat(cache) => cache.reset(),
            AnyKVCache::PreAlloc(cache) => cache.reset(),
        }
    }
}

// ─── Pre-allocated KVCache ──────────────────────────────────────────────────

/// In-place copy operation for CUDA: copies source data into a pre-allocated
/// buffer at a specific position using `copy2d` (strided device-to-device copy).
///
/// The buffer has shape `[batch, num_heads, max_seq, head_dim]`.
/// The source has shape `[batch, num_heads, new_seq, head_dim]`.
/// We write into `[:, :, pos:pos+new_seq, :]`.
#[cfg(feature = "cuda")]
struct KVCacheAppend {
    /// Destination position (sequence index to start writing at)
    dst_pos: usize,
    /// max_seq dimension of the pre-allocated buffer
    max_seq: usize,
    /// head_dim (D) — elements per head per position
    head_dim: usize,
    /// num_heads * batch (number of "rows" to copy in the 2D sense)
    num_head_rows: usize,
    /// Number of new sequence positions being written
    new_seq: usize,
}

#[cfg(feature = "cuda")]
impl InplaceOp2 for KVCacheAppend {
    fn name(&self) -> &'static str {
        "kv_cache_append"
    }

    fn cpu_fwd(
        &self,
        _s1: &mut candle_core::CpuStorage,
        _l1: &Layout,
        _s2: &candle_core::CpuStorage,
        _l2: &Layout,
    ) -> candle_core::Result<()> {
        candle_core::bail!("KVCacheAppend is CUDA-only; use KVCache on CPU")
    }

    fn cuda_fwd(
        &self,
        dst_storage: &mut CudaStorage,
        _dst_layout: &Layout,
        src_storage: &CudaStorage,
        src_layout: &Layout,
    ) -> candle_core::Result<()> {
        // Use copy2d for strided device-to-device copy.
        //
        // Buffer layout (contiguous): [B, H, max_seq, D]
        //   Element (b, h, s, d) is at flat index: ((b*H + h) * max_seq + s) * D + d
        //
        // Source layout (contiguous):  [B, H, new_seq, D]
        //   Element (b, h, s, d) is at flat index: ((b*H + h) * new_seq + s) * D + d
        //
        // We want to copy new_seq positions per head:
        //   For each (b, h): copy new_seq * D elements
        //     from src offset: (b*H + h) * new_seq * D
        //     to   dst offset: ((b*H + h) * max_seq + dst_pos) * D
        //
        // copy2d params:
        //   d1 = num_head_rows * new_seq  (total rows of D elements)
        //   d2 = D                        (elements per row)
        //   src_s = D                     (source stride: rows are contiguous)
        //   dst_s = D                     (within a head, positions are contiguous in dst too)
        //   BUT: between heads there's a gap in dst.
        //
        // Actually, copy2d only supports uniform stride. We need to handle the
        // head gaps. For new_seq=1 (the common decode case), each head is one
        // row of D elements, with source stride D and dest stride max_seq*D.
        // For prefill (new_seq>1), we need a different strategy.

        if self.new_seq == 1 {
            // Decode case: one position per head, strided write.
            // d1 = num_head_rows, d2 = D, src_s = D, dst_s = max_seq * D
            let src_offset = src_layout.start_offset();
            let dst_offset = self.dst_pos * self.head_dim;
            src_storage.copy2d(
                dst_storage,
                self.num_head_rows,           // d1: number of heads (rows)
                self.head_dim,                // d2: elements per row
                self.head_dim,                // src_s: contiguous heads in source
                self.max_seq * self.head_dim, // dst_s: heads spaced by max_seq*D in dest
                src_offset,
                dst_offset,
            )?;
        } else {
            // Prefill case: multiple positions per head.
            // Each head has new_seq * D contiguous elements in source,
            // and max_seq * D slots in destination.
            // copy2d with: d1 = num_head_rows, d2 = new_seq * D,
            //   src_s = new_seq * D, dst_s = max_seq * D
            let src_offset = src_layout.start_offset();
            let dst_offset = self.dst_pos * self.head_dim;
            src_storage.copy2d(
                dst_storage,
                self.num_head_rows,
                self.new_seq * self.head_dim,
                self.new_seq * self.head_dim,
                self.max_seq * self.head_dim,
                src_offset,
                dst_offset,
            )?;
        }
        Ok(())
    }
}

/// Pre-allocated KV cache with in-place writes.
///
/// Allocates fixed-size K and V buffers at construction time. During generation,
/// new K/V data is written in-place via `InplaceOp2` + `copy2d` (CUDA) or
/// `Tensor::slice_set` (Metal/CPU), avoiding allocation and full-buffer copies.
#[allow(dead_code)] // num_heads and head_dim are used in the CUDA InplaceOp2 path
pub struct PreAllocKVCache {
    /// Pre-allocated K buffer: `[batch, num_heads, max_seq, head_dim]`
    k_buf: Tensor,
    /// Pre-allocated V buffer: `[batch, num_heads, max_seq, head_dim]`
    v_buf: Tensor,
    /// Current number of filled sequence positions
    current_len: usize,
    /// Maximum sequence length
    max_seq: usize,
    /// Number of KV heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Whether to use CUDA InplaceOp2 path (false = use slice_set fallback)
    use_inplace: bool,
}

impl PreAllocKVCache {
    /// Create a new pre-allocated KV cache.
    ///
    /// # Arguments
    /// * `batch` - Batch size (typically 1)
    /// * `num_heads` - Number of KV heads
    /// * `max_seq` - Maximum sequence length to pre-allocate
    /// * `head_dim` - Dimension per head
    /// * `dtype` - Data type (e.g. BF16)
    /// * `device` - Target device
    pub fn new(
        batch: usize,
        num_heads: usize,
        max_seq: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let shape = (batch, num_heads, max_seq, head_dim);
        let k_buf = Tensor::zeros(shape, dtype, device)?;
        let v_buf = Tensor::zeros(shape, dtype, device)?;

        let use_inplace = device.is_cuda();

        Ok(Self {
            k_buf,
            v_buf,
            current_len: 0,
            max_seq,
            num_heads,
            head_dim,
            use_inplace,
        })
    }

    /// Append new K and V values, advance the position, and return views
    /// of the full K and V sequences so far.
    ///
    /// `k` and `v` have shape `[batch, num_heads, new_seq, head_dim]`.
    pub fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let new_seq = k.dim(2)?;
        let new_len = self.current_len + new_seq;
        if new_len > self.max_seq {
            anyhow::bail!(
                "KV cache overflow: current={} + new={} > max={}",
                self.current_len,
                new_seq,
                self.max_seq
            );
        }

        self.append_to_buf(&self.k_buf.clone(), k, new_seq)?;
        self.append_to_buf(&self.v_buf.clone(), v, new_seq)?;
        self.current_len = new_len;

        // Return narrow views of filled portion (zero-copy on CUDA)
        let k_view = self.k_buf.narrow(2, 0, new_len)?;
        let v_view = self.v_buf.narrow(2, 0, new_len)?;
        Ok((k_view, v_view))
    }

    fn append_to_buf(&self, buf: &Tensor, src: &Tensor, new_seq: usize) -> Result<()> {
        let pos = self.current_len;
        if pos + new_seq > self.max_seq {
            anyhow::bail!(
                "KV cache overflow: pos={} + new_seq={} > max_seq={}",
                pos,
                new_seq,
                self.max_seq
            );
        }

        if self.use_inplace {
            #[cfg(feature = "cuda")]
            {
                let src_contiguous = src.contiguous()?;
                let op = KVCacheAppend {
                    dst_pos: pos,
                    max_seq: self.max_seq,
                    head_dim: self.head_dim,
                    num_head_rows: src.dim(0)? * self.num_heads,
                    new_seq,
                };
                buf.inplace_op2(&src_contiguous, &op)?;
                return Ok(());
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("In-place KV cache requires CUDA feature");
            }
        }

        // Metal/CPU fallback: slice_set writes in-place via copy2d (blit on Metal).
        let src_contiguous = src.contiguous()?;
        buf.slice_set(&src_contiguous, 2, pos)?;
        Ok(())
    }

    /// Reset the cache for reuse (e.g. between code predictor frames).
    pub fn reset(&mut self) {
        self.current_len = 0;
    }

    /// Current filled length.
    pub fn len(&self) -> usize {
        self.current_len
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_kv_cache_new() {
        let cache = KVCache::new();
        assert!(cache.k.is_none());
        assert!(cache.v.is_none());
    }

    #[test]
    fn test_kv_cache_update() {
        let device = Device::Cpu;
        let mut cache = KVCache::new();

        let k1 = Tensor::randn(0.0f32, 1.0, (1, 2, 4, 16), &device).unwrap();
        let k_out = cache.update_k(&k1).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 4, 16]);

        let k2 = Tensor::randn(0.0f32, 1.0, (1, 2, 3, 16), &device).unwrap();
        let k_out = cache.update_k(&k2).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 7, 16]); // 4 + 3 = 7
    }

    #[test]
    fn test_kv_cache_reset() {
        let device = Device::Cpu;
        let mut cache = KVCache::new();

        let k = Tensor::randn(0.0f32, 1.0, (1, 2, 4, 16), &device).unwrap();
        cache.update_k(&k).unwrap();
        assert!(cache.k.is_some());

        cache.reset();
        assert!(cache.k.is_none());
        assert!(cache.v.is_none());
    }

    #[test]
    fn test_prealloc_kv_cache_creation() {
        let device = Device::Cpu;
        let cache = PreAllocKVCache::new(1, 2, 16, 8, DType::F32, &device).unwrap();
        assert_eq!(cache.current_len, 0);
        assert_eq!(cache.max_seq, 16);
        assert!(!cache.use_inplace); // CPU doesn't use inplace
    }

    #[test]
    fn test_prealloc_kv_cache_reset() {
        let device = Device::Cpu;
        let mut cache = PreAllocKVCache::new(1, 2, 16, 8, DType::F32, &device).unwrap();
        cache.current_len = 5;
        cache.reset();
        assert_eq!(cache.current_len, 0);
    }
}

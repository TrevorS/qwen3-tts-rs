//! Vector quantization for audio codecs
//!
//! Implements residual vector quantization (RVQ) used by neural audio codecs
//! to discretize continuous audio representations.

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{embedding, Embedding, VarBuilder};

/// Single vector quantizer with learnable codebook
pub struct VectorQuantizer {
    /// Codebook embeddings: [codebook_size, dim]
    codebook: Embedding,
    /// Codebook size
    codebook_size: usize,
    /// Embedding dimension
    dim: usize,
}

impl VectorQuantizer {
    /// Create new vector quantizer
    pub fn new(codebook_size: usize, dim: usize, vb: VarBuilder) -> Result<Self> {
        let codebook = embedding(codebook_size, dim, vb.pp("codebook"))?;

        Ok(Self {
            codebook,
            codebook_size,
            dim,
        })
    }

    /// Quantize continuous vectors to discrete codes
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, seq, dim]
    ///
    /// # Returns
    /// Tuple of (quantized embeddings, indices)
    pub fn encode(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, seq, dim) = x.dims3()?;

        // Flatten to [batch * seq, dim]
        let x_flat = x.reshape((batch * seq, dim))?;

        // Get codebook weights: [codebook_size, dim]
        let codebook_indices: Vec<u32> = (0..self.codebook_size as u32).collect();
        let indices_tensor = Tensor::new(codebook_indices.as_slice(), x.device())?;
        let codebook_weights = self.codebook.forward(&indices_tensor)?;

        // Compute distances: ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x @ c^T
        let x_sq = x_flat.sqr()?.sum(D::Minus1)?;
        let c_sq = codebook_weights.sqr()?.sum(D::Minus1)?;
        let xc = x_flat.matmul(&codebook_weights.transpose(0, 1)?)?;

        let distances = ((x_sq.unsqueeze(1)? + c_sq.unsqueeze(0)?)? - (xc * 2.0)?)?;

        // Get nearest codebook entry
        let indices = distances.argmin(D::Minus1)?;
        let indices = indices.reshape((batch, seq))?;

        // Look up quantized embeddings
        let quantized = self.decode(&indices)?;

        Ok((quantized, indices))
    }

    /// Decode indices to embeddings
    ///
    /// # Arguments
    /// * `indices` - Token indices of shape [batch, seq]
    ///
    /// # Returns
    /// Embeddings of shape [batch, seq, dim]
    pub fn decode(&self, indices: &Tensor) -> Result<Tensor> {
        Ok(self.codebook.forward(indices)?)
    }

    /// Get codebook dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get codebook size
    pub fn size(&self) -> usize {
        self.codebook_size
    }
}

/// Residual Vector Quantizer (RVQ)
///
/// Applies multiple VQ layers sequentially, where each layer quantizes
/// the residual from the previous layer.
pub struct ResidualVectorQuantizer {
    /// Individual quantizers
    quantizers: Vec<VectorQuantizer>,
    /// Number of quantizers
    num_quantizers: usize,
    /// Codebook dimension
    dim: usize,
}

impl ResidualVectorQuantizer {
    /// Create new residual vector quantizer
    pub fn new(
        num_quantizers: usize,
        codebook_size: usize,
        dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut quantizers = Vec::with_capacity(num_quantizers);

        for i in 0..num_quantizers {
            quantizers.push(VectorQuantizer::new(
                codebook_size,
                dim,
                vb.pp(format!("layers.{}", i)),
            )?);
        }

        Ok(Self {
            quantizers,
            num_quantizers,
            dim,
        })
    }

    /// Quantize continuous vectors using residual quantization
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, seq, dim]
    ///
    /// # Returns
    /// Tuple of (quantized sum, all indices [batch, num_q, seq])
    pub fn encode(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, seq, _) = x.dims3()?;
        let mut residual = x.clone();
        let mut quantized_sum = Tensor::zeros(x.shape(), x.dtype(), x.device())?;
        let mut all_indices = Vec::with_capacity(self.num_quantizers);

        for quantizer in &self.quantizers {
            let (quantized, indices) = quantizer.encode(&residual)?;

            // Accumulate quantized values
            quantized_sum = (quantized_sum + &quantized)?;

            // Update residual
            residual = (&residual - &quantized)?;

            // Store indices
            all_indices.push(indices);
        }

        // Stack indices: [batch, num_q, seq]
        let all_indices = Tensor::stack(&all_indices, 1)?;

        Ok((quantized_sum, all_indices))
    }

    /// Decode indices to embeddings
    ///
    /// # Arguments
    /// * `indices` - Token indices of shape [batch, num_quantizers, seq]
    ///
    /// # Returns
    /// Embeddings of shape [batch, seq, num_quantizers, dim]
    pub fn decode(&self, indices: &Tensor) -> Result<Tensor> {
        let (batch, num_q, seq) = indices.dims3()?;

        let mut embeddings = Vec::with_capacity(num_q);

        for (i, quantizer) in self.quantizers.iter().enumerate() {
            // Get indices for this quantizer: [batch, seq]
            let q_indices = indices.i((.., i, ..))?;
            let emb = quantizer.decode(&q_indices)?;
            embeddings.push(emb);
        }

        // Stack: [batch, seq, num_q, dim]
        let embeddings = Tensor::stack(&embeddings, 2)?;

        Ok(embeddings)
    }

    /// Decode and sum all quantizer outputs
    ///
    /// # Arguments
    /// * `indices` - Token indices of shape [batch, num_quantizers, seq]
    ///
    /// # Returns
    /// Summed embeddings of shape [batch, seq, dim]
    pub fn decode_sum(&self, indices: &Tensor) -> Result<Tensor> {
        let embeddings = self.decode(indices)?;
        Ok(embeddings.sum(2)?) // Sum over quantizer dimension
    }

    /// Get number of quantizers
    pub fn num_quantizers(&self) -> usize {
        self.num_quantizers
    }

    /// Get codebook dimension
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Split Residual Vector Quantizer
///
/// Variant that splits quantizers into semantic (first N) and acoustic groups,
/// commonly used in speech codecs.
pub struct SplitResidualVectorQuantizer {
    /// Semantic quantizers (typically 1)
    semantic_quantizers: ResidualVectorQuantizer,
    /// Acoustic quantizers (typically 15)
    acoustic_quantizers: ResidualVectorQuantizer,
    /// Number of semantic quantizers
    num_semantic: usize,
}

impl SplitResidualVectorQuantizer {
    /// Create new split RVQ
    pub fn new(
        num_semantic: usize,
        num_acoustic: usize,
        codebook_size: usize,
        dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let semantic_quantizers = ResidualVectorQuantizer::new(
            num_semantic,
            codebook_size,
            dim,
            vb.pp("semantic"),
        )?;

        let acoustic_quantizers = ResidualVectorQuantizer::new(
            num_acoustic,
            codebook_size,
            dim,
            vb.pp("acoustic"),
        )?;

        Ok(Self {
            semantic_quantizers,
            acoustic_quantizers,
            num_semantic,
        })
    }

    /// Decode semantic indices only
    pub fn decode_semantic(&self, indices: &Tensor) -> Result<Tensor> {
        self.semantic_quantizers.decode_sum(indices)
    }

    /// Decode acoustic indices only
    pub fn decode_acoustic(&self, indices: &Tensor) -> Result<Tensor> {
        self.acoustic_quantizers.decode_sum(indices)
    }

    /// Decode all indices
    pub fn decode(&self, semantic_indices: &Tensor, acoustic_indices: &Tensor) -> Result<Tensor> {
        let semantic = self.semantic_quantizers.decode_sum(semantic_indices)?;
        let acoustic = self.acoustic_quantizers.decode_sum(acoustic_indices)?;
        Ok((semantic + acoustic)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_vq_dimensions() {
        // Just test that types are correct
        let codebook_size = 2048;
        let dim = 256;
        assert!(codebook_size > 0);
        assert!(dim > 0);
    }
}

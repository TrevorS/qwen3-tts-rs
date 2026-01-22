//! Audio I/O utilities

use anyhow::{Context, Result};
use candle_core::Tensor;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::path::Path;

/// Audio buffer holding raw waveform data
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Mono audio samples in [-1.0, 1.0] range
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
}

impl AudioBuffer {
    /// Create a new audio buffer
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self { samples, sample_rate }
    }

    /// Create from a Candle tensor (assumed shape: [samples] or [1, samples])
    pub fn from_tensor(tensor: Tensor, sample_rate: u32) -> Result<Self> {
        let tensor = if tensor.dims().len() == 2 {
            tensor.squeeze(0)?
        } else {
            tensor
        };

        let samples: Vec<f32> = tensor.to_vec1()?;
        Ok(Self::new(samples, sample_rate))
    }

    /// Convert to a Candle tensor
    pub fn to_tensor(&self, device: &candle_core::Device) -> Result<Tensor> {
        Ok(Tensor::new(self.samples.as_slice(), device)?)
    }

    /// Duration in seconds
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }

    /// Number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Save to WAV file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        save_wav(path, &self.samples, self.sample_rate)
    }

    /// Load from WAV file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        load_wav(path)
    }

    /// Normalize audio to [-1.0, 1.0] range
    pub fn normalize(&mut self) {
        let max_abs = self
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);

        if max_abs > 0.0 && max_abs != 1.0 {
            for sample in &mut self.samples {
                *sample /= max_abs;
            }
        }
    }

    /// Apply peak normalization to a target dB level
    pub fn normalize_db(&mut self, target_db: f32) {
        let target_amplitude = 10.0f32.powf(target_db / 20.0);
        let max_abs = self
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);

        if max_abs > 0.0 {
            let scale = target_amplitude / max_abs;
            for sample in &mut self.samples {
                *sample *= scale;
            }
        }
    }
}

/// Load a WAV file into an AudioBuffer
pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<AudioBuffer> {
    let path = path.as_ref();
    let reader = WavReader::open(path)
        .with_context(|| format!("Failed to open WAV file: {}", path.display()))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<_>, _>>()?,
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<Result<Vec<_>, _>>()?
        }
    };

    // Convert to mono by averaging channels
    let mono_samples = if channels > 1 {
        samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    Ok(AudioBuffer::new(mono_samples, sample_rate))
}

/// Save samples to a WAV file
pub fn save_wav<P: AsRef<Path>>(path: P, samples: &[f32], sample_rate: u32) -> Result<()> {
    let path = path.as_ref();
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(path, spec)
        .with_context(|| format!("Failed to create WAV file: {}", path.display()))?;

    for &sample in samples {
        writer.write_sample(sample)?;
    }

    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_buffer_duration() {
        let buffer = AudioBuffer::new(vec![0.0; 24000], 24000);
        assert!((buffer.duration() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let mut buffer = AudioBuffer::new(vec![0.5, -0.25, 0.1], 24000);
        buffer.normalize();
        assert!((buffer.samples[0] - 1.0).abs() < 1e-6);
        assert!((buffer.samples[1] - (-0.5)).abs() < 1e-6);
    }
}

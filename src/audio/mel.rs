//! Mel-spectrogram computation
//!
//! Implementation based on librosa's mel spectrogram computation,
//! optimized for the Qwen3-TTS speaker encoder requirements.

use anyhow::Result;
use candle_core::{Device, Tensor};
use num_complex::Complex;
use rustfft::{num_complex::Complex as FftComplex, FftPlanner};
use std::f32::consts::PI;

/// Configuration for mel spectrogram computation
#[derive(Debug, Clone)]
pub struct MelConfig {
    /// Sample rate of input audio
    pub sample_rate: u32,
    /// FFT window size
    pub n_fft: usize,
    /// Hop length between frames
    pub hop_length: usize,
    /// Window length (defaults to n_fft)
    pub win_length: Option<usize>,
    /// Number of mel bands
    pub n_mels: usize,
    /// Minimum frequency for mel filterbank
    pub fmin: f32,
    /// Maximum frequency for mel filterbank (defaults to sample_rate / 2)
    pub fmax: Option<f32>,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            n_fft: 400,
            hop_length: 160,
            win_length: None,
            n_mels: 128,
            fmin: 0.0,
            fmax: None,
        }
    }
}

/// Mel-spectrogram extractor
pub struct MelSpectrogram {
    config: MelConfig,
    /// Precomputed mel filterbank
    mel_basis: Vec<Vec<f32>>,
    /// Precomputed Hann window
    window: Vec<f32>,
}

impl MelSpectrogram {
    /// Configuration tuned for the ECAPA-TDNN speaker encoder.
    ///
    /// Uses n_fft=1024, hop=256, win=1024 to match the Python reference
    /// `Qwen3TTSSpeakerEncoderConfig`.
    pub fn speaker_encoder() -> MelConfig {
        MelConfig {
            sample_rate: 24000,
            n_fft: 1024,
            hop_length: 256,
            win_length: Some(1024),
            n_mels: 128,
            fmin: 0.0,
            fmax: None,
        }
    }

    /// Create a new mel spectrogram extractor
    pub fn new(config: MelConfig) -> Self {
        let win_length = config.win_length.unwrap_or(config.n_fft);
        let fmax = config.fmax.unwrap_or(config.sample_rate as f32 / 2.0);

        // Compute mel filterbank
        let mel_basis = Self::create_mel_filterbank(
            config.sample_rate,
            config.n_fft,
            config.n_mels,
            config.fmin,
            fmax,
        );

        // Compute Hann window
        let window = Self::hann_window(win_length);

        Self {
            config,
            mel_basis,
            window,
        }
    }

    /// Compute mel spectrogram from audio samples
    pub fn compute(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        // Compute STFT
        let stft = self.stft(samples);

        // Compute power spectrogram
        let power_spec: Vec<Vec<f32>> = stft
            .iter()
            .map(|frame| frame.iter().map(|c| c.norm_sqr()).collect())
            .collect();

        // Apply mel filterbank
        self.apply_mel_filterbank(&power_spec)
    }

    /// Compute mel spectrogram and return as tensor
    pub fn compute_tensor(&self, samples: &[f32], device: &Device) -> Result<Tensor> {
        let mel = self.compute(samples);
        let n_frames = mel.len();
        let n_mels = self.config.n_mels;

        // Flatten and create tensor
        let flat: Vec<f32> = mel.into_iter().flatten().collect();
        let tensor = Tensor::new(flat.as_slice(), device)?
            .reshape((n_frames, n_mels))?
            .transpose(0, 1)?; // [n_mels, n_frames]

        Ok(tensor)
    }

    /// Compute log mel spectrogram
    pub fn compute_log(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let mel = self.compute(samples);
        mel.into_iter()
            .map(|frame| frame.into_iter().map(|v| (v.max(1e-10)).ln()).collect())
            .collect()
    }

    /// Compute mel spectrogram for the speaker encoder.
    ///
    /// Differs from [`Self::compute`] in two ways:
    /// - Uses **magnitude** spectrum (`sqrt(re² + im² + 1e-9)`) rather than power spectrum
    /// - Applies `log(clamp(mel, 1e-5))` compression
    ///
    /// Returns a tensor of shape `[n_mels, n_frames]`.
    pub fn compute_for_speaker_encoder(
        &self,
        samples: &[f32],
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        let stft = self.stft(samples);

        // Magnitude spectrum (not power)
        let mag_spec: Vec<Vec<f32>> = stft
            .iter()
            .map(|frame| {
                frame
                    .iter()
                    .map(|c| (c.re * c.re + c.im * c.im + 1e-9).sqrt())
                    .collect()
            })
            .collect();

        // Apply mel filterbank
        let mel = self.apply_mel_filterbank(&mag_spec);

        // Log compression with floor
        let log_mel: Vec<Vec<f32>> = mel
            .into_iter()
            .map(|frame| frame.into_iter().map(|v| v.max(1e-5).ln()).collect())
            .collect();

        let n_frames = log_mel.len();
        let n_mels = self.config.n_mels;

        let flat: Vec<f32> = log_mel.into_iter().flatten().collect();
        let tensor = Tensor::new(flat.as_slice(), device)?
            .reshape((n_frames, n_mels))?
            .transpose(0, 1)?; // [n_mels, n_frames]

        Ok(tensor)
    }

    /// Short-time Fourier transform
    fn stft(&self, samples: &[f32]) -> Vec<Vec<Complex<f32>>> {
        let n_fft = self.config.n_fft;
        let hop_length = self.config.hop_length;
        let win_length = self.window.len();

        // Reflect-pad signal to match PyTorch's torch.stft with center=False
        // after manual reflect padding: pad = (n_fft - hop_length) / 2
        let pad_length = (n_fft - hop_length) / 2;
        let mut padded = Vec::with_capacity(pad_length + samples.len() + pad_length);

        // Left reflect padding: mirror from position 1 outward
        for i in (1..=pad_length).rev() {
            let idx = if i < samples.len() {
                i
            } else {
                samples.len() - 1
            };
            padded.push(samples[idx]);
        }
        padded.extend_from_slice(samples);
        // Right reflect padding: mirror from position len-2 inward
        for i in 0..pad_length {
            let idx = if samples.len() >= 2 + i {
                samples.len() - 2 - i
            } else {
                0
            };
            padded.push(samples[idx]);
        }

        // Setup FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);

        let n_frames = (padded.len() - n_fft) / hop_length + 1;
        let mut result = Vec::with_capacity(n_frames);

        for i in 0..n_frames {
            let start = i * hop_length;

            // Apply window and prepare FFT input
            let mut buffer: Vec<FftComplex<f32>> = (0..n_fft)
                .map(|j| {
                    let sample = if j < win_length && start + j < padded.len() {
                        padded[start + j] * self.window[j]
                    } else {
                        0.0
                    };
                    FftComplex::new(sample, 0.0)
                })
                .collect();

            // Perform FFT
            fft.process(&mut buffer);

            // Take positive frequencies only (n_fft/2 + 1)
            let frame: Vec<Complex<f32>> = buffer
                .iter()
                .take(n_fft / 2 + 1)
                .map(|c| Complex::new(c.re, c.im))
                .collect();

            result.push(frame);
        }

        result
    }

    /// Apply mel filterbank to power spectrogram
    fn apply_mel_filterbank(&self, power_spec: &[Vec<f32>]) -> Vec<Vec<f32>> {
        power_spec
            .iter()
            .map(|frame| {
                self.mel_basis
                    .iter()
                    .map(|filter| filter.iter().zip(frame.iter()).map(|(f, p)| f * p).sum())
                    .collect()
            })
            .collect()
    }

    /// Convert frequency in Hz to mel scale (Slaney / O'Shaughnessy).
    ///
    /// This is the librosa default (`htk=False`): linear below 1000 Hz,
    /// logarithmic above.
    fn hz_to_mel(f: f32) -> f32 {
        const F_SP: f32 = 200.0 / 3.0; // 66.667 Hz per mel below break
        const MIN_LOG_HZ: f32 = 1000.0;
        const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP; // 15.0
        const LOGSTEP: f32 = 0.068_751_74; // ln(6.4) / 27

        if f < MIN_LOG_HZ {
            f / F_SP
        } else {
            MIN_LOG_MEL + (f / MIN_LOG_HZ).ln() / LOGSTEP
        }
    }

    /// Convert mel value to Hz (Slaney / O'Shaughnessy).
    fn mel_to_hz(m: f32) -> f32 {
        const F_SP: f32 = 200.0 / 3.0;
        const MIN_LOG_HZ: f32 = 1000.0;
        const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
        const LOGSTEP: f32 = 0.068_751_74; // ln(6.4) / 27

        if m < MIN_LOG_MEL {
            m * F_SP
        } else {
            MIN_LOG_HZ * ((m - MIN_LOG_MEL) * LOGSTEP).exp()
        }
    }

    /// Create mel filterbank matrix (matches `librosa.filters.mel` defaults).
    ///
    /// Uses the Slaney mel scale with Slaney area-normalization, matching
    /// `librosa.filters.mel(sr=..., n_fft=..., n_mels=..., fmin=..., fmax=..., norm="slaney")`
    /// which is the default in librosa ≥ 0.10.
    fn create_mel_filterbank(
        sample_rate: u32,
        n_fft: usize,
        n_mels: usize,
        fmin: f32,
        fmax: f32,
    ) -> Vec<Vec<f32>> {
        let n_freqs = n_fft / 2 + 1;

        // Create linearly spaced mel points
        let mel_min = Self::hz_to_mel(fmin);
        let mel_max = Self::hz_to_mel(fmax);
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();

        // Convert to Hz
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();

        // FFT bin center frequencies
        let fft_freqs: Vec<f32> = (0..n_freqs)
            .map(|i| i as f32 * sample_rate as f32 / n_fft as f32)
            .collect();

        // Build triangular filterbank (librosa ramps approach)
        let mut filterbank = vec![vec![0.0f32; n_freqs]; n_mels];

        for i in 0..n_mels {
            let f_lower = hz_points[i];
            let f_center = hz_points[i + 1];
            let f_upper = hz_points[i + 2];

            for (j, &freq) in fft_freqs.iter().enumerate() {
                if freq >= f_lower && freq <= f_center && f_center > f_lower {
                    filterbank[i][j] = (freq - f_lower) / (f_center - f_lower);
                } else if freq > f_center && freq <= f_upper && f_upper > f_center {
                    filterbank[i][j] = (f_upper - freq) / (f_upper - f_center);
                }
            }

            // Slaney area-normalization: scale each filter so energy is
            // approximately constant per channel
            let band_width = hz_points[i + 2] - hz_points[i];
            if band_width > 0.0 {
                let enorm = 2.0 / band_width;
                for val in &mut filterbank[i] {
                    *val *= enorm;
                }
            }
        }

        filterbank
    }

    /// Create Hann window
    fn hann_window(length: usize) -> Vec<f32> {
        (0..length)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / length as f32).cos()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_mel_config_default() {
        let config = MelConfig::default();
        assert_eq!(config.sample_rate, 24000);
        assert_eq!(config.n_fft, 400);
        assert_eq!(config.hop_length, 160);
        assert_eq!(config.n_mels, 128);
        assert!((config.fmin - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mel_config_custom() {
        let config = MelConfig {
            sample_rate: 16000,
            n_fft: 512,
            hop_length: 256,
            win_length: Some(512),
            n_mels: 80,
            fmin: 80.0,
            fmax: Some(7600.0),
        };
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.fmax, Some(7600.0));
    }

    #[test]
    fn test_hann_window() {
        let window = MelSpectrogram::hann_window(4);
        assert_eq!(window.len(), 4);
        assert!((window[0] - 0.0).abs() < 1e-6);
        assert!((window[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hann_window_larger() {
        let window = MelSpectrogram::hann_window(256);
        assert_eq!(window.len(), 256);
        // Window starts near zero
        assert!(window[0] < 0.01);
        // Peak at center
        assert!(window[128] > 0.99);
        // Values are symmetric around center
        assert!((window[1] - window[255]).abs() < 0.01);
        assert!((window[64] - window[192]).abs() < 0.01);
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let mel = MelSpectrogram::new(MelConfig::default());
        assert_eq!(mel.mel_basis.len(), 128);
        assert_eq!(mel.mel_basis[0].len(), 201); // n_fft/2 + 1 = 400/2 + 1
    }

    #[test]
    fn test_mel_filterbank_triangular() {
        let mel = MelSpectrogram::new(MelConfig {
            n_mels: 4,
            ..Default::default()
        });
        // Each filter should have triangular shape (non-negative values)
        for filter in &mel.mel_basis {
            for &val in filter {
                assert!(val >= 0.0);
            }
        }
    }

    #[test]
    fn test_compute_mel_silence() {
        let mel = MelSpectrogram::new(MelConfig::default());
        let samples = vec![0.0f32; 24000];
        let result = mel.compute(&samples);
        assert!(!result.is_empty());
        assert_eq!(result[0].len(), 128);
        // Silence should produce very small values
        for frame in &result {
            for &val in frame {
                assert!(val < 1e-6);
            }
        }
    }

    #[test]
    fn test_compute_mel_sine_wave() {
        let mel = MelSpectrogram::new(MelConfig::default());
        // Generate 440Hz sine wave (A4 note)
        let samples: Vec<f32> = (0..24000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let result = mel.compute(&samples);
        assert!(!result.is_empty());
        // Should have non-zero energy
        let total_energy: f32 = result.iter().flat_map(|frame| frame.iter()).sum();
        assert!(total_energy > 0.0);
    }

    #[test]
    fn test_compute_log_mel() {
        let mel = MelSpectrogram::new(MelConfig::default());
        let samples: Vec<f32> = (0..24000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let result = mel.compute_log(&samples);
        assert!(!result.is_empty());
        // Log mel should have negative values (log of small numbers)
        let has_negative = result
            .iter()
            .flat_map(|frame| frame.iter())
            .any(|&v| v < 0.0);
        assert!(has_negative);
    }

    #[test]
    fn test_compute_tensor() {
        let mel = MelSpectrogram::new(MelConfig::default());
        let samples = vec![0.0f32; 4800]; // 0.2 seconds
        let device = Device::Cpu;
        let tensor = mel.compute_tensor(&samples, &device).unwrap();
        // Shape should be [n_mels, n_frames]
        assert_eq!(tensor.dims()[0], 128);
    }

    #[test]
    fn test_stft_output_frames() {
        let mel = MelSpectrogram::new(MelConfig {
            n_fft: 400,
            hop_length: 160,
            ..Default::default()
        });
        // With padding, number of frames should be predictable
        let samples = vec![0.0f32; 1600]; // 10 hops
        let result = mel.compute(&samples);
        assert!(!result.is_empty());
    }
}

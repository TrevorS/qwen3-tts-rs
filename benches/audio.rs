//! Micro-benchmarks for audio processing (mel spectrograms, resampling).
//!
//! Run with: `cargo bench -- audio`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use qwen3_tts::audio::{resample, AudioBuffer, MelConfig, MelSpectrogram};
use std::f32::consts::PI;
use std::hint::black_box;

/// Generate a 440 Hz sine wave at 24 kHz for the given duration in seconds.
fn sine_wave(duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    let n = (duration_secs * sample_rate as f32) as usize;
    (0..n)
        .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
        .collect()
}

fn bench_mel_spectrogram(c: &mut Criterion) {
    let mel = MelSpectrogram::new(MelConfig::default());
    let mut group = c.benchmark_group("mel_spectrogram");

    for duration in [0.5, 2.0, 10.0] {
        let samples = sine_wave(duration, 24000);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{duration}s")),
            &duration,
            |b, _| {
                b.iter(|| mel.compute(black_box(&samples)));
            },
        );
    }
    group.finish();
}

fn bench_resample(c: &mut Criterion) {
    let mut group = c.benchmark_group("resample");

    for (from_rate, to_rate) in [(12000u32, 24000u32), (48000, 24000)] {
        for duration in [0.5, 2.0, 10.0] {
            let samples = sine_wave(duration, from_rate);
            let audio = AudioBuffer::new(samples, from_rate);

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{from_rate}to{to_rate}_{duration}s")),
                &(from_rate, to_rate, duration),
                |b, _| {
                    b.iter(|| resample(black_box(&audio), to_rate).unwrap());
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_mel_spectrogram, bench_resample);
criterion_main!(benches);

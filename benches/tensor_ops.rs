//! Micro-benchmarks for tensor operations (codes_to_tensor).
//!
//! Run with: `cargo bench -- tensor_ops`

use candle_core::Device;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use qwen3_tts::codes_to_tensor;
use std::hint::black_box;

/// Build `n_frames` dummy codec frames (16 codebooks each).
fn make_frames(n_frames: usize) -> Vec<Vec<u32>> {
    (0..n_frames)
        .map(|f| (0..16).map(|q| ((f * 16 + q) % 3072) as u32).collect())
        .collect()
}

fn bench_codes_to_tensor(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("codes_to_tensor");

    // 12 frames ≈ 1s, 60 frames ≈ 5s, 240 frames ≈ 20s of audio at 12 Hz
    for n_frames in [12, 60, 240] {
        let frames = make_frames(n_frames);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{n_frames}_frames")),
            &n_frames,
            |b, _| {
                b.iter(|| codes_to_tensor(black_box(&frames), &device).unwrap());
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_codes_to_tensor);
criterion_main!(benches);

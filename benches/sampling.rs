//! Micro-benchmarks for sampling and logit processing functions.
//!
//! Run with: `cargo bench -- sampling`

use candle_core::{DType, Device, Tensor};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use qwen3_tts::generation::{
    apply_repetition_penalty, apply_token_suppression, sample, GenerationConfig, SamplingContext,
};
use std::hint::black_box;

fn random_logits(vocab_size: usize, device: &Device) -> Tensor {
    // Deterministic "random" logits via a simple pattern
    let data: Vec<f32> = (0..vocab_size)
        .map(|i| (i as f32 * 0.1).sin() * 5.0)
        .collect();
    Tensor::new(data, device).unwrap().unsqueeze(0).unwrap() // [1, vocab]
}

fn bench_sample_top_k(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("sample_top_k");

    for vocab_size in [3072, 32000] {
        let logits = random_logits(vocab_size, &device);
        let config = GenerationConfig {
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0, // disable top-p to isolate top-k
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("vocab_{vocab_size}")),
            &vocab_size,
            |b, _| {
                let mut ctx = SamplingContext::new(Some(42));
                b.iter(|| sample(black_box(&logits), black_box(&config), &mut ctx).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_sample_top_p(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("sample_top_p");

    for p in [0.5, 0.9, 0.95] {
        let logits = random_logits(3072, &device);
        let config = GenerationConfig {
            temperature: 0.9,
            top_k: 0, // disable top-k to isolate top-p
            top_p: p,
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::from_parameter(format!("p_{p}")), &p, |b, _| {
            let mut ctx = SamplingContext::new(Some(42));
            b.iter(|| sample(black_box(&logits), black_box(&config), &mut ctx).unwrap());
        });
    }
    group.finish();
}

fn bench_repetition_penalty(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("repetition_penalty");

    for (penalty, n_prev) in [(1.05, 0), (1.05, 100), (1.05, 500), (1.5, 100), (1.5, 500)] {
        let logits = random_logits(3072, &device);
        let prev_tokens = if n_prev > 0 {
            let ids: Vec<u32> = (0..n_prev).map(|i| (i % 3072) as u32).collect();
            Tensor::new(ids, &device).unwrap()
        } else {
            Tensor::zeros(0, DType::U32, &device).unwrap()
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("pen_{penalty}_prev_{n_prev}")),
            &(penalty, n_prev),
            |b, _| {
                b.iter(|| {
                    apply_repetition_penalty(black_box(&logits), black_box(&prev_tokens), penalty)
                        .unwrap()
                });
            },
        );
    }
    group.finish();
}

fn bench_token_suppression(c: &mut Criterion) {
    let device = Device::Cpu;
    let logits = random_logits(3072, &device);

    c.bench_function("token_suppression_codec", |b| {
        b.iter(|| {
            apply_token_suppression(black_box(&logits), black_box(3072), black_box(2150)).unwrap()
        });
    });
}

criterion_group!(
    benches,
    bench_sample_top_k,
    bench_sample_top_p,
    bench_repetition_penalty,
    bench_token_suppression,
);
criterion_main!(benches);

//! End-to-end TTS benchmark (requires model weights).
//!
//! Measures wall-clock time, RTF, tokens/sec, and optionally TTFA via streaming.
//!
//! Usage:
//! ```sh
//! cargo run --release --features cli --bin e2e_bench -- \
//!     --model-dir test_data --device auto --iterations 3
//!
//! # With JSON output:
//! cargo run --release --features cli --bin e2e_bench -- \
//!     --model-dir test_data --json-output results.json
//! ```

use anyhow::Result;
use clap::Parser;
use qwen3_tts::{
    device_info, models::talker::Speaker, parse_device, AudioBuffer, Qwen3TTS, SynthesisOptions,
    SAMPLES_PER_FRAME,
};
use serde::Serialize;
use std::time::Instant;

// ── CLI ──────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "e2e_bench", about = "End-to-end TTS benchmark")]
struct Args {
    /// Device: auto, cpu, cuda, cuda:N, metal
    #[arg(long, default_value = "auto")]
    device: String,

    /// Path to model directory
    #[arg(long)]
    model_dir: String,

    /// Number of warmup runs (not measured)
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Number of timed iterations (results are averaged)
    #[arg(long, default_value_t = 3)]
    iterations: usize,

    /// Random seed for reproducible generation
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Write JSON results to this path
    #[arg(long)]
    json_output: Option<String>,

    /// Also measure time-to-first-audio via streaming
    #[arg(long)]
    streaming: bool,
}

// ── Result types ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
struct BenchmarkResult {
    label: String,
    text: String,
    word_count: usize,
    wall_clock_ms: f64,
    audio_duration_secs: f64,
    /// Wall-clock / audio duration. Lower = faster. < 1.0 = faster than real-time.
    rtf: f64,
    /// Time to first audio chunk (streaming only).
    ttfa_ms: Option<f64>,
    /// Semantic frames per second of wall-clock time.
    tokens_per_sec: f64,
    frames_generated: usize,
    /// Resident set size on Linux, None elsewhere.
    peak_memory_mb: Option<f64>,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    device: String,
    model_dir: String,
    iterations: usize,
    results: Vec<BenchmarkResult>,
}

// ── Test corpus ──────────────────────────────────────────────────────────

fn test_corpus() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "short",
            "The quick brown fox jumps over the lazy dog near the river bank.",
        ),
        (
            "medium",
            "In a quiet village nestled between rolling hills and dense forests, \
             there lived an old clockmaker who spent his days repairing timepieces \
             from centuries past. His workshop, filled with the gentle ticking of \
             a hundred clocks, was a place where time itself seemed to slow down \
             and the outside world faded into silence.",
        ),
        (
            "long",
            "The development of artificial intelligence has been one of the most \
             transformative technological advances of the twenty-first century. From \
             natural language processing to computer vision, machine learning models \
             have achieved remarkable performance across a wide range of tasks that \
             were once considered the exclusive domain of human intelligence. Speech \
             synthesis, in particular, has seen dramatic improvements with the \
             introduction of neural network architectures that can generate \
             high-fidelity audio from text input. These systems learn complex \
             patterns of prosody, intonation, and rhythm from large datasets of \
             recorded speech, producing output that is increasingly difficult to \
             distinguish from natural human speech. The implications of this \
             technology extend across many fields, including accessibility, \
             entertainment, education, and human-computer interaction.",
        ),
    ]
}

// ── Memory measurement ───────────────────────────────────────────────────

fn peak_memory_mb() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmRSS:"))
                    .and_then(|l| {
                        l.split_whitespace()
                            .nth(1)
                            .and_then(|v| v.parse::<f64>().ok())
                    })
                    .map(|kb| kb / 1024.0)
            })
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

// ── Benchmark runner ─────────────────────────────────────────────────────

fn run_single(
    model: &Qwen3TTS,
    text: &str,
    seed: u64,
    streaming: bool,
) -> Result<(AudioBuffer, usize, f64, Option<f64>)> {
    let options = SynthesisOptions {
        seed: Some(seed),
        ..Default::default()
    };

    if streaming {
        let start = Instant::now();
        let mut ttfa: Option<f64> = None;
        let mut all_samples: Vec<f32> = Vec::new();
        let mut total_frames = 0usize;

        let session = model.synthesize_streaming(
            text,
            Speaker::Ryan,
            qwen3_tts::models::talker::Language::English,
            options,
        )?;

        for chunk_result in session {
            let chunk = chunk_result?;
            if ttfa.is_none() {
                ttfa = Some(start.elapsed().as_secs_f64() * 1000.0);
            }
            total_frames += chunk.len() / SAMPLES_PER_FRAME;
            all_samples.extend_from_slice(&chunk.samples);
        }

        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        let audio = AudioBuffer::new(all_samples, 24000);
        Ok((audio, total_frames, wall_ms, ttfa))
    } else {
        let start = Instant::now();
        let audio = model.synthesize(text, Some(options))?;
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        let frames = audio.len() / SAMPLES_PER_FRAME;
        Ok((audio, frames, wall_ms, None))
    }
}

fn run_benchmark(
    model: &Qwen3TTS,
    label: &str,
    text: &str,
    args: &Args,
) -> Result<BenchmarkResult> {
    let word_count = text.split_whitespace().count();

    // Warmup
    for _ in 0..args.warmup {
        let _ = run_single(model, text, args.seed, args.streaming)?;
    }

    // Timed runs
    let mut wall_times = Vec::with_capacity(args.iterations);
    let mut ttfa_times: Vec<f64> = Vec::new();
    let mut last_audio_dur = 0.0f64;
    let mut last_frames = 0usize;

    for _ in 0..args.iterations {
        let (audio, frames, wall_ms, ttfa) = run_single(model, text, args.seed, args.streaming)?;
        wall_times.push(wall_ms);
        if let Some(t) = ttfa {
            ttfa_times.push(t);
        }
        last_audio_dur = audio.len() as f64 / audio.sample_rate as f64;
        last_frames = frames;
    }

    let avg_wall_ms = wall_times.iter().sum::<f64>() / wall_times.len() as f64;
    let avg_ttfa = if ttfa_times.is_empty() {
        None
    } else {
        Some(ttfa_times.iter().sum::<f64>() / ttfa_times.len() as f64)
    };

    let wall_secs = avg_wall_ms / 1000.0;
    let rtf = if last_audio_dur > 0.0 {
        wall_secs / last_audio_dur
    } else {
        f64::INFINITY
    };
    let tokens_per_sec = if wall_secs > 0.0 {
        last_frames as f64 / wall_secs
    } else {
        0.0
    };

    Ok(BenchmarkResult {
        label: label.to_string(),
        text: text.to_string(),
        word_count,
        wall_clock_ms: avg_wall_ms,
        audio_duration_secs: last_audio_dur,
        rtf,
        ttfa_ms: avg_ttfa,
        tokens_per_sec,
        frames_generated: last_frames,
        peak_memory_mb: peak_memory_mb(),
    })
}

// ── Table formatting ─────────────────────────────────────────────────────

fn print_table(results: &[BenchmarkResult]) {
    println!();
    println!(
        "{:<8} {:>6} {:>10} {:>10} {:>8} {:>10} {:>8} {:>8}",
        "Label", "Words", "Wall (ms)", "Audio (s)", "RTF", "TTFA (ms)", "Tok/s", "Mem (MB)"
    );
    println!("{}", "-".repeat(78));

    for r in results {
        let ttfa = r
            .ttfa_ms
            .map(|t| format!("{t:.1}"))
            .unwrap_or_else(|| "-".into());
        let mem = r
            .peak_memory_mb
            .map(|m| format!("{m:.0}"))
            .unwrap_or_else(|| "-".into());

        println!(
            "{:<8} {:>6} {:>10.1} {:>10.2} {:>8.3} {:>10} {:>8.1} {:>8}",
            r.label,
            r.word_count,
            r.wall_clock_ms,
            r.audio_duration_secs,
            r.rtf,
            ttfa,
            r.tokens_per_sec,
            mem,
        );
    }
    println!();
}

// ── Main ─────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let device = parse_device(&args.device)?;
    println!("Device: {}", device_info(&device));
    println!("Model:  {}", args.model_dir);
    println!(
        "Config: {} warmup, {} iterations, seed {}",
        args.warmup, args.iterations, args.seed
    );
    if args.streaming {
        println!("Mode:   streaming (measuring TTFA)");
    }
    println!();

    println!("Loading model...");
    let model = Qwen3TTS::from_pretrained(&args.model_dir, device)?;
    println!("Model loaded.\n");

    let corpus = test_corpus();
    let mut results = Vec::with_capacity(corpus.len());

    for (label, text) in &corpus {
        print!("Benchmarking [{label}]...");
        std::io::Write::flush(&mut std::io::stdout())?;
        let result = run_benchmark(&model, label, text, &args)?;
        println!(
            " RTF={:.3} ({:.0}ms, {:.2}s audio)",
            result.rtf, result.wall_clock_ms, result.audio_duration_secs
        );
        results.push(result);
    }

    print_table(&results);

    if let Some(ref path) = args.json_output {
        let report = BenchmarkReport {
            device: device_info(model.device()),
            model_dir: args.model_dir.clone(),
            iterations: args.iterations,
            results: results.clone(),
        };
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(path, &json)?;
        println!("JSON results written to {path}");
    }

    Ok(())
}

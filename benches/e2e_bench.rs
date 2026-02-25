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
//!
//! # Compare sliding window vs full context decoder:
//! cargo run --release --features cli --bin e2e_bench -- \
//!     --model-dir test_data --compare-decode-window 80
//! ```

use anyhow::Result;
use clap::Parser;
use qwen3_tts::{
    device_info, models::talker::Speaker, parse_device, AudioBuffer, Qwen3TTS, SynthesisOptions,
    SynthesisTiming, SAMPLES_PER_FRAME,
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

    /// Only run these labels (comma-separated, e.g. "short,medium")
    #[arg(long, value_delimiter = ',')]
    labels: Vec<String>,

    /// Decode window size for sliding window decoder (0 = full context)
    #[arg(long, default_value_t = 0)]
    decode_window: usize,

    /// Compare sliding window vs full context: runs each benchmark twice
    /// with the specified window size and reports the difference
    #[arg(long)]
    compare_decode_window: Option<usize>,

    /// Profile code predictor to identify bottlenecks
    #[arg(long)]
    profile: bool,
}

// ── Result types ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
struct StageBreakdown {
    prefill_ms: f64,
    generation_ms: f64,
    generation_frames: usize,
    decode_ms: f64,
}

impl From<&SynthesisTiming> for StageBreakdown {
    fn from(t: &SynthesisTiming) -> Self {
        Self {
            prefill_ms: t.prefill_ms,
            generation_ms: t.generation_ms,
            generation_frames: t.generation_frames,
            decode_ms: t.decode_ms,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkResult {
    label: String,
    text: String,
    word_count: usize,
    wall_clock_ms: f64,
    /// Standard deviation of wall-clock times across iterations.
    wall_clock_stddev_ms: f64,
    /// Minimum wall-clock time across iterations.
    wall_clock_min_ms: f64,
    /// Maximum wall-clock time across iterations.
    wall_clock_max_ms: f64,
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
    /// Per-stage timing breakdown (averaged across iterations).
    stages: Option<StageBreakdown>,
    /// Decode window size (0 = full context).
    decode_window: usize,
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

// ── Environment checks ──────────────────────────────────────────────────

fn warn_cpu_governor() {
    #[cfg(target_os = "linux")]
    {
        if let Ok(governor) =
            std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        {
            let governor = governor.trim();
            if governor != "performance" {
                eprintln!(
                    "WARNING: CPU governor is '{}', not 'performance'. \
                     Results may be noisy. Fix with:\n  \
                     sudo cpupower frequency-set -g performance\n",
                    governor,
                );
            }
        }
    }
}

// ── Benchmark runner ─────────────────────────────────────────────────────

struct SingleRunResult {
    audio: AudioBuffer,
    frames: usize,
    wall_ms: f64,
    ttfa_ms: Option<f64>,
    timing: Option<SynthesisTiming>,
}

fn run_single(
    model: &Qwen3TTS,
    text: &str,
    seed: u64,
    streaming: bool,
    decode_window: usize,
) -> Result<SingleRunResult> {
    let options = SynthesisOptions {
        seed: Some(seed),
        decode_window_frames: decode_window,
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
        Ok(SingleRunResult {
            audio,
            frames: total_frames,
            wall_ms,
            ttfa_ms: ttfa,
            timing: None,
        })
    } else {
        let start = Instant::now();
        let (audio, timing) = model.synthesize_with_timing(
            text,
            Speaker::Ryan,
            qwen3_tts::models::talker::Language::English,
            Some(options),
        )?;
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        let frames = audio.len() / SAMPLES_PER_FRAME;
        Ok(SingleRunResult {
            audio,
            frames,
            wall_ms,
            ttfa_ms: None,
            timing: Some(timing),
        })
    }
}

fn run_benchmark(
    model: &Qwen3TTS,
    label: &str,
    text: &str,
    args: &Args,
    decode_window: usize,
) -> Result<BenchmarkResult> {
    let word_count = text.split_whitespace().count();

    // Warmup
    for _ in 0..args.warmup {
        let _ = run_single(model, text, args.seed, args.streaming, decode_window)?;
    }

    // Timed runs
    let mut wall_times = Vec::with_capacity(args.iterations);
    let mut ttfa_times: Vec<f64> = Vec::new();
    let mut timings: Vec<SynthesisTiming> = Vec::new();
    let mut last_audio_dur = 0.0f64;
    let mut last_frames = 0usize;

    for _ in 0..args.iterations {
        // Sync device before each timed run to drain any stale GPU work
        qwen3_tts::sync_device(model.device())?;

        let run = run_single(model, text, args.seed, args.streaming, decode_window)?;
        wall_times.push(run.wall_ms);
        if let Some(t) = run.ttfa_ms {
            ttfa_times.push(t);
        }
        if let Some(t) = run.timing {
            timings.push(t);
        }
        last_audio_dur = run.audio.len() as f64 / run.audio.sample_rate as f64;
        last_frames = run.frames;
    }

    let n = wall_times.len() as f64;
    let avg_wall_ms = wall_times.iter().sum::<f64>() / n;
    let wall_min = wall_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let wall_max = wall_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let wall_stddev = if wall_times.len() > 1 {
        let variance = wall_times
            .iter()
            .map(|t| (t - avg_wall_ms).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        variance.sqrt()
    } else {
        0.0
    };

    let avg_ttfa = if ttfa_times.is_empty() {
        None
    } else {
        Some(ttfa_times.iter().sum::<f64>() / ttfa_times.len() as f64)
    };

    // Average stage timings across all iterations
    let avg_stages = if timings.is_empty() {
        None
    } else {
        let tn = timings.len() as f64;
        Some(StageBreakdown {
            prefill_ms: timings.iter().map(|t| t.prefill_ms).sum::<f64>() / tn,
            generation_ms: timings.iter().map(|t| t.generation_ms).sum::<f64>() / tn,
            generation_frames: timings.last().map(|t| t.generation_frames).unwrap_or(0),
            decode_ms: timings.iter().map(|t| t.decode_ms).sum::<f64>() / tn,
        })
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
        wall_clock_stddev_ms: wall_stddev,
        wall_clock_min_ms: wall_min,
        wall_clock_max_ms: wall_max,
        audio_duration_secs: last_audio_dur,
        rtf,
        ttfa_ms: avg_ttfa,
        tokens_per_sec,
        frames_generated: last_frames,
        peak_memory_mb: peak_memory_mb(),
        stages: avg_stages,
        decode_window,
    })
}

// ── Table formatting ─────────────────────────────────────────────────────

fn print_table(results: &[BenchmarkResult]) {
    let has_stages = results.iter().any(|r| r.stages.is_some());

    println!();
    if has_stages {
        println!(
            "{:<8} {:>6} {:>10} {:>10} {:>8} {:>10} {:>8} {:>10} {:>10} {:>10} {:>10}",
            "Label",
            "Words",
            "Wall (ms)",
            "±stddev",
            "Audio (s)",
            "RTF",
            "Tok/s",
            "Mem (MB)",
            "Prefill",
            "Generate",
            "Decode"
        );
        println!("{}", "-".repeat(116));
    } else {
        println!(
            "{:<8} {:>6} {:>10} {:>10} {:>8} {:>10} {:>8} {:>8}",
            "Label", "Words", "Wall (ms)", "Audio (s)", "RTF", "TTFA (ms)", "Tok/s", "Mem (MB)"
        );
        println!("{}", "-".repeat(78));
    }

    for r in results {
        let mem = r
            .peak_memory_mb
            .map(|m| format!("{m:.0}"))
            .unwrap_or_else(|| "-".into());

        if has_stages {
            let (prefill, gen, decode) = if let Some(ref s) = r.stages {
                let total = s.prefill_ms + s.generation_ms + s.decode_ms;
                let pct = |v: f64| {
                    if total > 0.0 {
                        format!("{v:.0}ms ({:.0}%)", v / total * 100.0)
                    } else {
                        format!("{v:.0}ms")
                    }
                };
                (pct(s.prefill_ms), pct(s.generation_ms), pct(s.decode_ms))
            } else {
                ("-".into(), "-".into(), "-".into())
            };
            println!(
                "{:<8} {:>6} {:>10.1} {:>10.1} {:>10.2} {:>8.3} {:>10.1} {:>8} {:>10} {:>10} {:>10}",
                r.label,
                r.word_count,
                r.wall_clock_ms,
                r.wall_clock_stddev_ms,
                r.audio_duration_secs,
                r.rtf,
                r.tokens_per_sec,
                mem,
                prefill,
                gen,
                decode,
            );
        } else {
            let ttfa = r
                .ttfa_ms
                .map(|t| format!("{t:.1}"))
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
    }
    println!();
}

// ── Main ─────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    // Use chrome tracing when `profiling` feature is active, otherwise plain fmt.
    let _profiling_guard = qwen3_tts::profiling::init();
    if _profiling_guard.is_none() {
        tracing_subscriber::fmt::init();
    }
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
    if let Some(window) = args.compare_decode_window {
        println!("Compare: decode_window=0 (full) vs decode_window={window} (sliding)");
    } else if args.decode_window > 0 {
        println!("Decode: sliding window ({} frames)", args.decode_window);
    }
    println!();

    // Warn if CPU governor isn't set to performance (Linux only)
    warn_cpu_governor();

    println!("Loading model...");
    let model = Qwen3TTS::from_pretrained(&args.model_dir, device)?;
    println!("Model loaded.");

    // GPU warmup: run a small tensor op to ensure CUDA context, JIT kernels,
    // and memory pools are fully initialized before we start timing.
    qwen3_tts::sync_device(model.device())?;
    println!();

    let corpus: Vec<_> = test_corpus()
        .into_iter()
        .filter(|(label, _)| args.labels.is_empty() || args.labels.iter().any(|l| l == label))
        .collect();

    // Comparison mode: run each benchmark with both full context and sliding window
    if let Some(window_size) = args.compare_decode_window {
        print_comparison_table(&model, &corpus, &args, window_size)?;
    } else {
        // Normal mode
        let mut results = Vec::with_capacity(corpus.len());

        for (label, text) in &corpus {
            print!("Benchmarking [{label}]...");
            std::io::Write::flush(&mut std::io::stdout())?;
            let result = run_benchmark(&model, label, text, &args, args.decode_window)?;
            println!(
                " RTF={:.3} ({:.0}ms ±{:.0}, {:.2}s audio)",
                result.rtf,
                result.wall_clock_ms,
                result.wall_clock_stddev_ms,
                result.audio_duration_secs,
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
    }

    Ok(())
}

// ── Comparison mode ───────────────────────────────────────────────────────

fn print_comparison_table(
    model: &Qwen3TTS,
    corpus: &[(&str, &str)],
    args: &Args,
    window_size: usize,
) -> Result<()> {
    println!("=== Sliding Window Decoder Comparison ===");
    println!("Comparing full context (window=0) vs sliding window (window={window_size})");
    println!();
    println!(
        "{:<8} {:>8} {:>12} {:>12} {:>10} {:>10} {:>10}",
        "Label", "Words", "Full (ms)", "Window (ms)", "Speedup", "Full RTF", "Win RTF"
    );
    println!("{}", "-".repeat(78));

    let mut total_full_ms = 0.0f64;
    let mut total_window_ms = 0.0f64;

    for (label, text) in corpus {
        print!("[{label}] full...");
        std::io::Write::flush(&mut std::io::stdout())?;
        let full_result = run_benchmark(model, label, text, args, 0)?;
        print!(" window...");
        std::io::Write::flush(&mut std::io::stdout())?;
        let window_result = run_benchmark(model, label, text, args, window_size)?;

        let speedup = if window_result.wall_clock_ms > 0.0 {
            full_result.wall_clock_ms / window_result.wall_clock_ms
        } else {
            0.0
        };

        total_full_ms += full_result.wall_clock_ms;
        total_window_ms += window_result.wall_clock_ms;

        println!(
            " {:>8.1} {:>12.1} {:>12.1} {:>9.2}x {:>10.3} {:>10.3}",
            full_result.word_count,
            full_result.wall_clock_ms,
            window_result.wall_clock_ms,
            speedup,
            full_result.rtf,
            window_result.rtf
        );
    }

    println!("{}", "-".repeat(78));
    let total_speedup = if total_window_ms > 0.0 {
        total_full_ms / total_window_ms
    } else {
        0.0
    };
    println!(
        "{:<8} {:>8} {:>12.1} {:>12.1} {:>9.2}x",
        "TOTAL", "", total_full_ms, total_window_ms, total_speedup
    );
    println!();

    if total_speedup > 1.0 {
        println!(
            "✓ Sliding window is {:.1}% faster",
            (total_speedup - 1.0) * 100.0
        );
    } else if total_speedup < 1.0 {
        println!(
            "✗ Sliding window is {:.1}% slower",
            (1.0 - total_speedup) * 100.0
        );
    } else {
        println!("= No significant difference");
    }

    Ok(())
}

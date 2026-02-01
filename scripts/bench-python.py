"""
Benchmark the official Python Qwen3-TTS against the same test corpus
used by qwen3-tts-rs e2e_bench.

Requires qwen-tts package installed. Run from the Qwen3-TTS venv:

    cd ../Qwen3-TTS
    pip install -e .
    pip install flash-attn --no-build-isolation
    python ../qwen3-tts-rs/scripts/bench-python.py \
        --model-dir ../qwen3-tts-rs/test_data/models/1.7B-CustomVoice

Or without flash attention:
    python ../qwen3-tts-rs/scripts/bench-python.py \
        --model-dir ../qwen3-tts-rs/test_data/models/1.7B-CustomVoice \
        --no-flash-attn
"""

import argparse
import json
import time

import torch
from qwen_tts import Qwen3TTSModel

TEXTS = {
    "short": "The quick brown fox jumps over the lazy dog near the river bank.",
    "medium": (
        "In a quiet village nestled between rolling hills and dense forests, "
        "there lived an old clockmaker who spent his days repairing timepieces "
        "from centuries past. His workshop, filled with the gentle ticking of a "
        "hundred clocks, was a place where time itself seemed to slow down and "
        "the outside world faded into silence."
    ),
    "long": (
        "The development of artificial intelligence has been one of the most "
        "transformative technological advances of the twenty-first century. From "
        "natural language processing to computer vision, machine learning models "
        "have achieved remarkable performance across a wide range of tasks that "
        "were once considered the exclusive domain of human intelligence. Speech "
        "synthesis, in particular, has seen dramatic improvements with the "
        "introduction of neural network architectures that can generate "
        "high-fidelity audio from text input. These systems learn complex "
        "patterns of prosody, intonation, and rhythm from large datasets of "
        "recorded speech, producing output that is increasingly difficult to "
        "distinguish from natural human speech. The implications of this "
        "technology extend across many fields, including accessibility, "
        "entertainment, education, and human-computer interaction."
    ),
}

WORD_COUNTS = {"short": 13, "medium": 53, "long": 115}


def bench_model(model, label, text, iterations, warmup):
    for _ in range(warmup):
        wavs, sr = model.generate_custom_voice(
            text=text,
            language="English",
            speaker="Ryan",
            max_new_tokens=2048,
        )
        torch.cuda.synchronize()

    wall_times = []
    audio_duration = None
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        wavs, sr = model.generate_custom_voice(
            text=text,
            language="English",
            speaker="Ryan",
            max_new_tokens=2048,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        wall_times.append(t1 - t0)
        audio_duration = len(wavs[0]) / sr

    avg_wall = sum(wall_times) / len(wall_times)
    stddev = (sum((t - avg_wall) ** 2 for t in wall_times) / len(wall_times)) ** 0.5
    rtf = avg_wall / audio_duration if audio_duration else 0
    toks = (audio_duration * 12.0) / avg_wall if avg_wall > 0 else 0

    return {
        "label": label,
        "words": WORD_COUNTS[label],
        "wall_ms": avg_wall * 1000,
        "stddev_ms": stddev * 1000,
        "audio_s": audio_duration,
        "rtf": rtf,
        "tok_s": toks,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Python Qwen3-TTS")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--model-dir", default=None, help="Local model dir (overrides --model)")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    model_path = args.model_dir or args.model
    attn_impl = "eager" if args.no_flash_attn else "flash_attention_2"

    print("Device: CUDA")
    print(f"Model:  {model_path}")
    print(f"Attn:   {attn_impl}")
    print(f"Config: {args.warmup} warmup, {args.iterations} iterations")
    print()

    print("Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    print("Model loaded.\n")

    torch.cuda.reset_peak_memory_stats()
    results = []

    for label in ["short", "medium", "long"]:
        print(f"Benchmarking [{label}]...", end=" ", flush=True)
        r = bench_model(model, label, TEXTS[label], args.iterations, args.warmup)
        r["mem_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        results.append(r)
        print(
            f"RTF={r['rtf']:.3f} ({r['wall_ms']:.0f}ms ±{r['stddev_ms']:.0f}, {r['audio_s']:.2f}s audio)"
        )

    print()
    print(
        f"{'Label':<10s} {'Words':>5s} {'Wall (ms)':>10s} {'Audio (s)':>10s} {'RTF':>8s} {'Tok/s':>8s} {'Mem (MB)':>8s}"
    )
    print("-" * 62)
    for r in results:
        print(
            f"{r['label']:<10s} {r['words']:>5d} {r['wall_ms']:>10.1f} {r['audio_s']:>10.2f} {r['rtf']:>8.3f} {r['tok_s']:>8.1f} {r['mem_mb']:>8.0f}"
        )

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nJSON results written to {args.json_output}")


if __name__ == "__main__":
    main()

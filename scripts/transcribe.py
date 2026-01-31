#!/usr/bin/env python3
"""Transcribe WAV files using OpenAI Whisper (no ffmpeg required).

Loads audio via scipy to bypass the ffmpeg dependency, resamples to 16kHz,
and feeds directly to the Whisper Python API.

Usage:
    python transcribe.py [--model tiny|base|small|medium|large-v3] [--language en] \
                         [--expected "text"] file1.wav [file2.wav ...]
"""

import argparse
import sys

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import whisper


def load_wav_16k(path: str) -> tuple[np.ndarray, dict]:
    """Load a WAV file and resample to 16kHz mono float32.

    Returns (audio_16k, stats) where stats contains RMS, peak, silence%.
    """
    sr, audio = wav.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)

    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    silence = float(np.mean(np.abs(audio) < 0.01))
    duration = len(audio) / sr

    # Resample to 16kHz
    if sr != 16000:
        num_samples = int(len(audio) * 16000 / sr)
        audio = scipy.signal.resample(audio, num_samples).astype(np.float32)

    stats = {
        "duration": duration,
        "rms": rms,
        "peak": peak,
        "silence": silence,
        "sample_rate": sr,
    }
    return audio, stats


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis."""
    ref_words = reference.lower().strip().split()
    hyp_words = hypothesis.lower().strip().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Levenshtein on word level
    n = len(ref_words)
    m = len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / n


def main():
    parser = argparse.ArgumentParser(description="Transcribe WAV files with Whisper")
    parser.add_argument("files", nargs="+", help="WAV file paths")
    parser.add_argument(
        "--model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: large-v3). Use large-v3 for TTS evaluation.",
    )
    parser.add_argument("--language", default="en", help="Language hint (default: en)")
    parser.add_argument("--expected", default=None, help="Expected transcription for WER")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    print(f"Loading Whisper {args.model} model...", file=sys.stderr)
    model = whisper.load_model(args.model)

    results = []
    for path in args.files:
        try:
            audio, stats = load_wav_16k(path)
        except Exception as e:
            print(f"{path}: ERROR loading: {e}")
            results.append({"file": path, "error": str(e)})
            continue

        result = model.transcribe(audio, language=args.language)
        text = result["text"].strip()

        name = path.split("/")[-1]
        entry = {
            "file": name,
            "transcription": text,
            **stats,
        }

        if args.expected:
            wer = word_error_rate(args.expected, text)
            entry["wer"] = wer

        results.append(entry)

        if args.json:
            continue

        print(f"{name}:")
        print(f'  transcription: "{text}"')
        print(f"  duration: {stats['duration']:.2f}s")
        print(f"  rms: {stats['rms']:.4f}")
        print(f"  peak: {stats['peak']:.4f}")
        print(f"  silence: {stats['silence']:.1%}")
        if args.expected:
            print(f"  wer: {wer:.1%}")
        print()

    if args.json:
        import json

        print(json.dumps(results, indent=2))

    # Summary if multiple files
    if len(results) > 1 and not args.json:
        print("=== Summary ===")
        for r in results:
            if "error" in r:
                status = "ERROR"
            elif not r["transcription"]:
                status = "UNINTELLIGIBLE"
            elif args.expected:
                status = f"WER={r['wer']:.0%}"
            else:
                status = "OK" if r["transcription"] else "EMPTY"
            print(f'  {r["file"]}: {status} — "{r.get("transcription", "")}"')


if __name__ == "__main__":
    main()

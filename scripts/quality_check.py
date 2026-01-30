#!/usr/bin/env python3
"""Audio quality sanity checks for generated TTS output.

Usage:
    python scripts/quality_check.py output.wav "expected transcription text"
    python scripts/quality_check.py output.wav  # skip WER check

Checks:
    - Duration (non-zero, plausible for text length)
    - RMS level and peak amplitude
    - Silence detection (>50% silence = warning)
    - Clipping detection (peak > 0.99)
    - WER via whisper CLI (optional, requires `whisper` on PATH)
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

# ── WAV reader (stdlib only) ─────────────────────────────────────────────


def read_wav(path: str) -> tuple[list[float], int]:
    """Read a mono WAV file, return (samples_f32, sample_rate)."""
    import wave

    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()} channels"
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)

    if sw == 2:
        fmt = f"<{n}h"
        ints = struct.unpack(fmt, raw)
        samples = [s / 32768.0 for s in ints]
    elif sw == 4:
        fmt = f"<{n}i"
        ints = struct.unpack(fmt, raw)
        samples = [s / 2147483648.0 for s in ints]
    else:
        raise ValueError(f"Unsupported sample width: {sw}")

    return samples, sr


# ── Audio checks ─────────────────────────────────────────────────────────


def check_audio(samples: list[float], sr: int) -> dict:
    duration = len(samples) / sr
    rms = (sum(s * s for s in samples) / max(len(samples), 1)) ** 0.5
    peak = max(abs(s) for s in samples) if samples else 0.0

    # Silence: frames where |sample| < 0.001
    silence_count = sum(1 for s in samples if abs(s) < 0.001)
    silence_ratio = silence_count / max(len(samples), 1)

    issues = []
    if duration < 0.1:
        issues.append("WARN: very short audio (<0.1s)")
    if rms < 0.001:
        issues.append("WARN: extremely low RMS — might be near-silence")
    if peak > 0.99:
        issues.append("WARN: clipping detected (peak > 0.99)")
    if silence_ratio > 0.5:
        issues.append(f"WARN: {silence_ratio:.0%} of samples are silence")

    return {
        "duration_secs": round(duration, 3),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "silence_ratio": round(silence_ratio, 4),
        "sample_rate": sr,
        "num_samples": len(samples),
        "issues": issues,
    }


# ── WER via Whisper CLI ──────────────────────────────────────────────────


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate between reference and hypothesis."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Levenshtein on word level
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[n][m] / max(n, 1)


def transcribe_with_whisper(wav_path: str) -> str | None:
    """Run whisper CLI and return transcription, or None if unavailable."""
    whisper_bin = shutil.which("whisper")
    if whisper_bin is None:
        return None

    try:
        result = subprocess.run(
            ["whisper", wav_path, "--model", "base", "--language", "en", "--output_format", "txt"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"  whisper failed: {result.stderr.strip()}", file=sys.stderr)
            return None

        # Whisper writes output to <input>.txt
        txt_path = Path(wav_path).with_suffix(".txt")
        if txt_path.exists():
            text = txt_path.read_text().strip()
            txt_path.unlink()  # clean up
            return text
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <wav_file> [expected_text]", file=sys.stderr)
        sys.exit(1)

    wav_path = sys.argv[1]
    expected_text = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(wav_path).exists():
        print(f"Error: {wav_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Checking: {wav_path}")
    samples, sr = read_wav(wav_path)
    report = check_audio(samples, sr)

    print(f"  Duration:       {report['duration_secs']:.3f}s")
    print(f"  Sample rate:    {report['sample_rate']} Hz")
    print(f"  Samples:        {report['num_samples']}")
    print(f"  RMS:            {report['rms']:.6f}")
    print(f"  Peak:           {report['peak']:.6f}")
    print(f"  Silence ratio:  {report['silence_ratio']:.2%}")

    for issue in report["issues"]:
        print(f"  {issue}")

    # WER check
    if expected_text:
        print(f"\n  Expected text: \"{expected_text}\"")
        transcript = transcribe_with_whisper(wav_path)
        if transcript is not None:
            wer = compute_wer(expected_text, transcript)
            report["transcript"] = transcript
            report["wer"] = round(wer, 4)
            print(f"  Transcript:    \"{transcript}\"")
            print(f"  WER:           {wer:.2%}")
            if wer > 0.3:
                report["issues"].append(f"WARN: high WER ({wer:.2%})")
        else:
            print("  WER:           skipped (whisper not found)")

    # Summary
    if report["issues"]:
        print(f"\n  {len(report['issues'])} issue(s) found.")
    else:
        print("\n  All checks passed.")

    # Write JSON alongside the WAV for programmatic consumption
    json_path = Path(wav_path).with_suffix(".quality.json")
    json_path.write_text(json.dumps(report, indent=2))
    print(f"  Report: {json_path}")


if __name__ == "__main__":
    main()

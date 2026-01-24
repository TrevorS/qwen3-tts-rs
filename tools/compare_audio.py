#!/usr/bin/env python3
"""Compare audio outputs between Rust and Python implementations.

This tool compares WAV files sample-by-sample, computing metrics and
generating comparison reports with optional spectrograms.

Usage:
    python3 tools/compare_audio.py --rust test_data/rust_audio --python test_data/reference_audio --seed 42 --frames 25
    python3 tools/compare_audio.py --file1 audio1.wav --file2 audio2.wav
    python3 tools/compare_audio.py --rust test_data/rust_audio --python test_data/reference_audio --all
"""

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_audio_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load audio from WAV file."""
    if HAS_SOUNDFILE:
        data, sr = sf.read(path)
        return data.astype(np.float32), sr
    else:
        # Fallback: use wave module
        import wave
        with wave.open(str(path), 'rb') as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            data = wf.readframes(n_frames)
            # Assuming 16-bit PCM
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            return samples, sr


def load_audio_bin(path: Path) -> np.ndarray:
    """Load audio from binary float32 file."""
    data = path.read_bytes()
    return np.frombuffer(data, dtype=np.float32)


def load_codes_bin(path: Path) -> np.ndarray:
    """Load codes from binary int64 file."""
    data = path.read_bytes()
    return np.frombuffer(data, dtype=np.int64)


def compute_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute comparison metrics between two arrays."""
    min_len = min(len(a), len(b))
    a = a[:min_len]
    b = b[:min_len]

    diff = a - b
    abs_diff = np.abs(diff)

    metrics = {
        "length_a": len(a),
        "length_b": len(b),
        "min_length": min_len,
        "max_diff": float(np.max(abs_diff)),
        "mean_diff": float(np.mean(abs_diff)),
        "std_diff": float(np.std(diff)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_value_a": float(np.max(np.abs(a))),
        "max_value_b": float(np.max(np.abs(b))),
    }

    # Correlation (if non-zero signals)
    if np.std(a) > 1e-10 and np.std(b) > 1e-10:
        metrics["correlation"] = float(np.corrcoef(a, b)[0, 1])
    else:
        metrics["correlation"] = None

    # SNR (signal-to-noise ratio, treating diff as noise)
    signal_power = np.mean(a ** 2)
    noise_power = np.mean(diff ** 2)
    if noise_power > 1e-20:
        metrics["snr_db"] = float(10 * np.log10(signal_power / noise_power))
    else:
        metrics["snr_db"] = float('inf')

    return metrics


def compare_codes(rust_path: Path, python_path: Path) -> dict:
    """Compare codec outputs."""
    rust_codes = load_codes_bin(rust_path)
    python_codes = load_codes_bin(python_path)

    min_len = min(len(rust_codes), len(python_codes))
    rust_codes = rust_codes[:min_len]
    python_codes = python_codes[:min_len]

    match = np.array_equal(rust_codes, python_codes)
    diff_indices = np.where(rust_codes != python_codes)[0]

    result = {
        "rust_count": len(rust_codes),
        "python_count": len(python_codes),
        "match": match,
        "diff_count": len(diff_indices),
    }

    if len(diff_indices) > 0:
        result["first_diff_index"] = int(diff_indices[0])
        result["first_diffs"] = [
            {
                "index": int(i),
                "rust": int(rust_codes[i]),
                "python": int(python_codes[i])
            }
            for i in diff_indices[:5]
        ]

    return result


def compare_audio(rust_path: Path, python_path: Path, sample_rate: int = 24000) -> dict:
    """Compare audio outputs."""
    # Try loading as binary first (more reliable for exact comparison)
    rust_bin = rust_path.with_suffix('.bin')
    python_bin = python_path.with_suffix('.bin')

    if rust_bin.exists() and python_bin.exists():
        rust_audio = load_audio_bin(rust_bin)
        python_audio = load_audio_bin(python_bin)
    elif rust_path.exists() and python_path.exists():
        rust_audio, _ = load_audio_wav(rust_path)
        python_audio, _ = load_audio_wav(python_path)
    else:
        return {"error": f"Files not found: {rust_path} or {python_path}"}

    return compute_metrics(rust_audio, python_audio)


def plot_comparison(
    rust_audio: np.ndarray,
    python_audio: np.ndarray,
    sample_rate: int,
    output_path: Path,
    title: str = "Audio Comparison"
):
    """Generate comparison plots."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plots")
        return

    min_len = min(len(rust_audio), len(python_audio))
    rust_audio = rust_audio[:min_len]
    python_audio = python_audio[:min_len]

    time = np.arange(min_len) / sample_rate
    diff = rust_audio - python_audio

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Waveform comparison
    ax = axes[0]
    ax.plot(time, rust_audio, label='Rust', alpha=0.7, linewidth=0.5)
    ax.plot(time, python_audio, label='Python', alpha=0.7, linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'{title} - Waveforms')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Difference waveform
    ax = axes[1]
    ax.plot(time, diff, color='red', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Difference')
    ax.set_title('Difference (Rust - Python)')
    ax.grid(True, alpha=0.3)

    # Spectrogram - Rust
    ax = axes[2]
    ax.specgram(rust_audio, Fs=sample_rate, cmap='viridis')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectrogram - Rust')

    # Spectrogram - Python
    ax = axes[3]
    ax.specgram(python_audio, Fs=sample_rate, cmap='viridis')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectrogram - Python')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to: {output_path}")


def plot_spectrogram_diff(
    rust_audio: np.ndarray,
    python_audio: np.ndarray,
    sample_rate: int,
    output_path: Path
):
    """Generate spectrogram difference visualization."""
    if not HAS_MATPLOTLIB:
        return

    from scipy import signal

    min_len = min(len(rust_audio), len(python_audio))
    rust_audio = rust_audio[:min_len]
    python_audio = python_audio[:min_len]

    # Compute spectrograms
    nperseg = min(256, min_len // 4) if min_len >= 256 else min_len
    f, t, rust_spec = signal.spectrogram(rust_audio, fs=sample_rate, nperseg=nperseg)
    _, _, python_spec = signal.spectrogram(python_audio, fs=sample_rate, nperseg=nperseg)

    # Compute difference in log scale
    rust_log = 10 * np.log10(rust_spec + 1e-10)
    python_log = 10 * np.log10(python_spec + 1e-10)
    diff_log = rust_log - python_log

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Rust spectrogram
    ax = axes[0]
    im = ax.pcolormesh(t, f, rust_log, shading='auto', cmap='viridis')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectrogram - Rust (dB)')
    plt.colorbar(im, ax=ax)

    # Python spectrogram
    ax = axes[1]
    im = ax.pcolormesh(t, f, python_log, shading='auto', cmap='viridis')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectrogram - Python (dB)')
    plt.colorbar(im, ax=ax)

    # Difference
    ax = axes[2]
    vmax = max(abs(diff_log.min()), abs(diff_log.max()))
    im = ax.pcolormesh(t, f, diff_log, shading='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectrogram Difference (Rust - Python, dB)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved spectrogram diff to: {output_path}")


def compare_single_generation(
    rust_dir: Path,
    python_dir: Path,
    seed: int,
    frames: int,
    output_dir: Optional[Path] = None,
    generate_plots: bool = True
) -> dict:
    """Compare a single generation run."""
    result = {
        "seed": seed,
        "frames": frames,
    }

    # Compare codes
    rust_codes_path = rust_dir / f"codes_seed{seed}_frames{frames}.bin"
    python_codes_path = python_dir / f"codes_seed{seed}_frames{frames}.bin"

    if rust_codes_path.exists() and python_codes_path.exists():
        result["codes"] = compare_codes(rust_codes_path, python_codes_path)
    else:
        result["codes"] = {"error": "Codes files not found"}

    # Compare audio
    rust_audio_path = rust_dir / f"audio_seed{seed}_frames{frames}.wav"
    python_audio_path = python_dir / f"audio_seed{seed}_frames{frames}.wav"

    result["audio"] = compare_audio(rust_audio_path, python_audio_path)

    # Generate plots
    if generate_plots and output_dir and HAS_MATPLOTLIB:
        rust_bin = rust_dir / f"audio_seed{seed}_frames{frames}.bin"
        python_bin = python_dir / f"audio_seed{seed}_frames{frames}.bin"

        if rust_bin.exists() and python_bin.exists():
            rust_audio = load_audio_bin(rust_bin)
            python_audio = load_audio_bin(python_bin)

            plot_path = output_dir / f"comparison_seed{seed}_frames{frames}.png"
            plot_comparison(rust_audio, python_audio, 24000, plot_path,
                          f"Seed {seed}, {frames} frames")
            result["plot_path"] = str(plot_path)

            try:
                spec_path = output_dir / f"spectrogram_diff_seed{seed}_frames{frames}.png"
                plot_spectrogram_diff(rust_audio, python_audio, 24000, spec_path)
                result["spectrogram_path"] = str(spec_path)
            except ImportError:
                pass  # scipy not available

    return result


def find_all_generations(directory: Path) -> list[tuple[int, int]]:
    """Find all seed/frame combinations in a directory."""
    generations = set()
    for path in directory.glob("codes_seed*_frames*.bin"):
        name = path.stem
        try:
            parts = name.split("_")
            seed = int(parts[1].replace("seed", ""))
            frames = int(parts[2].replace("frames", ""))
            generations.add((seed, frames))
        except (IndexError, ValueError):
            continue
    return sorted(generations)


def print_summary(results: list[dict]):
    """Print comparison summary."""
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    for r in results:
        seed = r.get("seed", "?")
        frames = r.get("frames", "?")
        print(f"\n--- Seed {seed}, {frames} frames ---")

        # Codes comparison
        codes = r.get("codes", {})
        if "error" in codes:
            print(f"  Codes: {codes['error']}")
        else:
            if codes.get("match"):
                print(f"  Codes: MATCH ({codes['rust_count']} values)")
            else:
                print(f"  Codes: MISMATCH ({codes['diff_count']} differences)")
                if "first_diffs" in codes:
                    for d in codes["first_diffs"][:3]:
                        print(f"    Index {d['index']}: Rust={d['rust']}, Python={d['python']}")

        # Audio comparison
        audio = r.get("audio", {})
        if "error" in audio:
            print(f"  Audio: {audio['error']}")
        else:
            print(f"  Audio:")
            print(f"    Samples: Rust={audio['length_a']}, Python={audio['length_b']}")
            print(f"    Max diff: {audio['max_diff']:.2e}")
            print(f"    Mean diff: {audio['mean_diff']:.2e}")
            print(f"    RMSE: {audio['rmse']:.2e}")
            if audio.get("correlation") is not None:
                print(f"    Correlation: {audio['correlation']:.6f}")
            if audio.get("snr_db") is not None:
                snr = audio["snr_db"]
                if snr == float('inf'):
                    print(f"    SNR: inf (identical)")
                else:
                    print(f"    SNR: {snr:.1f} dB")

            # Status
            max_diff = audio["max_diff"]
            if max_diff < 1e-6:
                status = "IDENTICAL"
            elif max_diff < 1e-4:
                status = "MATCH (< 1e-4)"
            elif max_diff < 1e-2:
                status = "CLOSE (< 1e-2)"
            else:
                status = "DIFFERENT"
            print(f"    Status: {status}")


def main():
    parser = argparse.ArgumentParser(description="Compare Rust and Python audio outputs")
    parser.add_argument("--rust", type=str, help="Rust output directory")
    parser.add_argument("--python", type=str, help="Python reference directory")
    parser.add_argument("--file1", type=str, help="First audio file (for direct comparison)")
    parser.add_argument("--file2", type=str, help="Second audio file (for direct comparison)")
    parser.add_argument("--seed", type=int, help="Specific seed to compare")
    parser.add_argument("--frames", type=int, help="Specific frame count to compare")
    parser.add_argument("--all", action="store_true", help="Compare all matching seed/frame combinations")
    parser.add_argument("--output", type=str, help="Output directory for reports and plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    results = []

    # Direct file comparison
    if args.file1 and args.file2:
        file1 = Path(args.file1)
        file2 = Path(args.file2)

        if file1.suffix == '.bin':
            audio1 = load_audio_bin(file1)
            audio2 = load_audio_bin(file2)
        else:
            audio1, sr1 = load_audio_wav(file1)
            audio2, sr2 = load_audio_wav(file2)

        metrics = compute_metrics(audio1, audio2)
        results.append({"file1": str(file1), "file2": str(file2), "audio": metrics})

        if not args.no_plots and args.output and HAS_MATPLOTLIB:
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True, parents=True)
            plot_path = output_dir / "comparison.png"
            plot_comparison(audio1, audio2, 24000, plot_path, "Direct Comparison")

    # Directory comparison
    elif args.rust and args.python:
        rust_dir = Path(args.rust)
        python_dir = Path(args.python)
        output_dir = Path(args.output) if args.output else rust_dir / "comparison"
        output_dir.mkdir(exist_ok=True, parents=True)

        if args.all:
            # Find all common generations
            rust_gens = set(find_all_generations(rust_dir))
            python_gens = set(find_all_generations(python_dir))
            common = sorted(rust_gens & python_gens)

            if not common:
                print("No common seed/frame combinations found")
                print(f"  Rust: {rust_gens}")
                print(f"  Python: {python_gens}")
                sys.exit(1)

            for seed, frames in common:
                result = compare_single_generation(
                    rust_dir, python_dir, seed, frames,
                    output_dir, not args.no_plots
                )
                results.append(result)

        elif args.seed is not None and args.frames is not None:
            result = compare_single_generation(
                rust_dir, python_dir, args.seed, args.frames,
                output_dir, not args.no_plots
            )
            results.append(result)

        else:
            print("Specify --seed and --frames, or use --all")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)

    # Output results
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_summary(results)

    # Save JSON report
    if args.output:
        report_path = Path(args.output) / "comparison_report.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved report to: {report_path}")


if __name__ == "__main__":
    main()

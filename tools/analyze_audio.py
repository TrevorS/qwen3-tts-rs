#!/usr/bin/env python3
"""Audio analysis tools for debugging TTS output.

Usage:
    uv run python analyze_audio.py waveform audio.wav
    uv run python analyze_audio.py spectrogram audio.wav
    uv run python analyze_audio.py compare rust.wav python.wav
    uv run python analyze_audio.py stats audio.wav
    uv run python analyze_audio.py codes codes.bin --frames 25
"""

import argparse
import struct
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile


def load_wav(path: str) -> tuple[int, np.ndarray]:
    """Load a WAV file and return sample rate and samples."""
    sample_rate, samples = wavfile.read(path)
    # Normalize to float32 [-1, 1]
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    elif samples.dtype == np.int32:
        samples = samples.astype(np.float32) / 2147483648.0
    elif samples.dtype != np.float32:
        samples = samples.astype(np.float32)
    return sample_rate, samples


def load_binary_audio(path: str) -> np.ndarray:
    """Load raw f32 binary audio."""
    data = Path(path).read_bytes()
    samples = struct.unpack(f'{len(data)//4}f', data)
    return np.array(samples, dtype=np.float32)


def load_codes(path: str, num_frames: int, num_quantizers: int = 16) -> np.ndarray:
    """Load binary codes file."""
    data = Path(path).read_bytes()
    # Codes are stored as i64
    codes = struct.unpack(f'{len(data)//8}q', data)
    return np.array(codes, dtype=np.int64).reshape(num_frames, num_quantizers)


def plot_waveform(samples: np.ndarray, sample_rate: int, title: str, output: str):
    """Plot waveform."""
    duration = len(samples) / sample_rate
    time = np.linspace(0, duration, len(samples))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Full waveform
    axes[0].plot(time, samples, linewidth=0.5)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'{title} - Full Waveform')
    axes[0].set_xlim(0, duration)
    axes[0].grid(True, alpha=0.3)

    # Zoomed view (first 0.5s or 10% of duration)
    zoom_duration = min(0.5, duration * 0.1)
    zoom_samples = int(zoom_duration * sample_rate)
    axes[1].plot(time[:zoom_samples], samples[:zoom_samples], linewidth=0.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title(f'{title} - First {zoom_duration:.2f}s (Zoomed)')
    axes[1].set_xlim(0, zoom_duration)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved waveform to: {output}")
    plt.close()


def plot_spectrogram(samples: np.ndarray, sample_rate: int, title: str, output: str):
    """Plot spectrogram."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Spectrogram
    nperseg = min(1024, len(samples) // 8)
    f, t, Sxx = signal.spectrogram(samples, sample_rate, nperseg=nperseg)

    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    im = axes[0].pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_title(f'{title} - Spectrogram')
    axes[0].set_ylim(0, min(8000, sample_rate // 2))  # Focus on speech frequencies
    plt.colorbar(im, ax=axes[0], label='Power (dB)')

    # Mel-like spectrogram (log frequency scale)
    im2 = axes[1].pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title(f'{title} - Spectrogram (Log Scale)')
    axes[1].set_yscale('symlog', linthresh=100)
    axes[1].set_ylim(50, min(8000, sample_rate // 2))
    plt.colorbar(im2, ax=axes[1], label='Power (dB)')

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved spectrogram to: {output}")
    plt.close()


def analyze_stats(samples: np.ndarray, sample_rate: int, name: str):
    """Print audio statistics."""
    duration = len(samples) / sample_rate

    print(f"\n=== Audio Statistics: {name} ===")
    print(f"Duration: {duration:.3f}s")
    print(f"Samples: {len(samples)}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Min: {samples.min():.6f}")
    print(f"Max: {samples.max():.6f}")
    print(f"Mean: {samples.mean():.6f}")
    print(f"Std: {samples.std():.6f}")
    print(f"RMS: {np.sqrt(np.mean(samples**2)):.6f}")

    # Check for issues
    zero_crossings = np.sum(np.diff(np.signbit(samples)))
    zcr = zero_crossings / len(samples) * sample_rate
    print(f"Zero crossing rate: {zcr:.1f} Hz")

    # Check for clipping
    clipped = np.sum(np.abs(samples) > 0.99)
    print(f"Clipped samples: {clipped} ({100*clipped/len(samples):.2f}%)")

    # Check for silence
    silent = np.sum(np.abs(samples) < 0.001)
    print(f"Silent samples: {silent} ({100*silent/len(samples):.2f}%)")

    # Frequency analysis
    fft = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(len(samples), 1/sample_rate)
    magnitude = np.abs(fft)

    # Find dominant frequencies
    top_indices = np.argsort(magnitude)[-10:][::-1]
    print("\nDominant frequencies:")
    for i, idx in enumerate(top_indices[:5]):
        print(f"  {i+1}. {freqs[idx]:.1f} Hz (magnitude: {magnitude[idx]:.2f})")

    # Check energy distribution
    low_energy = np.sum(magnitude[freqs < 300]**2)
    mid_energy = np.sum(magnitude[(freqs >= 300) & (freqs < 3000)]**2)
    high_energy = np.sum(magnitude[freqs >= 3000]**2)
    total_energy = low_energy + mid_energy + high_energy

    print(f"\nEnergy distribution:")
    print(f"  Low (<300 Hz): {100*low_energy/total_energy:.1f}%")
    print(f"  Mid (300-3000 Hz): {100*mid_energy/total_energy:.1f}%")
    print(f"  High (>3000 Hz): {100*high_energy/total_energy:.1f}%")


def compare_audio(samples1: np.ndarray, samples2: np.ndarray,
                  sr1: int, sr2: int, name1: str, name2: str, output: str):
    """Compare two audio files."""
    if sr1 != sr2:
        print(f"Warning: Different sample rates ({sr1} vs {sr2})")

    min_len = min(len(samples1), len(samples2))
    s1 = samples1[:min_len]
    s2 = samples2[:min_len]

    diff = s1 - s2

    print(f"\n=== Comparison: {name1} vs {name2} ===")
    print(f"Length: {len(samples1)} vs {len(samples2)} samples")
    print(f"Max difference: {np.abs(diff).max():.6f}")
    print(f"Mean difference: {np.mean(np.abs(diff)):.6f}")
    print(f"RMSE: {np.sqrt(np.mean(diff**2)):.6f}")

    # Correlation
    corr = np.corrcoef(s1, s2)[0, 1]
    print(f"Correlation: {corr:.6f}")

    # Plot comparison
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    duration = min_len / sr1
    time = np.linspace(0, duration, min_len)

    # Waveforms
    axes[0].plot(time, s1, label=name1, alpha=0.7, linewidth=0.5)
    axes[0].plot(time, s2, label=name2, alpha=0.7, linewidth=0.5)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Waveform Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Difference
    axes[1].plot(time, diff, color='red', linewidth=0.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Difference')
    axes[1].set_title('Sample-by-Sample Difference')
    axes[1].grid(True, alpha=0.3)

    # Spectrograms side by side
    nperseg = min(1024, min_len // 8)

    f1, t1, Sxx1 = signal.spectrogram(s1, sr1, nperseg=nperseg)
    Sxx1_db = 10 * np.log10(Sxx1 + 1e-10)
    axes[2].pcolormesh(t1, f1, Sxx1_db, shading='gouraud', cmap='viridis')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title(f'{name1} Spectrogram')
    axes[2].set_ylim(0, min(8000, sr1 // 2))

    f2, t2, Sxx2 = signal.spectrogram(s2, sr2, nperseg=nperseg)
    Sxx2_db = 10 * np.log10(Sxx2 + 1e-10)
    axes[3].pcolormesh(t2, f2, Sxx2_db, shading='gouraud', cmap='viridis')
    axes[3].set_ylabel('Frequency (Hz)')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_title(f'{name2} Spectrogram')
    axes[3].set_ylim(0, min(8000, sr2 // 2))

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved comparison to: {output}")
    plt.close()


def analyze_codes(codes: np.ndarray, output: str):
    """Analyze and visualize codes."""
    num_frames, num_quantizers = codes.shape

    print(f"\n=== Codes Analysis ===")
    print(f"Shape: {codes.shape} (frames x quantizers)")
    print(f"Semantic tokens (q0): min={codes[:, 0].min()}, max={codes[:, 0].max()}")
    print(f"Acoustic tokens (q1-15): min={codes[:, 1:].min()}, max={codes[:, 1:].max()}")

    # Check for patterns
    print("\nSemantic token sequence (first 20):")
    print(codes[:min(20, num_frames), 0].tolist())

    # Check for repetition
    semantic = codes[:, 0]
    unique_semantic = len(np.unique(semantic))
    print(f"\nUnique semantic tokens: {unique_semantic}/{num_frames}")

    # Check acoustic token distribution
    print("\nAcoustic token stats per quantizer:")
    for q in range(1, min(5, num_quantizers)):
        col = codes[:, q]
        print(f"  Q{q}: unique={len(np.unique(col))}, range=[{col.min()}, {col.max()}]")

    # Plot codes as heatmap
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # All quantizers
    im1 = axes[0].imshow(codes.T, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Quantizer')
    axes[0].set_title('All Codes (16 quantizers x frames)')
    plt.colorbar(im1, ax=axes[0], label='Token ID')

    # Just semantic tokens over time
    axes[1].plot(codes[:, 0], 'b-', linewidth=1, label='Semantic (Q0)')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Token ID')
    axes[1].set_title('Semantic Token Sequence')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved codes analysis to: {output}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Audio analysis tools")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Waveform
    wave_parser = subparsers.add_parser('waveform', help='Plot waveform')
    wave_parser.add_argument('audio', help='Audio file (WAV or binary)')
    wave_parser.add_argument('-o', '--output', default='waveform.png')
    wave_parser.add_argument('--binary', action='store_true', help='Load as binary f32')
    wave_parser.add_argument('--sr', type=int, default=24000, help='Sample rate for binary')

    # Spectrogram
    spec_parser = subparsers.add_parser('spectrogram', help='Plot spectrogram')
    spec_parser.add_argument('audio', help='Audio file')
    spec_parser.add_argument('-o', '--output', default='spectrogram.png')
    spec_parser.add_argument('--binary', action='store_true')
    spec_parser.add_argument('--sr', type=int, default=24000)

    # Stats
    stats_parser = subparsers.add_parser('stats', help='Print audio statistics')
    stats_parser.add_argument('audio', help='Audio file')
    stats_parser.add_argument('--binary', action='store_true')
    stats_parser.add_argument('--sr', type=int, default=24000)

    # Compare
    cmp_parser = subparsers.add_parser('compare', help='Compare two audio files')
    cmp_parser.add_argument('audio1', help='First audio file')
    cmp_parser.add_argument('audio2', help='Second audio file')
    cmp_parser.add_argument('-o', '--output', default='comparison.png')

    # Codes
    codes_parser = subparsers.add_parser('codes', help='Analyze codes')
    codes_parser.add_argument('codes_file', help='Binary codes file')
    codes_parser.add_argument('--frames', type=int, required=True, help='Number of frames')
    codes_parser.add_argument('-o', '--output', default='codes_analysis.png')

    args = parser.parse_args()

    if args.command == 'waveform':
        if args.binary:
            samples = load_binary_audio(args.audio)
            sr = args.sr
        else:
            sr, samples = load_wav(args.audio)
        plot_waveform(samples, sr, Path(args.audio).stem, args.output)

    elif args.command == 'spectrogram':
        if args.binary:
            samples = load_binary_audio(args.audio)
            sr = args.sr
        else:
            sr, samples = load_wav(args.audio)
        plot_spectrogram(samples, sr, Path(args.audio).stem, args.output)

    elif args.command == 'stats':
        if args.binary:
            samples = load_binary_audio(args.audio)
            sr = args.sr
        else:
            sr, samples = load_wav(args.audio)
        analyze_stats(samples, sr, args.audio)

    elif args.command == 'compare':
        sr1, samples1 = load_wav(args.audio1)
        sr2, samples2 = load_wav(args.audio2)
        compare_audio(samples1, samples2, sr1, sr2,
                     Path(args.audio1).stem, Path(args.audio2).stem, args.output)

    elif args.command == 'codes':
        codes = load_codes(args.codes_file, args.frames)
        analyze_codes(codes, args.output)


if __name__ == '__main__':
    main()

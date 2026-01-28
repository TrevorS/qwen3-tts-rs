#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "matplotlib",
#   "scipy",
# ]
# ///
"""
Run all model variant + device combinations, generate test WAV files,
analyze audio quality, and produce a self-contained HTML report with
spectrograms, waveforms, and stats.

Usage:
    ./scripts/test-variants.py                      # auto-detect, run all
    ./scripts/test-variants.py --device cuda         # CUDA only
    ./scripts/test-variants.py --batch 5             # 5 seeds (42..46)
    ./scripts/test-variants.py --random --batch 3    # 3 random-seeded runs
    ./scripts/test-variants.py --build --serve       # build first, serve after
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram as scipy_spectrogram


# ── Constants ───────────────────────────────────────────────────────────

SAMPLE_RATE = 24000
SILENCE_THRESHOLD_DB = -40.0

# Plot style
PLT_BG = "#1a1a2e"
PLT_FG = "#e0e0e0"
PLT_ACCENT = "#4caf50"
PLT_CMAP = "magma"


# ── Data classes ────────────────────────────────────────────────────────

@dataclass
class Model:
    name: str
    path: Path
    model_type: str  # "base", "customvoice", "voicedesign"


@dataclass
class TestCase:
    label: str
    model: Model
    extra_args: list[str]


@dataclass
class AudioStats:
    duration_s: float = 0.0
    num_samples: int = 0
    sample_rate: int = SAMPLE_RATE
    rms_db: float = -100.0
    peak_db: float = -100.0
    silence_pct: float = 100.0
    crest_factor_db: float = 0.0


@dataclass
class TestResult:
    label: str
    device: str
    seed: int
    status: str  # "PASS" or "FAIL"
    elapsed_s: float = 0.0
    file_size: str = "-"
    wav_path: Path | None = None
    stats: AudioStats = field(default_factory=AudioStats)
    spectrogram_b64: str = ""
    waveform_b64: str = ""


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run TTS variant tests with audio analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--device", action="append", dest="devices", default=[],
                    help="Device(s) to test. Repeat for multiple. Default: auto-detect.")
    p.add_argument("--build", action="store_true",
                    help="Build release binary before testing.")
    p.add_argument("--serve", action="store_true",
                    help="Start HTTP server after tests.")
    p.add_argument("--random", action="store_true",
                    help="Use a random seed.")
    p.add_argument("--batch", type=int, default=1,
                    help="Run each test N times with sequential seeds (default: 1).")
    p.add_argument("--seed", type=int, default=42,
                    help="Base seed (default: 42). Ignored with --random.")
    p.add_argument("--text", default="Hello world, this is a test.",
                    help="Text to synthesize.")
    p.add_argument("--instruct",
                    default="A cheerful young female voice with clear pronunciation and natural intonation.",
                    help="Voice description for VoiceDesign models.")
    p.add_argument("--duration", type=float, default=3.0,
                    help="Duration in seconds (default: 3.0).")
    p.add_argument("--hostname", default=os.environ.get("HOSTNAME", "localhost"),
                    help="Hostname for HTTP URLs.")
    p.add_argument("--port", type=int, default=8765,
                    help="HTTP server port (default: 8765).")
    return p.parse_args()


# ── Paths ───────────────────────────────────────────────────────────────

def find_paths() -> tuple[Path, Path, Path]:
    """Return (repo_root, project_root, binary_path)."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    project_root = repo_root.parent
    binary = repo_root / "target" / "release" / "generate_audio"
    return repo_root, project_root, binary


# ── Build ───────────────────────────────────────────────────────────────

def build_binary(repo_root: Path) -> None:
    print("Building release binary...")
    cargo_toml = repo_root / "Cargo.toml"

    # Try flash-attn first, fall back to cuda, then cpu
    for features, desc in [
        ("flash-attn,cli", "bf16 + Flash Attention 2"),
        ("cuda,cli", "standard CUDA"),
        ("cli", "CPU only"),
    ]:
        result = subprocess.run(
            ["cargo", "build", "--release", "--features", features,
             "--manifest-path", str(cargo_toml)],
            capture_output=True,
        )
        if result.returncode == 0:
            print(f"Built with: {features} ({desc})")
            return

    print("ERROR: cargo build failed", file=sys.stderr)
    sys.exit(1)


# ── Model discovery ────────────────────────────────────────────────────

def discover_models(repo_root: Path) -> list[Model]:
    models_dir = repo_root / "test_data" / "models"
    models = []

    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if not (model_dir / "model.safetensors").exists():
            continue

        name = model_dir.name
        model_type = detect_model_type(model_dir, name)
        models.append(Model(name=name, path=model_dir, model_type=model_type))

    return models


def detect_model_type(model_dir: Path, name: str) -> str:
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            tts_type = cfg.get("tts_model_type", "")
            if tts_type == "voice_design":
                return "voicedesign"
            elif tts_type == "custom_voice":
                return "customvoice"
            elif "speaker_encoder" in cfg:
                return "base"
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback heuristics
    if "base" in name:
        return "base"
    elif "voicedesign" in name or "voice-design" in name:
        return "voicedesign"
    return "customvoice"


# ── Device detection ────────────────────────────────────────────────────

def detect_devices(binary: Path) -> list[str]:
    devices = ["cpu"]
    # Try running with --device cuda to see if it's supported
    try:
        result = subprocess.run(
            [str(binary), "--device", "cuda", "--text", "x", "--duration", "0.1",
             "--model-dir", "/nonexistent"],
            capture_output=True, text=True, timeout=5,
        )
        if "CUDA" in result.stderr or "cuda" in result.stderr.lower():
            devices.append("cuda")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return devices


# ── Test matrix ─────────────────────────────────────────────────────────

def build_test_matrix(
    models: list[Model],
    repo_root: Path,
    instruct_text: str,
) -> list[TestCase]:
    ref_audio = repo_root / "examples" / "data" / "apollo11_one_small_step.wav"
    ref_text = "That's one small step for man, one giant leap for mankind."
    tests: list[TestCase] = []

    for m in models:
        if m.model_type == "base":
            if ref_audio.exists():
                tests.append(TestCase(
                    label=f"{m.name}-xvector", model=m,
                    extra_args=["--model-dir", str(m.path),
                                "--ref-audio", str(ref_audio), "--x-vector-only"],
                ))
                tests.append(TestCase(
                    label=f"{m.name}-icl", model=m,
                    extra_args=["--model-dir", str(m.path),
                                "--ref-audio", str(ref_audio),
                                "--ref-text", ref_text],
                ))
            else:
                print(f"WARN: Skipping {m.name} (no reference audio: {ref_audio})")

        elif m.model_type == "voicedesign":
            tests.append(TestCase(
                label=f"{m.name}-instruct", model=m,
                extra_args=["--model-dir", str(m.path),
                            "--instruct", instruct_text],
            ))

        else:  # customvoice
            tok_args = []
            if (m.path / "tokenizer.json").exists():
                tok_args = ["--tokenizer-dir", str(m.path)]
            for speaker in ["ryan", "serena"]:
                tests.append(TestCase(
                    label=f"{m.name}-{speaker}", model=m,
                    extra_args=["--model-dir", str(m.path)] + tok_args +
                               ["--speaker", speaker],
                ))

    return tests


# ── Seeds ───────────────────────────────────────────────────────────────

def resolve_seeds(base_seed: int, batch: int, random: bool) -> list[int]:
    if random:
        base_seed = int.from_bytes(os.urandom(4), "little")
        print(f"Random seed: {base_seed}")
    seeds = [base_seed + i for i in range(batch)]
    if batch > 1:
        print(f"Batch mode: {batch} seeds ({seeds[0]}..{seeds[-1]})")
    return seeds


# ── Test execution ──────────────────────────────────────────────────────

def run_test(
    binary: Path,
    test: TestCase,
    device: str,
    seed: int,
    text: str,
    duration: float,
    output_path: Path,
) -> TestResult:
    cmd = [
        str(binary),
        "--device", device,
        *test.extra_args,
        "--text", text,
        "--duration", str(duration),
        "--seed", str(seed),
        "--output", str(output_path),
    ]

    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0

    status = "PASS" if result.returncode == 0 else "FAIL"

    file_size = "-"
    if output_path.exists():
        size_bytes = output_path.stat().st_size
        if size_bytes >= 1024 * 1024:
            file_size = f"{size_bytes / (1024*1024):.1f}M"
        else:
            file_size = f"{size_bytes // 1024}K"

    return TestResult(
        label=test.label,
        device=device,
        seed=seed,
        status=status,
        elapsed_s=round(elapsed, 1),
        file_size=file_size,
        wav_path=output_path if output_path.exists() else None,
    )


def run_all_tests(
    binary: Path,
    tests: list[TestCase],
    devices: list[str],
    seeds: list[int],
    text: str,
    duration: float,
    output_base: Path,
) -> list[TestResult]:
    batch = len(seeds)
    total = len(tests) * len(devices) * batch
    results: list[TestResult] = []
    run_num = 0

    for device in devices:
        device_dir = output_base / device
        device_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            for test in tests:
                run_num += 1
                label = f"{test.label}-s{seed}" if batch > 1 else test.label
                outfile = device_dir / f"{label}.wav"

                print(f"[{run_num}/{total}] {device:<12s} {label:<40s} ", end="", flush=True)

                r = run_test(binary, test, device, seed, text, duration, outfile)
                # Override label for batch mode
                r.label = label
                results.append(r)

                if r.status == "PASS":
                    print(f"{r.status:<6s} {r.elapsed_s:>5.1f}s  {r.file_size}")
                else:
                    print(f"{r.status:<6s} {r.elapsed_s:>5.1f}s  (failed)")

    return results


# ── Audio analysis ──────────────────────────────────────────────────────

def analyze_audio(results: list[TestResult]) -> None:
    """Compute stats + generate visualizations for each passed result."""
    passed = [r for r in results if r.status == "PASS" and r.wav_path]
    if not passed:
        return

    print(f"\nAnalyzing {len(passed)} audio files...")
    for i, r in enumerate(passed):
        print(f"  [{i+1}/{len(passed)}] {r.label}...", end="", flush=True)
        try:
            sr, data = wavfile.read(str(r.wav_path))
            samples = _normalize_samples(data)
            r.stats = compute_stats(samples, sr)
            r.spectrogram_b64 = render_spectrogram(samples, sr)
            r.waveform_b64 = render_waveform(samples, sr)
            print(" done")
        except Exception as e:
            print(f" error: {e}")


def _normalize_samples(data: np.ndarray) -> np.ndarray:
    """Convert WAV data to float32 in [-1, 1]."""
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        return data
    elif data.dtype == np.float64:
        return data.astype(np.float32)
    return data.astype(np.float32)


def compute_stats(samples: np.ndarray, sr: int) -> AudioStats:
    n = len(samples)
    if n == 0:
        return AudioStats()

    duration = n / sr
    rms = np.sqrt(np.mean(samples ** 2))
    peak = np.max(np.abs(samples))

    eps = 1e-10
    rms_db = 20 * np.log10(rms + eps)
    peak_db = 20 * np.log10(peak + eps)
    crest_db = peak_db - rms_db

    # Silence: frames below threshold (using 1024-sample frames)
    frame_size = 1024
    n_frames = n // frame_size
    if n_frames > 0:
        frames = samples[:n_frames * frame_size].reshape(n_frames, frame_size)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        frame_db = 20 * np.log10(frame_rms + eps)
        silence_pct = 100.0 * np.mean(frame_db < SILENCE_THRESHOLD_DB)
    else:
        silence_pct = 0.0

    return AudioStats(
        duration_s=round(duration, 2),
        num_samples=n,
        sample_rate=sr,
        rms_db=round(float(rms_db), 1),
        peak_db=round(float(peak_db), 1),
        silence_pct=round(float(silence_pct), 1),
        crest_factor_db=round(float(crest_db), 1),
    )


# ── Visualization ───────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=PLT_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def render_spectrogram(samples: np.ndarray, sr: int) -> str:
    fig, ax = plt.subplots(figsize=(8, 2.2))
    fig.patch.set_facecolor(PLT_BG)
    ax.set_facecolor(PLT_BG)

    nperseg = min(1024, len(samples))
    noverlap = nperseg * 3 // 4
    f, t, Sxx = scipy_spectrogram(samples, fs=sr, nperseg=nperseg,
                                   noverlap=noverlap, window="hann")

    # Limit to 0-10kHz
    freq_mask = f <= 10000
    Sxx_db = 10 * np.log10(Sxx[freq_mask] + 1e-10)

    ax.pcolormesh(t, f[freq_mask], Sxx_db, shading="gouraud",
                  cmap=PLT_CMAP, vmin=-80, vmax=0)
    ax.set_ylabel("Hz", color=PLT_FG, fontsize=9)
    ax.set_xlabel("Time (s)", color=PLT_FG, fontsize=9)
    ax.tick_params(colors=PLT_FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333")

    return _fig_to_b64(fig)


def render_waveform(samples: np.ndarray, sr: int) -> str:
    fig, ax = plt.subplots(figsize=(8, 1.4))
    fig.patch.set_facecolor(PLT_BG)
    ax.set_facecolor(PLT_BG)

    # Downsample for plotting if too many samples
    if len(samples) > 4000:
        stride = len(samples) // 4000
        plot_samples = samples[::stride]
    else:
        plot_samples = samples

    t = np.linspace(0, len(samples) / sr, len(plot_samples))
    ax.fill_between(t, plot_samples, alpha=0.4, color=PLT_ACCENT, linewidth=0)
    ax.plot(t, plot_samples, color=PLT_ACCENT, linewidth=0.4, alpha=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Amp", color=PLT_FG, fontsize=9)
    ax.set_xlabel("Time (s)", color=PLT_FG, fontsize=9)
    ax.tick_params(colors=PLT_FG, labelsize=8)
    ax.axhline(0, color="#333", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color("#333")

    return _fig_to_b64(fig)


# ── HTML generation ─────────────────────────────────────────────────────

def generate_html(
    results: list[TestResult],
    output_path: Path,
    text: str,
    seeds: list[int],
    devices: list[str],
) -> None:
    batch = len(seeds)
    if batch > 1:
        seed_info = f"Seeds: {seeds[0]}&ndash;{seeds[-1]} ({batch} runs)"
    else:
        seed_info = f"Seed: {seeds[0]}"

    pass_count = sum(1 for r in results if r.status == "PASS")
    fail_count = sum(1 for r in results if r.status == "FAIL")

    html_parts = [HTML_HEAD]
    html_parts.append(f'<h1>TTS Variant Test Results</h1>')
    html_parts.append(
        f'<p class="meta">Text: &quot;{text}&quot; &mdash; '
        f'{seed_info} &mdash; '
        f'{pass_count} passed, {fail_count} failed</p>'
    )

    # Summary table
    html_parts.append(render_summary_table(results, devices))

    # Group results
    if batch > 1:
        # Group by seed
        for seed in seeds:
            seed_results = [r for r in results if r.seed == seed]
            html_parts.append(f'<h2>Seed {seed}</h2>')
            html_parts.append('<div class="card-grid">')
            for r in seed_results:
                html_parts.append(render_card(r))
            html_parts.append('</div>')
    else:
        # Group by device
        for device in devices:
            dev_results = [r for r in results if r.device == device]
            html_parts.append(f'<h2>{device}</h2>')
            html_parts.append('<div class="card-grid">')
            for r in dev_results:
                html_parts.append(render_card(r))
            html_parts.append('</div>')

    html_parts.append("</body></html>")

    output_path.write_text("\n".join(html_parts))
    print(f"HTML: {output_path}")


def render_summary_table(results: list[TestResult], devices: list[str]) -> str:
    rows = ['<h2>Summary</h2>', '<table>', '<tr>']
    rows.append('<th>Test</th><th>Seed</th><th>Device</th>'
                '<th>Status</th><th>Time</th><th>Size</th>'
                '<th>Duration</th><th>RMS</th><th>Peak</th><th>Silence</th>')
    rows.append('</tr>')

    for r in results:
        status_cls = "pass" if r.status == "PASS" else "fail"
        s = r.stats
        rows.append(
            f'<tr>'
            f'<td>{r.label}</td>'
            f'<td>{r.seed}</td>'
            f'<td>{r.device}</td>'
            f'<td class="{status_cls}">{r.status}</td>'
            f'<td>{r.elapsed_s}s</td>'
            f'<td>{r.file_size}</td>'
            f'<td>{s.duration_s}s</td>'
            f'<td>{s.rms_db}dB</td>'
            f'<td>{s.peak_db}dB</td>'
            f'<td>{s.silence_pct}%</td>'
            f'</tr>'
        )

    rows.append('</table>')
    return "\n".join(rows)


def render_card(r: TestResult) -> str:
    if r.status != "PASS" or not r.wav_path:
        return (
            f'<div class="card">'
            f'  <div class="card-header">'
            f'    <h3>{r.label}</h3>'
            f'    <span class="fail">FAIL</span>'
            f'  </div>'
            f'</div>'
        )

    s = r.stats
    relpath = r.wav_path.name  # relative to device dir
    device_rel = f"{r.device}/{relpath}"

    parts = [
        f'<div class="card">',
        f'  <div class="card-header">',
        f'    <h3>{r.label}</h3>',
        f'    <span class="badge">seed {r.seed} &middot; {r.device} &middot; {r.elapsed_s}s &middot; {r.file_size}</span>',
        f'  </div>',
        f'  <audio controls preload="metadata" src="{device_rel}"></audio>',
    ]

    if r.spectrogram_b64:
        parts.append(f'  <img class="viz" src="data:image/png;base64,{r.spectrogram_b64}" alt="spectrogram">')
    if r.waveform_b64:
        parts.append(f'  <img class="viz" src="data:image/png;base64,{r.waveform_b64}" alt="waveform">')

    parts.append(
        f'  <div class="stats">'
        f'    <span>Duration: {s.duration_s}s</span>'
        f'    <span>RMS: {s.rms_db}dB</span>'
        f'    <span>Peak: {s.peak_db}dB</span>'
        f'    <span>Silence: {s.silence_pct}%</span>'
        f'    <span>Crest: {s.crest_factor_db}dB</span>'
        f'  </div>'
    )
    parts.append('</div>')
    return "\n".join(parts)


HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TTS Variant Test Results</title>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: system-ui, -apple-system, sans-serif;
    max-width: 1400px; margin: 0 auto; padding: 20px;
    background: #1a1a2e; color: #e0e0e0;
  }
  h1 { color: #fff; border-bottom: 2px solid #444; padding-bottom: 10px; }
  h2 { color: #ccc; margin-top: 36px; margin-bottom: 12px; }
  .meta { color: #888; font-size: 14px; margin-bottom: 20px; }

  /* Summary table */
  table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }
  th, td { border: 1px solid #333; padding: 6px 10px; text-align: left; white-space: nowrap; }
  th { background: #2a2a4a; color: #fff; position: sticky; top: 0; }
  tr:nth-child(even) { background: #1e1e3a; }
  tr:nth-child(odd) { background: #222244; }
  .pass { color: #4caf50; font-weight: bold; }
  .fail { color: #f44336; font-weight: bold; }

  /* Card grid */
  .card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
    gap: 16px;
  }
  .card {
    background: #222244; border: 1px solid #333; border-radius: 8px;
    padding: 16px; overflow: hidden;
  }
  .card-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
  .card-header h3 { margin: 0; color: #fff; font-size: 15px; }
  .badge { color: #888; font-size: 12px; }
  .card audio { width: 100%; margin: 8px 0; }
  .card .viz { width: 100%; border-radius: 4px; margin: 4px 0; }
  .stats {
    display: flex; flex-wrap: wrap; gap: 12px;
    color: #aaa; font-size: 12px; font-family: monospace;
    margin-top: 8px; padding-top: 8px; border-top: 1px solid #333;
  }
</style>
</head>
<body>"""


# ── JSON output ─────────────────────────────────────────────────────────

def generate_json(results: list[TestResult], output_path: Path, output_base: Path) -> None:
    records = []
    for r in results:
        rec = {
            "label": r.label,
            "device": r.device,
            "seed": r.seed,
            "status": r.status,
            "time": r.elapsed_s,
            "size": r.file_size,
            "file": str(r.wav_path.relative_to(output_base)) if r.wav_path else None,
        }
        rec.update(asdict(r.stats))
        records.append(rec)

    output_path.write_text(json.dumps(records, indent=2))
    print(f"JSON: {output_path}")


# ── HTTP server ─────────────────────────────────────────────────────────

def serve_results(output_base: Path, hostname: str, port: int) -> None:
    import http.server
    import threading

    os.chdir(output_base)

    handler = http.server.SimpleHTTPRequestHandler
    server = http.server.HTTPServer(("", port), handler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://{hostname}:{port}/"
    print(f"\n  {url}\n")
    print("Press Ctrl+C to stop.")

    try:
        thread.join()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.shutdown()


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    repo_root, project_root, binary = find_paths()
    output_base = project_root / "test_data" / "variant_tests"
    output_base.mkdir(parents=True, exist_ok=True)

    # Build
    if args.build:
        build_binary(repo_root)
        print()

    # Verify binary
    if not binary.exists():
        print(f"ERROR: Binary not found: {binary}", file=sys.stderr)
        print(f"Run with --build or: cargo build --release --features cli "
              f"--manifest-path {repo_root / 'Cargo.toml'}")
        sys.exit(1)

    # Seeds
    seeds = resolve_seeds(args.seed, args.batch, args.random)

    # Devices
    devices = args.devices or detect_devices(binary)
    print(f"Devices: {', '.join(devices)}")

    # Models
    models = discover_models(repo_root)
    if not models:
        print(f"ERROR: No models found in {repo_root / 'test_data' / 'models'}")
        sys.exit(1)
    print(f"Models: {', '.join(m.name for m in models)}")

    # Test matrix
    tests = build_test_matrix(models, repo_root, args.instruct)
    total = len(tests) * len(devices) * len(seeds)
    print(f"Test matrix: {len(tests)} tests × {len(devices)} devices × {len(seeds)} seeds = {total} runs\n")

    # Run
    results = run_all_tests(binary, tests, devices, seeds, args.text,
                            args.duration, output_base)

    # Analyze
    analyze_audio(results)

    # Summary
    pass_count = sum(1 for r in results if r.status == "PASS")
    fail_count = len(results) - pass_count
    print(f"\nResults: {pass_count} passed, {fail_count} failed out of {len(results)} total")

    # Output
    generate_json(results, output_base / "results.json", output_base)
    generate_html(results, output_base / "index.html", args.text, seeds, devices)

    # Serve
    if args.serve:
        serve_results(output_base, args.hostname, args.port)
    elif fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

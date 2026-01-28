#!/usr/bin/env bash
#
# Run all model variant + device combinations and report results.
#
# Usage:
#   ./scripts/test-variants.sh                  # auto-detect devices, run all
#   ./scripts/test-variants.sh --device cpu      # CPU only
#   ./scripts/test-variants.sh --device cuda     # CUDA only
#   ./scripts/test-variants.sh --serve           # start HTTP server after tests
#   ./scripts/test-variants.sh --build           # build release binary first
#   ./scripts/test-variants.sh --hostname mymachine  # override hostname for URLs
#   ./scripts/test-variants.sh --random          # use a random seed
#   ./scripts/test-variants.sh --batch 5         # run each test 5x with seeds 42..46
#   ./scripts/test-variants.sh --random --batch 3  # 3 runs starting from a random seed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

BIN="$REPO_ROOT/target/release/generate_audio"
MODELS_DIR="$REPO_ROOT/test_data/models"
REF_AUDIO="$REPO_ROOT/examples/data/apollo11_one_small_step.wav"
REF_TEXT="That's one small step for man, one giant leap for mankind."
TEXT="Hello world, this is a test."
INSTRUCT="A cheerful young female voice with clear pronunciation and natural intonation."
OUTPUT_BASE="$PROJECT_ROOT/test_data/variant_tests"
SEED=42
DURATION=3.0

# Parse arguments
DEVICES=()
SERVE=false
BUILD=false
RANDOM_SEED=false
BATCH_COUNT=1
HOSTNAME="${HOSTNAME:-$(hostname)}"
HTTP_PORT=8765

while [[ $# -gt 0 ]]; do
    case $1 in
        --device)   DEVICES+=("$2"); shift 2 ;;
        --serve)    SERVE=true; shift ;;
        --build)    BUILD=true; shift ;;
        --random)   RANDOM_SEED=true; shift ;;
        --batch)    BATCH_COUNT="$2"; shift 2 ;;
        --hostname) HOSTNAME="$2"; shift 2 ;;
        --port)     HTTP_PORT="$2"; shift 2 ;;
        --text)     TEXT="$2"; shift 2 ;;
        --seed)     SEED="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--device cpu|cuda|metal] [--serve] [--build] [--hostname HOST]"
            echo ""
            echo "Options:"
            echo "  --device DEV    Test specific device(s). Repeat for multiple. Default: auto-detect."
            echo "  --serve         Start HTTP server after tests complete."
            echo "  --build         Build release binary before testing."
            echo "  --random        Use a random seed instead of the default (42)."
            echo "  --batch N       Run each test N times with seeds SEED, SEED+1, ..., SEED+N-1."
            echo "  --hostname H    Hostname for HTTP URLs (default: \$HOSTNAME)."
            echo "  --port P        HTTP server port (default: 8765)."
            echo "  --text TEXT     Text to synthesize (default: \"Hello world, this is a test.\")."
            echo "  --seed N        Base seed (default: 42). Combined with --random, this is ignored."
            echo "  --duration S    Duration in seconds (default: 3.0)."
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Resolve seed ────────────────────────────────────────────────────
if $RANDOM_SEED; then
    SEED=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')
    echo "Random seed: $SEED"
fi

# Build seeds array: [SEED, SEED+1, ..., SEED+BATCH_COUNT-1]
SEEDS=()
for ((b=0; b<BATCH_COUNT; b++)); do
    SEEDS+=( $((SEED + b)) )
done
if [[ $BATCH_COUNT -gt 1 ]]; then
    echo "Batch mode: $BATCH_COUNT seeds (${SEEDS[0]}..${SEEDS[-1]})"
fi

# ── Build ─────────────────────────────────────────────────────────────
if $BUILD; then
    echo "Building release binary..."
    # Detect best feature set: flash-attn > cuda > cpu-only
    if command -v nvcc &>/dev/null || [[ -x /usr/local/cuda/bin/nvcc ]]; then
        export PATH="/usr/local/cuda/bin:$PATH"
        # Try flash-attn first (bf16 + Flash Attention 2), fall back to cuda-only
        if cargo build --release --features "flash-attn,cli" --manifest-path "$REPO_ROOT/Cargo.toml" 2>/dev/null; then
            FEATURES="flash-attn,cli"
            echo "Built with: flash-attn,cli (bf16 + Flash Attention 2)"
        else
            FEATURES="cuda,cli"
            cargo build --release --features "$FEATURES" --manifest-path "$REPO_ROOT/Cargo.toml"
            echo "Built with: cuda,cli (flash-attn build failed, using standard CUDA)"
        fi
    else
        FEATURES="cli"
        cargo build --release --features "$FEATURES" --manifest-path "$REPO_ROOT/Cargo.toml"
        echo "Built with: cli (CPU only — no CUDA toolkit found)"
    fi
    echo ""
fi

# ── Verify binary ────────────────────────────────────────────────────
if [[ ! -x "$BIN" ]]; then
    echo "ERROR: Binary not found: $BIN"
    echo "Run with --build or: cargo build --release --features cli --manifest-path $REPO_ROOT/Cargo.toml"
    exit 1
fi

# ── Auto-detect devices ──────────────────────────────────────────────
if [[ ${#DEVICES[@]} -eq 0 ]]; then
    DEVICES=(cpu)
    # Check if binary was compiled with CUDA support
    if $BIN --device cuda --help &>/dev/null 2>&1; then
        # Check if CUDA device is actually available
        if $BIN --device cuda --text "x" --duration 0.1 --model-dir /nonexistent 2>&1 | grep -q "CUDA"; then
            DEVICES+=(cuda)
        fi
    fi
    echo "Auto-detected devices: ${DEVICES[*]}"
fi

# ── Discover models ──────────────────────────────────────────────────
declare -a MODEL_NAMES=()
declare -A MODEL_PATHS=()
declare -A MODEL_TYPES=()  # "base", "customvoice", or "voicedesign"

for model_dir in "$MODELS_DIR"/*/; do
    [[ -d "$model_dir" ]] || continue
    name="$(basename "$model_dir")"

    # Skip if no model.safetensors
    [[ -f "$model_dir/model.safetensors" ]] || continue

    MODEL_PATHS["$name"]="$model_dir"

    # Detect type from config.json tts_model_type field, then fallback heuristics
    if [[ -f "$model_dir/config.json" ]]; then
        tts_type=$(python3 -c "import json; print(json.load(open('$model_dir/config.json')).get('tts_model_type',''))" 2>/dev/null || echo "")
        case "$tts_type" in
            voice_design)
                MODEL_TYPES["$name"]="voicedesign"
                ;;
            custom_voice)
                MODEL_TYPES["$name"]="customvoice"
                ;;
            *)
                # Fallback: base models have speaker_encoder
                if grep -q 'speaker_encoder' "$model_dir/config.json" 2>/dev/null; then
                    MODEL_TYPES["$name"]="base"
                elif [[ "$name" == *base* ]]; then
                    MODEL_TYPES["$name"]="base"
                elif [[ "$name" == *voicedesign* || "$name" == *voice-design* ]]; then
                    MODEL_TYPES["$name"]="voicedesign"
                else
                    MODEL_TYPES["$name"]="customvoice"
                fi
                ;;
        esac
    else
        if [[ "$name" == *base* ]]; then
            MODEL_TYPES["$name"]="base"
        elif [[ "$name" == *voicedesign* || "$name" == *voice-design* ]]; then
            MODEL_TYPES["$name"]="voicedesign"
        else
            MODEL_TYPES["$name"]="customvoice"
        fi
    fi

    MODEL_NAMES+=("$name")
done

if [[ ${#MODEL_NAMES[@]} -eq 0 ]]; then
    echo "ERROR: No models found in $MODELS_DIR"
    echo "Run: ./scripts/download_test_data.sh"
    exit 1
fi

echo "Found ${#MODEL_NAMES[@]} models: ${MODEL_NAMES[*]}"
echo ""

# ── Define test cases ────────────────────────────────────────────────
# Each test: label + array of arguments (indexed by TEST_<N>_ARGS)
declare -a TEST_LABELS=()
test_count=0

add_test() {
    local label="$1"; shift
    TEST_LABELS+=("$label")
    # Store args as a declare-p'd array for safe retrieval
    local -a args=("$@")
    eval "TEST_${test_count}_ARGS=(\"\${args[@]}\")"
    test_count=$((test_count + 1))
}

for name in "${MODEL_NAMES[@]}"; do
    model_dir="${MODEL_PATHS[$name]}"
    model_type="${MODEL_TYPES[$name]}"

    if [[ "$model_type" == "base" ]]; then
        # Base models: x_vector_only and ICL
        if [[ -f "$REF_AUDIO" ]]; then
            add_test "${name}-xvector" \
                --model-dir "$model_dir" --ref-audio "$REF_AUDIO" --x-vector-only
            add_test "${name}-icl" \
                --model-dir "$model_dir" --ref-audio "$REF_AUDIO" --ref-text "$REF_TEXT"
        else
            echo "WARN: Skipping base model $name (no reference audio: $REF_AUDIO)"
        fi
    elif [[ "$model_type" == "voicedesign" ]]; then
        # VoiceDesign models: text-described voice via --instruct
        add_test "${name}-instruct" \
            --model-dir "$model_dir" --instruct "$INSTRUCT"
    else
        # CustomVoice: preset speakers
        tokenizer_args=()
        if [[ -f "$model_dir/tokenizer.json" ]]; then
            tokenizer_args=(--tokenizer-dir "$model_dir")
        fi
        add_test "${name}-ryan" \
            --model-dir "$model_dir" "${tokenizer_args[@]}" --speaker ryan
        add_test "${name}-serena" \
            --model-dir "$model_dir" "${tokenizer_args[@]}" --speaker serena
    fi
done

total_runs=$(( ${#TEST_LABELS[@]} * ${#DEVICES[@]} * BATCH_COUNT ))
echo "Test matrix: ${#TEST_LABELS[@]} tests x ${#DEVICES[@]} devices x ${BATCH_COUNT} seed(s) = ${total_runs} runs"
echo ""

# ── Run tests ────────────────────────────────────────────────────────
# Results arrays
declare -a RESULT_LABELS=()
declare -a RESULT_DEVICES=()
declare -a RESULT_SEEDS=()
declare -a RESULT_TIMES=()
declare -a RESULT_STATUSES=()
declare -a RESULT_SIZES=()
declare -a RESULT_FILES=()

run_num=0

for device in "${DEVICES[@]}"; do
    device_dir="$OUTPUT_BASE/$device"
    mkdir -p "$device_dir"

    for seed in "${SEEDS[@]}"; do
        for i in "${!TEST_LABELS[@]}"; do
            base_label="${TEST_LABELS[$i]}"
            # Retrieve the args array for this test
            eval "test_args=(\"\${TEST_${i}_ARGS[@]}\")"
            run_num=$((run_num + 1))

            # When batching, include seed in label and filename
            if [[ $BATCH_COUNT -gt 1 ]]; then
                label="${base_label}-s${seed}"
            else
                label="$base_label"
            fi
            outfile="$device_dir/${label}.wav"

            printf "[%d/%d] %-12s %-35s " "$run_num" "$total_runs" "$device" "$label"

            # Run and capture time
            start_time=$(date +%s%N)
            if "$BIN" --device "$device" "${test_args[@]}" \
                --text "$TEXT" --duration "$DURATION" --seed "$seed" \
                --output "$outfile" >/dev/null 2>&1; then
                status="PASS"
            else
                status="FAIL"
            fi
            end_time=$(date +%s%N)
            elapsed_ms=$(( (end_time - start_time) / 1000000 ))
            elapsed_s=$(awk "BEGIN{printf \"%.1f\", $elapsed_ms/1000}")

            # File size
            if [[ -f "$outfile" ]]; then
                size=$(du -h "$outfile" | cut -f1)
            else
                size="-"
            fi

            RESULT_LABELS+=("$label")
            RESULT_DEVICES+=("$device")
            RESULT_SEEDS+=("$seed")
            RESULT_TIMES+=("$elapsed_s")
            RESULT_STATUSES+=("$status")
            RESULT_SIZES+=("$size")
            RESULT_FILES+=("$outfile")

            if [[ "$status" == "PASS" ]]; then
                printf "%-6s %6ss  %s\n" "$status" "$elapsed_s" "$size"
            else
                printf "%-6s %6ss  (failed)\n" "$status" "$elapsed_s"
            fi
        done
    done
done

# ── Summary table ────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Build comparison table if multiple devices
if [[ ${#DEVICES[@]} -gt 1 ]]; then
    # Header
    printf "%-40s" "Test"
    for d in "${DEVICES[@]}"; do
        printf "  %10s" "$d"
    done
    printf "  %10s\n" "Speedup"

    printf "%-40s" "$(printf '%.0s─' {1..40})"
    for d in "${DEVICES[@]}"; do
        printf "  %10s" "──────────"
    done
    printf "  %10s\n" "──────────"

    # Collect unique labels
    declare -A seen_labels=()
    declare -a unique_labels=()
    for label in "${RESULT_LABELS[@]}"; do
        if [[ -z "${seen_labels[$label]:-}" ]]; then
            seen_labels["$label"]=1
            unique_labels+=("$label")
        fi
    done

    for label in "${unique_labels[@]}"; do
        printf "%-40s" "$label"
        declare -A device_times=()
        for i in "${!RESULT_LABELS[@]}"; do
            if [[ "${RESULT_LABELS[$i]}" == "$label" ]]; then
                d="${RESULT_DEVICES[$i]}"
                t="${RESULT_TIMES[$i]}"
                s="${RESULT_STATUSES[$i]}"
                if [[ "$s" == "PASS" ]]; then
                    printf "  %9ss" "$t"
                    device_times["$d"]="$t"
                else
                    printf "  %10s" "FAIL"
                fi
            fi
        done

        # Calculate speedup (cpu / fastest-gpu)
        if [[ -n "${device_times[cpu]:-}" ]]; then
            cpu_t="${device_times[cpu]}"
            best_gpu=""
            for d in "${DEVICES[@]}"; do
                [[ "$d" == "cpu" ]] && continue
                if [[ -n "${device_times[$d]:-}" ]]; then
                    if [[ -z "$best_gpu" ]] || awk "BEGIN{exit !($best_gpu > ${device_times[$d]})}" 2>/dev/null; then
                        best_gpu="${device_times[$d]}"
                    fi
                fi
            done
            if [[ -n "$best_gpu" ]]; then
                speedup=$(awk "BEGIN{printf \"%.1fx\", $cpu_t/$best_gpu}")
                printf "  %10s" "$speedup"
            fi
        fi
        echo ""
    done
else
    printf "%-40s  %10s  %6s  %s\n" "Test" "Device" "Time" "Size"
    printf "%-40s  %10s  %6s  %s\n" "$(printf '%.0s─' {1..40})" "──────────" "──────" "────"
    for i in "${!RESULT_LABELS[@]}"; do
        printf "%-40s  %10s  %5ss  %s\n" \
            "${RESULT_LABELS[$i]}" "${RESULT_DEVICES[$i]}" \
            "${RESULT_TIMES[$i]}" "${RESULT_SIZES[$i]}"
    done
fi

# Pass/fail summary
pass_count=0
fail_count=0
for s in "${RESULT_STATUSES[@]}"; do
    if [[ "$s" == "PASS" ]]; then
        pass_count=$((pass_count + 1))
    else
        fail_count=$((fail_count + 1))
    fi
done

echo ""
echo "Results: $pass_count passed, $fail_count failed out of $total_runs total"

# ── Generate results JSON ────────────────────────────────────────────
results_json="$OUTPUT_BASE/results.json"
{
    echo "["
    for i in "${!RESULT_LABELS[@]}"; do
        [[ $i -gt 0 ]] && echo ","
        relpath="${RESULT_FILES[$i]#$OUTPUT_BASE/}"
        printf '  {"label":"%s","device":"%s","seed":%s,"time":"%s","status":"%s","size":"%s","file":"%s"}' \
            "${RESULT_LABELS[$i]}" "${RESULT_DEVICES[$i]}" \
            "${RESULT_SEEDS[$i]}" "${RESULT_TIMES[$i]}" "${RESULT_STATUSES[$i]}" \
            "${RESULT_SIZES[$i]}" "$relpath"
    done
    echo ""
    echo "]"
} > "$results_json"

# ── Generate index.html with audio players ──────────────────────────
index_html="$OUTPUT_BASE/index.html"
cat > "$index_html" <<'HTMLHEAD'
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TTS Variant Test Results</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #e0e0e0; }
  h1 { color: #fff; border-bottom: 2px solid #444; padding-bottom: 10px; }
  h2 { color: #ccc; margin-top: 30px; }
  .meta { color: #888; font-size: 14px; margin-bottom: 20px; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; }
  th, td { border: 1px solid #333; padding: 8px 12px; text-align: left; }
  th { background: #2a2a4a; color: #fff; }
  tr:nth-child(even) { background: #1e1e3a; }
  tr:nth-child(odd) { background: #222244; }
  .pass { color: #4caf50; font-weight: bold; }
  .fail { color: #f44336; font-weight: bold; }
  .speedup { color: #ff9800; font-weight: bold; }
  audio { width: 100%; max-width: 400px; }
  .card { background: #222244; border: 1px solid #333; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .card h3 { margin: 0 0 8px 0; color: #fff; }
  .card .details { color: #888; font-size: 13px; margin-bottom: 8px; }
</style>
</head>
<body>
<h1>TTS Variant Test Results</h1>
HTMLHEAD

# Add metadata
{
    if [[ $BATCH_COUNT -gt 1 ]]; then
        seed_info="Seeds: ${SEEDS[0]}&ndash;${SEEDS[-1]} (${BATCH_COUNT} runs)"
    else
        seed_info="Seed: ${SEED}"
    fi
    echo "<p class=\"meta\">Generated: $(date -Iseconds) &mdash; Text: &quot;${TEXT}&quot; &mdash; ${seed_info} &mdash; Duration: ${DURATION}s</p>"

    # Summary table
    echo "<h2>Summary</h2>"
    echo "<table><tr><th>Test</th>"
    for d in "${DEVICES[@]}"; do
        echo "<th>$d</th>"
    done
    if [[ ${#DEVICES[@]} -gt 1 ]]; then
        echo "<th>Speedup</th>"
    fi
    echo "</tr>"

    # Collect unique labels
    declare -A html_seen=()
    declare -a html_unique=()
    for label in "${RESULT_LABELS[@]}"; do
        if [[ -z "${html_seen[$label]:-}" ]]; then
            html_seen["$label"]=1
            html_unique+=("$label")
        fi
    done

    for label in "${html_unique[@]}"; do
        echo "<tr><td>$label</td>"
        declare -A html_times=()
        for i in "${!RESULT_LABELS[@]}"; do
            if [[ "${RESULT_LABELS[$i]}" == "$label" ]]; then
                d="${RESULT_DEVICES[$i]}"
                t="${RESULT_TIMES[$i]}"
                s="${RESULT_STATUSES[$i]}"
                if [[ "$s" == "PASS" ]]; then
                    echo "<td class=\"pass\">${t}s</td>"
                    html_times["$d"]="$t"
                else
                    echo "<td class=\"fail\">FAIL</td>"
                fi
            fi
        done

        if [[ ${#DEVICES[@]} -gt 1 && -n "${html_times[cpu]:-}" ]]; then
            cpu_t="${html_times[cpu]}"
            best_gpu=""
            for d in "${DEVICES[@]}"; do
                [[ "$d" == "cpu" ]] && continue
                if [[ -n "${html_times[$d]:-}" ]]; then
                    if [[ -z "$best_gpu" ]] || awk "BEGIN{exit !($best_gpu > ${html_times[$d]})}" 2>/dev/null; then
                        best_gpu="${html_times[$d]}"
                    fi
                fi
            done
            if [[ -n "$best_gpu" ]]; then
                speedup=$(awk "BEGIN{printf \"%.1fx\", $cpu_t/$best_gpu}")
                echo "<td class=\"speedup\">$speedup</td>"
            else
                echo "<td>-</td>"
            fi
        elif [[ ${#DEVICES[@]} -gt 1 ]]; then
            echo "<td>-</td>"
        fi
        echo "</tr>"
    done
    echo "</table>"

    # Audio players grouped by device
    for device in "${DEVICES[@]}"; do
        echo "<h2>$device</h2>"
        for i in "${!RESULT_LABELS[@]}"; do
            if [[ "${RESULT_DEVICES[$i]}" == "$device" && "${RESULT_STATUSES[$i]}" == "PASS" ]]; then
                relpath="${RESULT_FILES[$i]#$OUTPUT_BASE/}"
                echo "<div class=\"card\">"
                echo "  <h3>${RESULT_LABELS[$i]}</h3>"
                echo "  <div class=\"details\">seed ${RESULT_SEEDS[$i]} &mdash; ${RESULT_TIMES[$i]}s &mdash; ${RESULT_SIZES[$i]}</div>"
                echo "  <audio controls preload=\"metadata\" src=\"$relpath\"></audio>"
                echo "</div>"
            fi
        done
    done

    echo "<p class=\"meta\">$pass_count passed, $fail_count failed out of $total_runs total</p>"
    echo "</body></html>"
} >> "$index_html"

echo ""
echo "Results: $results_json"
echo "Listen:  $index_html"

# ── HTTP server ──────────────────────────────────────────────────────
if $SERVE; then
    echo ""
    echo "Starting HTTP server on port $HTTP_PORT..."

    # Kill existing server
    pkill -f "python3 -m http.server $HTTP_PORT" 2>/dev/null || true
    sleep 0.5

    cd "$OUTPUT_BASE"
    python3 -m http.server "$HTTP_PORT" &>/dev/null &
    server_pid=$!
    echo "Server PID: $server_pid"
    echo ""
    echo "  http://${HOSTNAME}:${HTTP_PORT}/"
    echo ""
    echo "Stop server: kill $server_pid"
fi

# Exit with failure if any test failed
[[ $fail_count -eq 0 ]]

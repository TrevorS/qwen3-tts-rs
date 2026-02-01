#!/usr/bin/env bash
# Count CUDA kernel launches per frame from a chrome-trace JSON.
#
# Runs e2e_bench with profiling enabled, then parses the trace to count
# kernel launches. Useful for validating that fused ops reduce launch count.
#
# Usage: ./scripts/count-kernels.sh [model_dir]
#        make count-kernels MODEL_DIR=test_data/models/1.7B-CustomVoice
#
# Requires: jq, cargo with cuda+profiling features (or Docker)

set -euo pipefail

MODEL_DIR="${1:-${MODEL_DIR:-test_data/models/1.7B-CustomVoice}}"
TRACE_FILE="trace-kernel-count.json"

echo "=== Kernel Launch Counter ==="
echo "Model: $MODEL_DIR"
echo

# Run e2e_bench with profiling to generate trace
TRACE_FILE_ENV="$TRACE_FILE" cargo run --profile=profiling \
    --features=profiling,cuda,cli --bin e2e_bench -- \
    --model-dir "$MODEL_DIR" --iterations 1

if [[ ! -f "$TRACE_FILE" ]]; then
    echo "Error: trace file not found at $TRACE_FILE"
    echo "The e2e_bench binary may write to a different path — check for trace*.json"
    ls -la trace*.json 2>/dev/null || true
    exit 1
fi

echo
echo "=== Trace Analysis ==="

# Count total events
TOTAL_EVENTS=$(jq '[.traceEvents[] | select(.ph == "X" or .ph == "B")] | length' "$TRACE_FILE")
echo "Total span events: $TOTAL_EVENTS"

# Count unique span names
echo
echo "--- Span counts (top 20) ---"
jq -r '[.traceEvents[] | select(.ph == "X" or .ph == "B") | .name] | group_by(.) | map({name: .[0], count: length}) | sort_by(-.count) | .[:20][] | "\(.count)\t\(.name)"' "$TRACE_FILE"

# Count spans that look like kernel-related operations
echo
echo "--- Kernel-related spans ---"
jq -r '[.traceEvents[] | select(.ph == "X" or .ph == "B") | .name | select(test("matmul|norm|rope|silu|softmax|attention|linear|embed"; "i"))] | group_by(.) | map({name: .[0], count: length}) | sort_by(-.count) | .[] | "\(.count)\t\(.name)"' "$TRACE_FILE"

# Try to identify per-frame counts
DECODE_STEPS=$(jq '[.traceEvents[] | select(.name == "decode_step" and (.ph == "X" or .ph == "B"))] | length' "$TRACE_FILE")
if [[ "$DECODE_STEPS" -gt 0 ]]; then
    echo
    echo "Decode steps: $DECODE_STEPS"
    echo "Avg spans per decode step: $((TOTAL_EVENTS / DECODE_STEPS))"
fi

# Cleanup
rm -f "$TRACE_FILE"
echo
echo "Done."

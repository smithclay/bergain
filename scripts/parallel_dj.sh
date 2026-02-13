#!/usr/bin/env bash
# parallel_dj.sh — Run bergain DJ with multiple OpenRouter models in parallel
#
# Usage:
#   ./scripts/parallel_dj.sh [--bars N] [--bpm N]
#
# Requires: OPENROUTER_API_KEY in env or .env file
set -euo pipefail

# Source .env if present (dj.py also calls load_dotenv, but this ensures
# the vars are available to LiteLLM's provider detection early)
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

BARS=64
BPM=128
CRITIC_LM="openrouter/openai/gpt-5-nano"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bars)      BARS="$2";      shift 2 ;;
        --bpm)       BPM="$2";       shift 2 ;;
        --critic-lm) CRITIC_LM="$2"; shift 2 ;;
        *)           shift ;;
    esac
done

# Models to compare — diverse mix of providers via OpenRouter
#MODELS=(
#    "openrouter/anthropic/claude-sonnet-4.5"
#    "openrouter/openai/gpt-5-mini"
#    "openrouter/deepseek/deepseek-v3.2"
#    "openrouter/google/gemini-3-flash-preview"
#    "openrouter/moonshotai/kimi-k2.5"
#)

MODELS=(
    "openrouter/openai/gpt-5-mini"
    "openrouter/deepseek/deepseek-v3.2"
)

OUTDIR="output/parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

# Generate a shared palette so all models use the same samples
PALETTE="$OUTDIR/palette.json"
uv run python -c "
import json
from bergain.indexer import build_index
from bergain.dj import _build_role_map, _pick_random_palette
role_map = _build_role_map(build_index('sample_pack'))
palette = _pick_random_palette(role_map)
with open('$PALETTE', 'w') as f:
    json.dump(palette, f, indent=2)
print(json.dumps({r: p.split('/')[-1] for r, p in palette.items()}, indent=2))
"
echo "Shared palette saved to $PALETTE"
echo ""

echo "=== Parallel DJ Shootout ==="
echo "Bars: $BARS | BPM: $BPM | Critic: $CRITIC_LM | Output: $OUTDIR"
echo "Models: ${#MODELS[@]}"
echo ""

PIDS=()
NAMES=()
STATUSES=()

for model in "${MODELS[@]}"; do
    slug=$(echo "$model" | sed 's|openrouter/||; s|/|_|g')
    outfile="$OUTDIR/${slug}.wav"
    logfile="$OUTDIR/${slug}.log"

    echo "Starting: $model -> $outfile"
    PYTHONUNBUFFERED=1 uv run bergain dj \
        --lm "$model" \
        --critic-lm "$CRITIC_LM" \
        --palette "$PALETTE" \
        --bars "$BARS" \
        --bpm "$BPM" \
        -o "$outfile" \
        > "$logfile" 2>&1 &

    PIDS+=($!)
    NAMES+=("$slug")
done

echo ""
echo "All ${#PIDS[@]} models launched. Waiting..."
echo ""

# Wait for each and track results
FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    name=${NAMES[$i]}
    if wait "$pid"; then
        echo "  OK  $name"
        STATUSES+=("ok")
    else
        echo "  FAIL $name (see $OUTDIR/${name}.log)"
        STATUSES+=("fail")
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== Done ==="
echo "Succeeded: $(( ${#PIDS[@]} - FAILED )) / ${#PIDS[@]}"

# Show tail of failed logs
if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo "Failed logs:"
    for i in "${!PIDS[@]}"; do
        if [[ "${STATUSES[$i]}" == "fail" ]]; then
            echo "--- ${NAMES[$i]} ---"
            tail -20 "$OUTDIR/${NAMES[$i]}.log"
            echo ""
        fi
    done
fi

# List output files with sizes
echo ""
echo "Output files:"
ls -lh "$OUTDIR"/*.wav 2>/dev/null || echo "  (no WAV files produced)"

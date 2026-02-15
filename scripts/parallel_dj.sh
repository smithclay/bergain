#!/usr/bin/env bash
# parallel_dj.sh — Run bergain DJ across models and/or prompts in parallel
#
# Usage:
#   ./scripts/parallel_dj.sh [--bars N] [--bpm N] [--prompt FILE]... [--outdir DIR] [--palette FILE] [--palette-dir DIR]
#
# Multiple --prompt flags run each prompt variant in parallel.
# If no --prompt is given, uses the built-in prompt.
#
# --outdir DIR        Use DIR instead of auto-generated output/parallel_YYYYMMDD_HHMMSS/
# --palette FILE      Skip palette generation, use this palette.json
# --palette-dir DIR   Pick a random curated palette from DIR (ignored if --palette is set)
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
PROMPT_FILES=()
CUSTOM_OUTDIR=""
CUSTOM_PALETTE=""
PALETTE_DIR=""
NO_CACHE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bars)        BARS="$2";            shift 2 ;;
        --bpm)         BPM="$2";             shift 2 ;;
        --critic-lm)   CRITIC_LM="$2";       shift 2 ;;
        --prompt)      PROMPT_FILES+=("$2");  shift 2 ;;
        --outdir)      CUSTOM_OUTDIR="$2";    shift 2 ;;
        --palette)     CUSTOM_PALETTE="$2";   shift 2 ;;
        --palette-dir) PALETTE_DIR="$2";      shift 2 ;;
        --no-cache)    NO_CACHE="--no-cache"; shift ;;
        *)             shift ;;
    esac
done

# Default: built-in prompt (empty string = no --prompt flag)
if [[ ${#PROMPT_FILES[@]} -eq 0 ]]; then
    PROMPT_FILES=("")
fi

MODELS=(
    "openrouter/openai/gpt-5-mini"
)

if [[ -n "$CUSTOM_OUTDIR" ]]; then
    OUTDIR="$CUSTOM_OUTDIR"
else
    OUTDIR="output/parallel_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTDIR"

# Generate or reuse a shared palette so all runs use the same samples
if [[ -n "$CUSTOM_PALETTE" ]]; then
    PALETTE="$OUTDIR/palette.json"
    cp "$CUSTOM_PALETTE" "$PALETTE"
    echo "Using provided palette: $CUSTOM_PALETTE"
elif [[ -n "$PALETTE_DIR" ]]; then
    # Pick a random curated palette from the directory
    PICKED=$(python3 -c "import random,glob; f=glob.glob('$PALETTE_DIR/*.json'); print(random.choice(f)) if f else exit(1)" 2>/dev/null)
    if [[ -z "$PICKED" ]]; then
        echo "ERROR: No .json files found in $PALETTE_DIR" >&2
        exit 1
    fi
    PALETTE="$OUTDIR/palette.json"
    cp "$PICKED" "$PALETTE"
    echo "Using curated palette: $PICKED"
else
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
fi
echo ""

# Count total runs
TOTAL_RUNS=$(( ${#MODELS[@]} * ${#PROMPT_FILES[@]} ))
echo "=== Parallel DJ Shootout ==="
echo "Bars: $BARS | BPM: $BPM | Critic: $CRITIC_LM | Output: $OUTDIR"
echo "Models: ${#MODELS[@]} | Prompts: ${#PROMPT_FILES[@]} | Total runs: $TOTAL_RUNS"
echo ""

PIDS=()
NAMES=()
PROMPT_LABELS=()
STATUSES=()

for model in "${MODELS[@]}"; do
    model_slug=$(echo "$model" | sed 's|openrouter/||; s|/|_|g')

    for prompt_file in "${PROMPT_FILES[@]}"; do
        if [[ -n "$prompt_file" ]]; then
            prompt_label=$(basename "$prompt_file" .txt)
            slug="${model_slug}__${prompt_label}"
            PROMPT_ARG="--prompt $prompt_file"
            # Copy prompt into output dir for reproducibility
            cp "$prompt_file" "$OUTDIR/${prompt_label}.prompt.txt"
        else
            prompt_label="built-in"
            slug="${model_slug}__built-in"
            PROMPT_ARG=""
        fi

        outfile="$OUTDIR/${slug}.wav"
        logfile="$OUTDIR/${slug}.log"

        echo "Starting: $model + $prompt_label -> $outfile"
        PYTHONUNBUFFERED=1 uv run bergain dj \
            --lm "$model" \
            --critic-lm "$CRITIC_LM" \
            --palette "$PALETTE" \
            --bars "$BARS" \
            --bpm "$BPM" \
            -o "$outfile" \
            $PROMPT_ARG \
            $NO_CACHE \
            > "$logfile" 2>&1 &

        PIDS+=($!)
        NAMES+=("$slug")
        PROMPT_LABELS+=("$prompt_label")
    done
done

echo ""
echo "All ${#PIDS[@]} runs launched. Waiting..."
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

# --- Aesthetics scoring via Modal endpoint ---
SCORER_URL="https://smithclay--bergain-aesthetics-judge-score.modal.run"
echo ""
echo "=== Aesthetics Scoring ==="
echo "Endpoint: $SCORER_URL"
echo ""

SCORES_JSON="$OUTDIR/scores.json"
echo "[" > "$SCORES_JSON"
FIRST=true

for i in "${!NAMES[@]}"; do
    name=${NAMES[$i]}
    prompt_label=${PROMPT_LABELS[$i]}
    wavfile="$OUTDIR/${name}.wav"
    if [[ "${STATUSES[$i]}" != "ok" ]] || [[ ! -f "$wavfile" ]]; then
        echo "  SKIP $name -- no WAV"
        continue
    fi

    echo -n "  Scoring $name... "
    response=$(curl -s --max-time 120 -X POST "$SCORER_URL" -F "file=@${wavfile}" 2>/dev/null)
    if echo "$response" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
        scores=$(echo "$response" | python3 -c "$(cat <<'PYEOF'
import sys, json
d = json.load(sys.stdin)["scores"]
print(f'CE={d["CE"]:.2f}  CU={d["CU"]:.2f}  PC={d["PC"]:.2f}  PQ={d["PQ"]:.2f}')
PYEOF
)")
        echo "$scores"
        # Append to scores.json
        if [ "$FIRST" = true ]; then
            FIRST=false
        else
            echo "," >> "$SCORES_JSON"
        fi
        echo "$response" | MODEL_NAME="$name" PROMPT_LABEL="$prompt_label" python3 -c "$(cat <<'PYEOF'
import sys, json, os
d = json.load(sys.stdin)
entry = {
    "name": os.environ["MODEL_NAME"],
    "prompt": os.environ["PROMPT_LABEL"],
    "scores": d["scores"],
}
print(json.dumps(entry, indent=2))
PYEOF
)" >> "$SCORES_JSON"
    else
        echo "FAIL"
    fi
done

echo "]" >> "$SCORES_JSON"
echo ""
echo "Scores saved to $SCORES_JSON"

# Save built-in prompt for reference if it was used
for pl in "${PROMPT_LABELS[@]}"; do
    if [[ "$pl" == "built-in" ]]; then
        uv run python -c "$(cat <<'PYEOF'
from bergain.dj import DJ_INSTRUCTIONS
import sys
with open(sys.argv[1], "w") as f:
    f.write(DJ_INSTRUCTIONS)
PYEOF
)" "$OUTDIR/built-in.prompt.txt"
        break
    fi
done

# Print summary table
echo ""
echo "=== Summary ==="
SCORES_FILE="$SCORES_JSON" python3 -c "$(cat <<'PYEOF'
import json, sys, os
with open(os.environ["SCORES_FILE"]) as f:
    data = json.load(f)
if not data:
    print("  No scores available")
    sys.exit()
print(f'{"Name":<40} {"Prompt":<16} {"CE":>6} {"CU":>6} {"PC":>6} {"PQ":>6}')
print("-" * 82)
for entry in data:
    s = entry["scores"]
    print(f'{entry["name"]:<40} {entry["prompt"]:<16} {s["CE"]:6.2f} {s["CU"]:6.2f} {s["PC"]:6.2f} {s["PQ"]:6.2f}')
print("-" * 82)
# Averages per prompt
prompts = sorted(set(e["prompt"] for e in data))
for p in prompts:
    subset = [e for e in data if e["prompt"] == p]
    avgs = {k: sum(e["scores"][k] for e in subset)/len(subset) for k in ["CE","CU","PC","PQ"]}
    label = f"AVG ({p})"
    print(f'{label:<40} {"":16} {avgs["CE"]:6.2f} {avgs["CU"]:6.2f} {avgs["PC"]:6.2f} {avgs["PQ"]:6.2f}')
PYEOF
)"

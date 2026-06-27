#!/usr/bin/env bash
# ============================================================================
# Subword (GPT-2 BPE) robustness run for the frequency-schedule sweep.
#
# Same three arms as run_char.sh, but on a real Zipfian *subword* vocabulary
# (~50K types) where the "rare = content word" story is linguistically clean.
# The embedding table jumps to ~19M params (~30M total), so we shorten the
# block and (by default) run a single seed to keep this affordable as a
# robustness check on top of the char sweep.
#
# Writes to a SEPARATE data dir (data_bpe) so it never clobbers the char corpus.
# Run the char sweep first; this is the secondary experiment.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"

# uv virtualenv setup (SKIP_ENV=1 to use the active python instead)
source experiments/setup_env.sh

DATASET="${DATASET:-tinystories}"
DATADIR="${DATADIR:-data_bpe}"
MAXSTEPS="${MAXSTEPS:-20000}"
SEEDS="${SEEDS:-0}"
ARMS="${ARMS:-uniform rare_first frequent_first}"
BETA="${BETA:-2.0}"
DEVICE="${DEVICE:-cuda}"
OUTROOT="${OUTROOT:-runs/bpe}"
# smaller block to offset the big embedding table; same depth/width otherwise
NLAYER="${NLAYER:-6}"; NEMBD="${NEMBD:-384}"; NHEAD="${NHEAD:-6}"
BLOCK="${BLOCK:-128}"; BATCH="${BATCH:-64}"
EVAL_NSAMPLES="${EVAL_NSAMPLES:-16}"; EVAL_STEPS="${EVAL_STEPS:-64}"
EXTRA="${EXTRA:-}"

COMMON_MODEL=(--data-dir "$DATADIR" --max-steps "$MAXSTEPS" --device "$DEVICE" \
              --n-layer "$NLAYER" --n-embd "$NEMBD" --n-head "$NHEAD" \
              --block-size "$BLOCK" --batch-size "$BATCH" $EXTRA)

echo "=== [1/3] prepare $DATASET (gpt2 BPE) -> $DATADIR ==="
python prepare.py --dataset "$DATASET" --tokenizer gpt2 --data-dir "$DATADIR"

AR_DIR="$OUTROOT/ar"
echo "=== [2/3] train shared AR baseline -> $AR_DIR ==="
python train_ar.py "${COMMON_MODEL[@]}" --seed 0 --out-dir "$AR_DIR"

echo "=== [3/3] sweep schedules x seeds ==="
for arm in $ARMS; do
  for seed in $SEEDS; do
    run="$OUTROOT/${arm}_s${seed}"
    echo "--- train MDM: schedule=$arm seed=$seed -> $run ---"
    python train.py "${COMMON_MODEL[@]}" --schedule "$arm" --zipf-beta "$BETA" \
                    --seed "$seed" --out-dir "$run"
    echo "--- eval (selection-free final ckpt): $run ---"
    python eval.py --mdm-ckpt "$run/ckpt_last.pt" --ar-ckpt "$AR_DIR/ckpt_ar.pt" \
                   --n-samples "$EVAL_NSAMPLES" --steps "$EVAL_STEPS" \
                   --out "$run/eval_table.md" --seed 0
  done
done

echo "=== aggregate ==="
python experiments/aggregate.py "$OUTROOT" | tee "$OUTROOT/scoreboard.md"
echo "done. scoreboard at $OUTROOT/scoreboard.md"

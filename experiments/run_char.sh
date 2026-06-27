#!/usr/bin/env bash
# ============================================================================
# Char-level frequency-schedule sweep: uniform vs rare_first vs frequent_first.
#
# Trains one MDM per (schedule, seed) plus a single shared AR baseline, then
# evals each MDM against that AR with the repo's eval harness. Every arm is
# judged by the SAME uniform-masking val ELBO and the SAME AR-scored sample
# PPL, so the comparison isolates the effect of the *training* schedule.
#
# On an A100-80GB the default 20k-step / 10M-param run is ~10-15 min per model;
# 1 AR + 9 MDM = ~2-2.5 h plus eval. Override anything via env vars, e.g.:
#   MAXSTEPS=2000 SEEDS="0" ARMS="uniform rare_first" bash experiments/run_char.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"

# uv virtualenv setup (SKIP_ENV=1 to use the active python instead)
source experiments/setup_env.sh

# ---- knobs (env-overridable) ----------------------------------------------
DATASET="${DATASET:-tinystories}"          # tinystories | shakespeare
DATADIR="${DATADIR:-data_char}"
MAXSTEPS="${MAXSTEPS:-20000}"
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-uniform rare_first frequent_first}"
BETA="${BETA:-2.0}"
DEVICE="${DEVICE:-cuda}"
OUTROOT="${OUTROOT:-runs/char}"
# model size (defaults match config.py's ~10M char-level model)
NLAYER="${NLAYER:-6}"; NEMBD="${NEMBD:-384}"; NHEAD="${NHEAD:-6}"
BLOCK="${BLOCK:-256}"; BATCH="${BATCH:-64}"
# eval budget
EVAL_NSAMPLES="${EVAL_NSAMPLES:-16}"; EVAL_STEPS="${EVAL_STEPS:-64}"
EXTRA="${EXTRA:-}"                          # extra flags passed to train.py (e.g. --no-compile)

COMMON_MODEL=(--data-dir "$DATADIR" --max-steps "$MAXSTEPS" --device "$DEVICE" \
              --n-layer "$NLAYER" --n-embd "$NEMBD" --n-head "$NHEAD" \
              --block-size "$BLOCK" --batch-size "$BATCH" $EXTRA)

echo "=== [1/3] prepare $DATASET (char) -> $DATADIR ==="
python prepare.py --dataset "$DATASET" --tokenizer char --data-dir "$DATADIR"

AR_DIR="$OUTROOT/ar"
echo "=== [2/3] train shared AR baseline -> $AR_DIR ==="
python train_ar.py "${COMMON_MODEL[@]}" --seed 0 --out-dir "$AR_DIR"

# eval helper: always evaluates the selection-free FINAL checkpoint (ckpt_last),
# so no arm gets a uniform-ELBO best-val home-field advantage.
eval_run() {  # $1 = run dir
  python eval.py --mdm-ckpt "$1/ckpt_last.pt" --ar-ckpt "$AR_DIR/ckpt_ar.pt" \
                 --n-samples "$EVAL_NSAMPLES" --steps "$EVAL_STEPS" \
                 --out "$1/eval_table.md" --seed 0
}

echo "=== [3/3] sweep schedules x seeds ==="
for arm in $ARMS; do
  for seed in $SEEDS; do
    run="$OUTROOT/${arm}_s${seed}"
    echo "--- train MDM: schedule=$arm seed=$seed -> $run ---"
    python train.py "${COMMON_MODEL[@]}" --schedule "$arm" --zipf-beta "$BETA" \
                    --seed "$seed" --out-dir "$run"
    eval_run "$run"
  done
done

# Matched-noise attribution controls (opt-in: CONTROLS=1). For each Zipf arm,
# train a constant-exponent model with the SAME mean corpus mask rate but ZERO
# frequency ordering. If a Zipf arm beats its matched control, the ORDERING
# helped; if not, it was just the noise level. Defaults off to keep the base run
# small — turn on once the main sweep shows an effect worth attributing.
if [ "${CONTROLS:-0}" = "1" ]; then
  echo "=== controls: matched-noise (constant exponent, no ordering) ==="
  for target in rare_first frequent_first; do
    short=$([ "$target" = rare_first ] && echo match_rare || echo match_freq)
    for seed in ${CONTROL_SEEDS:-0}; do
      run="$OUTROOT/${short}_s${seed}"
      echo "--- train control: match $target seed=$seed -> $run ---"
      python train.py "${COMMON_MODEL[@]}" --schedule uniform --match-noise "$target" \
                      --zipf-beta "$BETA" --seed "$seed" --out-dir "$run"
      eval_run "$run"
    done
  done
fi

echo "=== aggregate ==="
python experiments/aggregate.py "$OUTROOT" | tee "$OUTROOT/scoreboard.md"
echo "done. scoreboard at $OUTROOT/scoreboard.md"

#!/usr/bin/env bash
# ============================================================================
# Local CPU smoke test: prove the whole pipeline runs end to end before you
# spend A100 time. Tiny model, tiny data, a handful of steps. NOT a real run.
#
#   bash experiments/smoke.sh          # char, ~1-2 min on CPU
#   TOKENIZER=gpt2 bash experiments/smoke.sh   # also exercise the BPE path
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"

# uv virtualenv setup (SKIP_ENV=1 to reuse an already-active venv)
source experiments/setup_env.sh

TOKENIZER="${TOKENIZER:-char}"
DATADIR="${DATADIR:-data_smoke_$TOKENIZER}"
DEVICE="${DEVICE:-cpu}"
OUTROOT="${OUTROOT:-runs/smoke_$TOKENIZER}"
# deliberately tiny (block-size 128 so eval's hardcoded infill total_len=128 fits)
TINY=(--device "$DEVICE" --no-compile --n-layer 2 --n-embd 64 --n-head 2 \
      --block-size 128 --batch-size 8 --max-steps 30 --eval-interval 15 \
      --sample-interval 1000 --data-dir "$DATADIR")

echo "=== 0) schedule math checks ==="
python experiments/check_schedule.py

echo "=== 1) prepare tiny shakespeare ($TOKENIZER) ==="
python prepare.py --dataset shakespeare --tokenizer "$TOKENIZER" --data-dir "$DATADIR"

echo "=== 2) tiny AR baseline ==="
python train_ar.py "${TINY[@]}" --seed 0 --out-dir "$OUTROOT/ar"

# three main arms + one matched-noise control (exercises the --match-noise path)
for spec in "uniform:--schedule uniform" \
            "rare_first:--schedule rare_first" \
            "frequent_first:--schedule frequent_first" \
            "match_freq:--schedule uniform --match-noise frequent_first"; do
  arm="${spec%%:*}"; flags="${spec#*:}"
  run="$OUTROOT/${arm}_s0"
  echo "=== 3) tiny MDM: $arm ($flags) ==="
  python train.py "${TINY[@]}" $flags --zipf-beta 2.0 --seed 0 --out-dir "$run"
  echo "--- eval $arm (selection-free final ckpt) ---"
  python eval.py --mdm-ckpt "$run/ckpt_last.pt" --ar-ckpt "$OUTROOT/ar/ckpt_ar.pt" \
                 --n-samples 2 --steps 4 --n-batches 3 --length 64 \
                 --out "$run/eval_table.md" --seed 0
done

echo "=== 4) aggregate ==="
python experiments/aggregate.py "$OUTROOT"
echo "SMOKE OK"

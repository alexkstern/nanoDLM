#!/usr/bin/env bash
# ============================================================================
# Quick single-seed head-to-head: uniform vs ONE Zipf direction, on BOTH
# char and gpt2-BPE. Designed to run right after cancelling a full run_char.sh:
# it REUSES any checkpoints already on disk (char AR + char uniform/seed0) and
# trains only what's missing, so you get a first signal fast.
#
#   SCHED=rare_first bash experiments/run_quick.sh        # spectral analog (default)
#   SCHED=frequent_first bash experiments/run_quick.sh    # keep-content-words direction
#
# Re-running with the other SCHED reuses everything shared (both AR baselines +
# both uniform arms) and only trains the new Zipf arm — so two runs give you all
# four comparisons cheaply.
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$SCRIPT_DIR")"; cd "$REPO"
source experiments/setup_env.sh

SCHED="${SCHED:-rare_first}"               # rare_first | frequent_first
BETA="${BETA:-2.0}"; SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
MAXSTEPS="${MAXSTEPS:-20000}"             # char (matches the cancelled run's arms)
BPE_MAXSTEPS="${BPE_MAXSTEPS:-12000}"     # bpe is heavier per step; shorter by default
BPE_MAXBYTES="${BPE_MAXBYTES:-104857600}" # 100MB so the bigger model sees more data
EVAL_NSAMPLES="${EVAL_NSAMPLES:-16}"; EVAL_STEPS="${EVAL_STEPS:-64}"

if [ "$DEVICE" = cuda ] && ! python -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "ERROR: DEVICE=cuda but torch.cuda.is_available()==False (torch=$(python -c 'import torch;print(torch.__version__)'))."; exit 1; fi

train_if_missing() { local ck="$1"; shift; if [ -f "$ck" ]; then echo "reuse $ck"; else python "$@"; fi; }
do_eval()          { local d="$1" ar="$2"; [ -f "$d/eval_table.md" ] || \
  python eval.py --mdm-ckpt "$d/ckpt_last.pt" --ar-ckpt "$ar" \
                 --n-samples "$EVAL_NSAMPLES" --steps "$EVAL_STEPS" --out "$d/eval_table.md" --seed 0; }

# ===== CHAR: reuse data_char + AR + uniform/seed0 from the cancelled run =====
CHAR=(--data-dir data_char --device "$DEVICE" --max-steps "$MAXSTEPS" \
      --n-layer 6 --n-embd 384 --n-head 6 --block-size 256 --batch-size 64)
[ -f data_char/train.bin ] || python prepare.py --dataset tinystories --tokenizer char --data-dir data_char
train_if_missing runs/char/ar/ckpt_ar.pt           train_ar.py "${CHAR[@]}" --seed 0 --out-dir runs/char/ar
train_if_missing runs/char/uniform_s0/ckpt_last.pt train.py    "${CHAR[@]}" --schedule uniform --seed 0 --out-dir runs/char/uniform_s0
do_eval runs/char/uniform_s0 runs/char/ar/ckpt_ar.pt
echo "=== CHAR: train zipf=$SCHED (beta=$BETA, seed=$SEED) ==="
train_if_missing "runs/char/${SCHED}_s${SEED}/ckpt_last.pt" train.py "${CHAR[@]}" \
    --schedule "$SCHED" --zipf-beta "$BETA" --seed "$SEED" --out-dir "runs/char/${SCHED}_s${SEED}"
do_eval "runs/char/${SCHED}_s${SEED}" runs/char/ar/ckpt_ar.pt
echo "===== CHAR scoreboard ====="; python experiments/aggregate.py runs/char | tee runs/char/scoreboard.md

# ===== BPE: from scratch (AR + uniform + zipf), single seed =====
BPE=(--data-dir data_bpe --device "$DEVICE" --max-steps "$BPE_MAXSTEPS" \
     --n-layer 6 --n-embd 384 --n-head 6 --block-size 256 --batch-size 64)
[ -f data_bpe/train.bin ] || python prepare.py --dataset tinystories --tokenizer gpt2 --data-dir data_bpe --max-bytes "$BPE_MAXBYTES"
train_if_missing runs/bpe/ar/ckpt_ar.pt           train_ar.py "${BPE[@]}" --seed 0 --out-dir runs/bpe/ar
echo "=== BPE: train uniform ==="
train_if_missing runs/bpe/uniform_s0/ckpt_last.pt train.py "${BPE[@]}" --schedule uniform --seed 0 --out-dir runs/bpe/uniform_s0
do_eval runs/bpe/uniform_s0 runs/bpe/ar/ckpt_ar.pt
echo "=== BPE: train zipf=$SCHED ==="
train_if_missing "runs/bpe/${SCHED}_s0/ckpt_last.pt" train.py "${BPE[@]}" \
    --schedule "$SCHED" --zipf-beta "$BETA" --seed 0 --out-dir "runs/bpe/${SCHED}_s0"
do_eval "runs/bpe/${SCHED}_s0" runs/bpe/ar/ckpt_ar.pt
echo "===== BPE scoreboard ====="; python experiments/aggregate.py runs/bpe | tee runs/bpe/scoreboard.md

echo "DONE. scoreboards: runs/char/scoreboard.md  and  runs/bpe/scoreboard.md"

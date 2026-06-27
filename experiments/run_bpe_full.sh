#!/usr/bin/env bash
# ============================================================================
# Definitive BPE answer: does rare_first beat uniform at the subword level, and
# is it the frequency ORDERING or just a lower effective noise level?
#
# 5 arms x 3 seeds + a shared AR scorer:
#   uniform                          baseline
#   rare_first / frequent_first      the two opposite orderings (mirror test:
#                                    rare < uniform < frequent on PPL => ordering)
#   match_rare / match_freq          constant-exponent controls with the SAME mean
#                                    corpus mask rate but ZERO ordering. If a Zipf
#                                    arm beats its match_* twin, the ORDERING did
#                                    it; if it only matches the twin, it was noise.
#
# Reuses any BPE checkpoints already on disk (AR, uniform_s0, rare_first_s0) so it
# only trains what's new. ~3 h on an A100. Leave it; read runs/bpe/scoreboard.md.
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$SCRIPT_DIR")"; cd "$REPO"
source experiments/setup_env.sh

SEEDS="${SEEDS:-0 1 2}"
BETA="${BETA:-2.0}"
DEVICE="${DEVICE:-cuda}"
MAXSTEPS="${MAXSTEPS:-12000}"                 # matches the BPE arms already on disk
MAXBYTES="${MAXBYTES:-104857600}"            # 100MB (matches existing data_bpe)
EVAL_NSAMPLES="${EVAL_NSAMPLES:-16}"; EVAL_STEPS="${EVAL_STEPS:-64}"

if [ "$DEVICE" = cuda ] && ! python -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "ERROR: DEVICE=cuda but torch.cuda.is_available()==False (torch=$(python -c 'import torch;print(torch.__version__)'))."; exit 1; fi

BPE=(--data-dir data_bpe --device "$DEVICE" --max-steps "$MAXSTEPS" \
     --n-layer 6 --n-embd 384 --n-head 6 --block-size 256 --batch-size 64)
train_if_missing(){ local ck="$1"; shift; if [ -f "$ck" ]; then echo "reuse $ck"; else python "$@"; fi; }
do_eval(){ local d="$1"; [ -f "$d/eval_table.md" ] || \
  python eval.py --mdm-ckpt "$d/ckpt_last.pt" --ar-ckpt runs/bpe/ar/ckpt_ar.pt \
                 --n-samples "$EVAL_NSAMPLES" --steps "$EVAL_STEPS" --out "$d/eval_table.md" --seed 0; }

[ -f data_bpe/train.bin ] || python prepare.py --dataset tinystories --tokenizer gpt2 --data-dir data_bpe --max-bytes "$MAXBYTES"
train_if_missing runs/bpe/ar/ckpt_ar.pt train_ar.py "${BPE[@]}" --seed 0 --out-dir runs/bpe/ar

run_arm(){  # $1=name, rest=schedule flags
  local name="$1"; shift
  for seed in $SEEDS; do
    local d="runs/bpe/${name}_s${seed}"
    echo "=== BPE arm=$name seed=$seed ==="
    train_if_missing "$d/ckpt_last.pt" train.py "${BPE[@]}" "$@" --seed "$seed" --out-dir "$d"
    do_eval "$d"
    python experiments/aggregate.py runs/bpe > runs/bpe/scoreboard.md || true   # live scoreboard
  done
}

run_arm uniform        --schedule uniform
run_arm rare_first     --schedule rare_first     --zipf-beta "$BETA"
run_arm frequent_first --schedule frequent_first --zipf-beta "$BETA"
run_arm match_rare     --schedule uniform --match-noise rare_first     --zipf-beta "$BETA"
run_arm match_freq     --schedule uniform --match-noise frequent_first --zipf-beta "$BETA"

echo "===== FINAL BPE scoreboard ====="; python experiments/aggregate.py runs/bpe | tee runs/bpe/scoreboard.md
echo "DONE -> runs/bpe/scoreboard.md"

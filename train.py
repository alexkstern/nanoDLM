"""Train a masked-diffusion language model on tiny Shakespeare.

The pedagogical payload is the eight-line training step in `loss_fn` below:
sample a per-example noise level t, mask each token independently with prob t,
predict the masked tokens, and weight the cross-entropy by 1/t.

Everything else is standard nanoGPT-style scaffolding.
"""
import argparse
import math
import os
import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch

from config import Config
from model import DLM
from sample import generate
from schedule import resolve_exponents, summarize, survival_and_weight

cfg = Config()

# --- CLI overrides -----------------------------------------------------------
# train.py runs as a flat script (Config() with no overrides). For sweeping the
# schedule / seed / scale from a shell driver we expose a thin argparse layer;
# anything not passed keeps its config.py default.
_p = argparse.ArgumentParser(description="Train a masked-diffusion LM (nanoDLM).")
_p.add_argument("--schedule", choices=["uniform", "rare_first", "frequent_first"],
                default=cfg.schedule, help="frequency-weighted masking schedule")
_p.add_argument("--zipf-beta", type=float, default=cfg.zipf_beta,
                help="schedule strength; 0 collapses to uniform")
_p.add_argument("--const-exp", type=float, default=cfg.const_exp,
                help="fixed exponent for every token (frequency-independent control)")
_p.add_argument("--match-noise", default=cfg.match_noise,
                choices=["", "rare_first", "frequent_first"],
                help="constant-exponent control matched to this arm's mean noise")
_p.add_argument("--seed", type=int, default=cfg.seed)
_p.add_argument("--max-steps", type=int, default=cfg.max_steps)
_p.add_argument("--out-dir", default=cfg.out_dir)
_p.add_argument("--data-dir", default=cfg.data_dir)
_p.add_argument("--batch-size", type=int, default=cfg.batch_size)
_p.add_argument("--block-size", type=int, default=cfg.block_size)
_p.add_argument("--n-layer", type=int, default=cfg.n_layer)
_p.add_argument("--n-head", type=int, default=cfg.n_head)
_p.add_argument("--n-embd", type=int, default=cfg.n_embd)
_p.add_argument("--lr", type=float, default=cfg.lr)
_p.add_argument("--eval-interval", type=int, default=cfg.eval_interval)
_p.add_argument("--sample-interval", type=int, default=cfg.sample_interval)
_p.add_argument("--device", default=cfg.device)
_p.add_argument("--no-compile", action="store_true", help="disable torch.compile")
_args = _p.parse_args()
for _k, _v in vars(_args).items():
    if _k == "no_compile":
        continue
    setattr(cfg, _k, _v)
if _args.no_compile:
    cfg.compile = False

torch.manual_seed(cfg.seed)
os.makedirs(cfg.out_dir, exist_ok=True)

# device / dtype ------------------------------------------------------------
if cfg.device == "cuda" and not torch.cuda.is_available():
    cfg.device = "mps" if torch.backends.mps.is_available() else "cpu"
device = cfg.device
dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
ptdtype = dtype_map[cfg.dtype if device == "cuda" else "float32"]
ctx = (torch.amp.autocast(device_type="cuda", dtype=ptdtype)
       if device == "cuda" else nullcontext())
print(f"device: {device}   dtype: {ptdtype}")

# data ----------------------------------------------------------------------
data_dir = os.path.join(os.path.dirname(__file__), cfg.data_dir)
with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
cfg.vocab_size = meta["vocab_size"]
itos = meta["itos"]
print(f"vocab size: {cfg.vocab_size} (+1 [MASK])")

train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

# Per-token-identity exponents for the frequency-weighted schedule (schedule.py).
# resolve_exponents applies the precedence (match_noise > const_exp > schedule)
# and returns None for the uniform / beta==0 baseline so loss_fn takes the
# byte-identical vanilla-MDLM code path.
_exp_np, _label, _counts, _ranks = resolve_exponents(
    cfg, os.path.join(data_dir, "train.bin"))
print(f"schedule: {_label}")
if _exp_np is not None:
    print(summarize(_exp_np, _counts, _ranks, itos))
    loss_exponents = torch.from_numpy(_exp_np).to(device)   # (vocab_size,)
else:
    loss_exponents = None


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + cfg.block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
    return x


# model ---------------------------------------------------------------------
model = DLM(cfg).to(device)
MASK_ID = model.mask_id
optimizer = model.configure_optimizers(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2))
if cfg.compile and device == "cuda":
    try:
        import triton  # noqa: F401
        print("compiling model...")
        model = torch.compile(model)
    except ImportError:
        print("triton not installed; skipping torch.compile (training will run, just slower)")


def loss_fn(model, x, self_cond_prob: float = 0.5, p_ar_mix: float = 0.0,
            exponents=None):
    """The whole pedagogical payload of nanoDLM.

    Self-conditioning (Chen et al. 2023): with probability self_cond_prob,
    do an extra no-grad forward to get the model's own argmax prediction
    and feed it back as idx_prev on the gradient-bearing forward. Otherwise
    idx_prev is None and the cond_proj path is a no-op (zero-init). At
    matched param count this is documented to lift MDM val by ~0.05-0.15
    nats. Cost: ~50% wallclock since half the batches do 2 forwards.

    DUO hybrid masking (Sahoo et al. 2024): with probability p_ar_mix,
    replace the random Bernoulli(t) mask with a contiguous-suffix mask
    so the model gets standard AR-style supervision (predict x[k:] from
    x[:k]). Same 1/t-weighted CE loss is applied — only the mask shape
    differs. Set p_ar_mix=0 to recover the pure-MDM training of LLaDA /
    MDLM / MD4. Set p_ar_mix=1 to train a pure AR with this codebase.

    Frequency-weighted schedule: if `exponents` is given (a (vocab,) tensor of
    per-token-identity exponents c_i from schedule.py), each token is masked
    with prob 1 - (1-t)**c_i instead of the uniform t, and weighted by the
    matching per-token absorbing-state ELBO weight
        w_i(t) = c_i * (1-t)**(c_i - 1) / (1 - (1-t)**c_i),
    which reduces to 1/t when c_i = 1. exponents=None takes the exact
    vanilla-MDLM path so the uniform baseline is bit-for-bit unchanged.
    """
    B, T = x.shape

    if torch.rand(()).item() < p_ar_mix:
        # Contiguous-suffix mask: per-example pivot k in [0, T-1], mask the
        # tail. At least one position is always masked so 1/t doesn't blow
        # up; the noise level t is just the fraction of masked positions.
        pivot = torch.randint(0, T, (B, 1), device=x.device)
        positions = torch.arange(T, device=x.device)[None, :]
        mask = positions >= pivot                                 # (B, T) bool
        t = mask.float().mean(dim=-1, keepdim=True).clamp(min=cfg.eps)
        weight = (1.0 / t).expand(B, T)
    elif exponents is None:
        # Standard absorbing-state MDM noising: every token masked w.p. t.
        t = torch.rand(B, 1, device=x.device).clamp(min=cfg.eps)
        mask = torch.rand(B, T, device=x.device) < t
        weight = (1.0 / t).expand(B, T)
    else:
        # Frequency-weighted noising. survival_i = (1-t)**c_i, so the per-token
        # mask prob is 1 - (1-t)**c_i and the per-token ELBO weight is
        # c_i*(1-t)**(c_i-1)/(1-(1-t)**c_i). Clamp t away from BOTH ends: the
        # lower clamp is the usual 1/t guard; the upper clamp bounds the
        # (1-t)**(c_i-1) factor when c_i < 1 (frequent tokens at high noise).
        t = torch.rand(B, 1, device=x.device).clamp(cfg.eps, 1.0 - cfg.eps_hi)
        c = exponents[x]                                          # (B, T)
        surv, weight = survival_and_weight(t, c)                  # both (B, T)
        mask = torch.rand(B, T, device=x.device) < (1.0 - surv)

    x_t = torch.where(mask, MASK_ID, x)                           # corrupt

    idx_prev = None
    if torch.rand(()).item() < self_cond_prob:
        # No-grad first pass produces the self-conditioning input. argmax is
        # safe because logits[..., MASK_ID] is -inf so MASK is never picked.
        with torch.no_grad():
            idx_prev = model(x_t).argmax(dim=-1)

    logits = model(x_t, idx_prev)                              # (B, T, V_with_mask)
    loss_tok = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        x.reshape(-1),
        reduction="none",
    ).view(B, T)
    # Sahoo et al. 2024 §3 ELBO: divide by the deterministic constant B*T,
    # not by the stochastic mask.sum(). With the stochastic denominator the
    # reported loss is ~2H instead of H, the small-t signal gets diluted by
    # heavily-masked siblings in the same batch, and per-batch gradient scale
    # depends on the random mask draw. The proper estimator below is the
    # standard MDM-NELBO and converges to nats/char.
    return (loss_tok * mask * weight).sum() / (B * T)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            with ctx:
                # Disable self-conditioning AND DUO hybrid at eval so val
                # loss is directly comparable to pure-MDM runs. The deployment
                # behaviour (self-cond on, hybrid mask off for sampling) is
                # what `sample.py` exercises.
                losses[k] = loss_fn(model, get_batch(split),
                                    self_cond_prob=0.0, p_ar_mix=0.0)
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(step):
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.1 + 0.5 * (cfg.lr - cfg.lr * 0.1) * (1 + math.cos(math.pi * progress))


# training loop -------------------------------------------------------------
# Save the best-val checkpoint (not just the last one) — matches train_ar.py
# and means `out/ckpt.pt` is always the model `eval.py` should compare.
print(f"training for {cfg.max_steps} steps")
t0 = time.time()
best_val = float("inf")
ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
for step in range(cfg.max_steps + 1):
    lr = get_lr(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    if step % cfg.eval_interval == 0:
        losses = estimate_loss()
        dt = time.time() - t0
        flag = ""
        if losses["val"] < best_val:
            best_val = losses["val"]
            m = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({"model": m.state_dict(), "config": cfg.__dict__,
                        "step": step, "val": best_val}, ckpt_path)
            flag = "  *best, saved"
        print(f"step {step:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} | "
              f"lr {lr:.2e} | {dt:6.1f}s{flag}")

    if step > 0 and step % cfg.sample_interval == 0:
        print("--- sample (steps=64) ---")
        m = model._orig_mod if hasattr(model, "_orig_mod") else model
        out = generate(m, length=128, steps=64, device=device, verbose=False)
        print("".join(itos[int(i)] for i in out[0].tolist()))
        print("-------------------------")

    if step == cfg.max_steps:
        break

    x = get_batch("train")
    with ctx:
        loss = loss_fn(model, x, p_ar_mix=cfg.p_ar_mix, exponents=loss_exponents)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

# Also save the FINAL-step checkpoint, selection-free. For the schedule study we
# eval this one: best-val selection uses the uniform-ELBO criterion, which is the
# uniform arm's own training objective and would hand it a home-field advantage
# (and the non-uniform arms an off-objective stopping rule). ckpt_last.pt is the
# same fixed criterion for every arm. ckpt.pt (best-val) is kept for parity with
# the upstream repo / AR baseline.
m = model._orig_mod if hasattr(model, "_orig_mod") else model
last_path = os.path.join(cfg.out_dir, "ckpt_last.pt")
torch.save({"model": m.state_dict(), "config": cfg.__dict__,
            "step": cfg.max_steps, "val": best_val}, last_path)
print(f"best val: {best_val:.3f} (best -> {ckpt_path}, final -> {last_path})")

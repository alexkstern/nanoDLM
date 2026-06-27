"""Shared eval harness — the AR vs MDM head-to-head.

Loads both checkpoints (out/ckpt.pt for MDM, out/ckpt_ar.pt for AR) and
runs four metrics, then prints a markdown table you can paste straight
into the README:

    Val NLL                     character-level negative log-likelihood
                                on the validation split. For AR this is
                                the model's natural objective. For MDM
                                this is the absorbing-state ELBO, which
                                is a *bound* on NLL; we mark it as ≤.
                                Lower is better.

    Sample PPL-under-AR         draw N samples from each model and score
                                them with the AR model. Same scorer for
                                both, so it is a clean comparison of
                                "how on-distribution do your samples
                                look?". Lower is better.

    Distinct-2 / Distinct-3     bigram and trigram type-token ratios on
                                the concatenated samples. Higher means
                                more diverse. Mode-collapsed models
                                score low here.

    Infill recovery             mask a known 20-char substring from a
                                random val example, ask the model to
                                infill, count chars recovered exactly.
                                Only MDM can do this; AR records N/A.
                                Higher is better.

Usage:
    python train.py       # produces out/ckpt.pt    (MDM, ~21 min)
    python train_ar.py    # produces out/ckpt_ar.pt (AR,  ~14 min)
    python eval.py
"""
import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from model import DLM
from model_ar import ARLM
from sample import generate, generate_blockwise

# Resolve the compute device once. On a CUDA box this is "cuda" (identical to
# the original hardcoded behaviour); off-CUDA it falls back to CPU so the eval
# harness can be dry-run locally. MPS is skipped here because a couple of the
# metric ops (kthvalue / multinomial paths) are flaky on it.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_mdm(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["config"])
    model = DLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_ar(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["config"])
    model = ARLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


# ---------- metrics ---------------------------------------------------------

@torch.no_grad()
def ar_val_nll(model, val_data: np.ndarray, cfg: Config, n_batches: int = 100) -> float:
    """Char-level NLL on val. True NLL for AR; not used for MDM here."""
    losses = []
    for _ in range(n_batches):
        ix = torch.randint(len(val_data) - cfg.block_size - 1, (cfg.batch_size,))
        x = torch.stack([torch.from_numpy(val_data[i:i+cfg.block_size].astype(np.int64)) for i in ix]).to(DEVICE)
        y = torch.stack([torch.from_numpy(val_data[i+1:i+1+cfg.block_size].astype(np.int64)) for i in ix]).to(DEVICE)
        _, loss = model(x, y)
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def mdm_val_elbo(model, val_data: np.ndarray, cfg: Config, eps: float = 1e-3,
                 n_batches: int = 100) -> float:
    """Absorbing-state ELBO (upper bound on NLL) on val. Mirrors train.loss_fn."""
    MASK_ID = model.mask_id
    losses = []
    for _ in range(n_batches):
        ix = torch.randint(len(val_data) - cfg.block_size, (cfg.batch_size,))
        x = torch.stack([torch.from_numpy(val_data[i:i+cfg.block_size].astype(np.int64)) for i in ix]).to(DEVICE)
        B, T = x.shape
        t = torch.rand(B, 1, device=x.device).clamp(min=eps)
        mask = torch.rand(B, T, device=x.device) < t
        x_t = torch.where(mask, MASK_ID, x)
        logits = model(x_t)
        loss_tok = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), x.reshape(-1),
            reduction="none",
        ).view(B, T)
        loss = (loss_tok * mask / t).sum() / (B * T)
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def mdm_val_elbo_own(model, val_data: np.ndarray, cfg: Config, exponents,
                     eps: float = 1e-3, eps_hi: float = 1e-4,
                     n_batches: int = 100) -> float:
    """Own-schedule ELBO: the bound under the arm's OWN per-token masking, the
    tightest self-consistent NLL bound for a non-uniformly-trained model. With
    exponents=None this equals the uniform ELBO. Compared against the common
    uniform ELBO it separates a true-NLL gap from mere bound-looseness."""
    if exponents is None:
        return mdm_val_elbo(model, val_data, cfg, eps=eps, n_batches=n_batches)
    from schedule import survival_and_weight
    MASK_ID = model.mask_id
    c_full = torch.from_numpy(exponents).to(DEVICE)
    losses = []
    for _ in range(n_batches):
        ix = torch.randint(len(val_data) - cfg.block_size, (cfg.batch_size,))
        x = torch.stack([torch.from_numpy(val_data[i:i+cfg.block_size].astype(np.int64)) for i in ix]).to(DEVICE)
        B, T = x.shape
        t = torch.rand(B, 1, device=x.device).clamp(eps, 1.0 - eps_hi)
        c = c_full[x]
        surv, weight = survival_and_weight(t, c)
        mask = torch.rand(B, T, device=x.device) < (1.0 - surv)
        x_t = torch.where(mask, MASK_ID, x)
        logits = model(x_t)
        loss_tok = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), x.reshape(-1),
            reduction="none").view(B, T)
        losses.append(((loss_tok * mask * weight).sum() / (B * T)).item())
    return float(np.mean(losses))


@torch.no_grad()
def score_with_ar(ar_model, text_ids: torch.Tensor, cfg: Config) -> float:
    """Per-char NLL of `text_ids` under the AR model. Lower = more on-distribution."""
    if text_ids.size(1) < 2:
        return float("nan")
    x = text_ids[:, :-1].to(DEVICE)
    y = text_ids[:, 1:].to(DEVICE)
    _, loss = ar_model(x, y)
    return float(loss.item())


@torch.no_grad()
def sample_ppl_under_ar(model, ar_model, cfg: Config, n_samples: int = 16,
                        length: int = 256, steps: int = 64, is_mdm: bool = True,
                        temperature: float = 1.0, top_p: float = 0.9,
                        schedule: str = "linear") -> float:
    """Generate n_samples from `model` and score them under `ar_model`. PPL = exp(NLL)."""
    all_ids = []
    if is_mdm:
        for _ in range(n_samples):
            out = generate(model, length=length, steps=steps, temperature=temperature,
                           top_p=top_p, schedule=schedule, device=DEVICE)
            all_ids.append(out)
    else:
        # AR sampling
        ar = model
        for _ in range(n_samples):
            x = torch.tensor([[0]], dtype=torch.long, device=DEVICE)  # prompt = '\n'
            for _ in range(length - 1):
                idx_cond = x[:, -cfg.block_size:]
                logits = ar(idx_cond)[:, -1, :] / temperature
                probs = logits.softmax(-1)
                if top_p < 1.0:
                    sp, si = probs.sort(dim=-1, descending=True)
                    cum = sp.cumsum(dim=-1)
                    drop = (cum - sp) > top_p
                    sp = sp.masked_fill(drop, 0.0)
                    probs = torch.zeros_like(probs).scatter_(-1, si, sp)
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                nxt = torch.multinomial(probs, 1)
                x = torch.cat([x, nxt], dim=1)
            all_ids.append(x)

    all_ids = torch.cat(all_ids, dim=0)              # (n_samples, length)
    nlls = []
    for i in range(all_ids.size(0)):
        nll = score_with_ar(ar_model, all_ids[i:i+1], cfg)
        if not np.isnan(nll):
            nlls.append(nll)
    return float(np.exp(np.mean(nlls)))


def distinct_n(token_lists, n: int) -> float:
    """type-token ratio of n-grams across all token sequences."""
    grams = []
    for toks in token_lists:
        grams.extend(tuple(toks[i:i+n]) for i in range(len(toks) - n + 1))
    if not grams:
        return float("nan")
    return len(set(grams)) / len(grams)


@torch.no_grad()
def diversity(model, cfg: Config, n_samples: int = 16, length: int = 256, steps: int = 64,
              is_mdm: bool = True, temperature: float = 1.0, top_p: float = 0.9):
    """distinct-2 and distinct-3 over `n_samples` generations."""
    seqs = []
    for _ in range(n_samples):
        if is_mdm:
            out = generate(model, length=length, steps=steps, temperature=temperature,
                           top_p=top_p, device=DEVICE)
        else:
            x = torch.tensor([[0]], dtype=torch.long, device=DEVICE)
            for _ in range(length - 1):
                idx_cond = x[:, -cfg.block_size:]
                logits = model(idx_cond)[:, -1, :] / temperature
                probs = logits.softmax(-1)
                if top_p < 1.0:
                    sp, si = probs.sort(dim=-1, descending=True)
                    cum = sp.cumsum(dim=-1)
                    drop = (cum - sp) > top_p
                    sp = sp.masked_fill(drop, 0.0)
                    probs = torch.zeros_like(probs).scatter_(-1, si, sp)
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                nxt = torch.multinomial(probs, 1)
                x = torch.cat([x, nxt], dim=1)
            out = x
        seqs.append(out[0].tolist())
    return distinct_n(seqs, 2), distinct_n(seqs, 3)


@torch.no_grad()
def infill_recovery(model, val_data: np.ndarray, cfg: Config, n_trials: int = 32,
                    span_len: int = 20, total_len: int = 128, steps: int = 64,
                    temperature: float = 1.0, top_p: float = 0.9) -> float:
    """For each trial, sample a val window, mask a known span in the middle,
    let the model infill, and report the average per-char recovery rate."""
    recoveries = []
    for _ in range(n_trials):
        i = int(torch.randint(0, len(val_data) - total_len, ()).item())
        gold = torch.from_numpy(val_data[i:i+total_len].astype(np.int64)).to(DEVICE)
        x_init = gold.clone()
        # Mask a span in the middle
        start = (total_len - span_len) // 2
        x_init[start:start+span_len] = model.mask_id
        x_init = x_init[None]                         # (1, total_len)
        out = generate(model, x_init=x_init, steps=steps, temperature=temperature,
                       top_p=top_p, device=DEVICE)
        recovered = (out[0, start:start+span_len] == gold[start:start+span_len]).float().mean().item()
        recoveries.append(recovered)
    return float(np.mean(recoveries))


# ---------- main ------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mdm-ckpt", default="out/ckpt.pt")
    p.add_argument("--ar-ckpt", default="out/ckpt_ar.pt")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-batches", type=int, default=100)
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0,
                   help="torch seed for the stochastic sampling metrics")
    p.add_argument("--out", default="out/eval_table.md")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    device = DEVICE
    for path in (args.mdm_ckpt, args.ar_ckpt):
        if not os.path.exists(path):
            raise SystemExit(f"missing checkpoint: {path}. "
                             f"Train both models first (train.py / train_ar.py).")

    print("loading MDM...")
    mdm, mdm_cfg = load_mdm(args.mdm_ckpt, device)
    print("loading AR...")
    ar, ar_cfg = load_ar(args.ar_ckpt, device)
    assert mdm_cfg.vocab_size == ar_cfg.vocab_size, "vocab mismatch — re-prepare"

    data_dir = os.path.join(os.path.dirname(__file__), mdm_cfg.data_dir)
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    print("\n[1/5] val NLL / ELBO (uniform common bound + own-schedule) ...")
    mdm_elbo = mdm_val_elbo(mdm, val_data, mdm_cfg, eps=mdm_cfg.eps, n_batches=args.n_batches)
    ar_nll = ar_val_nll(ar, val_data, ar_cfg, n_batches=args.n_batches)
    # Own-schedule ELBO: rebuild this arm's exponents from its training corpus
    # (same precedence as train.py) and bound under its own masking. Equals the
    # uniform ELBO for the uniform arm.
    from schedule import resolve_exponents
    _own_exp, _own_label, _, _ = resolve_exponents(
        mdm_cfg, os.path.join(data_dir, "train.bin"))
    mdm_elbo_own = mdm_val_elbo_own(
        mdm, val_data, mdm_cfg, _own_exp, eps=mdm_cfg.eps,
        eps_hi=getattr(mdm_cfg, "eps_hi", 1e-4), n_batches=args.n_batches)
    print(f"    uniform-ELBO={mdm_elbo:.3f}   own-ELBO[{_own_label}]={mdm_elbo_own:.3f}")

    print("[2/5] sample PPL-under-AR (MDM samples, schedule sweep) ...")
    mdm_ppl_by_sched = {}
    for sched in ("linear", "cosine", "cosine_inv"):
        mdm_ppl_by_sched[sched] = sample_ppl_under_ar(
            mdm, ar, mdm_cfg, n_samples=args.n_samples,
            length=args.length, steps=args.steps, is_mdm=True,
            temperature=args.temperature, top_p=args.top_p, schedule=sched)
        print(f"    {sched}: {mdm_ppl_by_sched[sched]:.2f}")
    mdm_ppl = mdm_ppl_by_sched["linear"]
    print("[3/5] sample PPL-under-AR (AR samples) ...")
    ar_ppl = sample_ppl_under_ar(ar, ar, ar_cfg, n_samples=args.n_samples,
                                 length=args.length, steps=args.steps, is_mdm=False,
                                 temperature=args.temperature, top_p=args.top_p)

    print("[4/5] diversity (distinct-2, distinct-3) ...")
    mdm_d2, mdm_d3 = diversity(mdm, mdm_cfg, n_samples=args.n_samples, length=args.length,
                               steps=args.steps, is_mdm=True,
                               temperature=args.temperature, top_p=args.top_p)
    ar_d2, ar_d3 = diversity(ar, ar_cfg, n_samples=args.n_samples, length=args.length,
                             steps=args.steps, is_mdm=False,
                             temperature=args.temperature, top_p=args.top_p)

    print("[5/5] infill recovery (MDM only) ...")
    mdm_inf = infill_recovery(mdm, val_data, mdm_cfg, n_trials=args.n_samples,
                              steps=args.steps, temperature=args.temperature, top_p=args.top_p)

    print("[6/6] block-wise sampling sweep (same total NFE) ...")
    blockwise_results = {}
    # All entries have total NFE = args.steps (e.g. 64). Block-len=length means
    # equivalent to full-seq (the same generate() codepath), shown as a baseline.
    for block_len, sub_steps in [(args.length, args.steps),
                                 (args.length // 2, args.steps // 2),
                                 (args.length // 4, args.steps // 4),
                                 (args.length // 8, args.steps // 8)]:
        if sub_steps < 1:
            continue
        nlls = []
        for _ in range(args.n_samples):
            out = generate_blockwise(mdm, length=args.length,
                                     block_len=block_len,
                                     steps_per_block=sub_steps,
                                     device=DEVICE, temperature=args.temperature,
                                     top_p=args.top_p)
            nll = score_with_ar(ar, out, mdm_cfg)
            if not np.isnan(nll):
                nlls.append(nll)
        blockwise_results[(block_len, sub_steps)] = float(np.exp(np.mean(nlls)))
        n_blocks = (args.length + block_len - 1) // block_len
        print(f"    block_len={block_len}, steps={sub_steps} "
              f"({n_blocks} blocks, NFE={n_blocks*sub_steps}): "
              f"PPL={blockwise_results[(block_len, sub_steps)]:.2f}")

    # ----- render headline table -----
    rows = [
        ("Sample PPL under AR scorer (PRIMARY, schedule-independent, lower better)",
         f"{mdm_ppl:.2f}",
         f"{ar_ppl:.2f}"),
        ("Val char NLL uniform-ELBO (common bound; favors uniform arm)",
         f"<= {mdm_elbo:.3f} (ELBO)",
         f"{ar_nll:.3f}"),
        ("Val char NLL own-schedule ELBO (each arm's own bound)",
         f"<= {mdm_elbo_own:.3f} (ELBO)",
         "N/A"),
        ("Sample distinct-2 (higher = more diverse)",
         f"{mdm_d2:.3f}", f"{ar_d2:.3f}"),
        ("Sample distinct-3",
         f"{mdm_d3:.3f}", f"{ar_d3:.3f}"),
        (f"Infill recovery @ span=20, NFE={args.steps} (higher better)",
         f"{mdm_inf:.3f}",
         "N/A (causal AR cannot infill)"),
    ]
    table = ["| Metric | MDM | AR |", "|---|---|---|"]
    table += [f"| {m} | {a} | {b} |" for (m, a, b) in rows]
    md = "\n".join(table)

    # ----- inference-schedule sweep table (the SAMPLER's commit schedule on a
    # single checkpoint — NOT the training/masking Zipf schedule) -----
    best = min(mdm_ppl_by_sched, key=mdm_ppl_by_sched.get)
    sched_rows = ["| Inference (denoising) schedule | PPL under AR | delta vs linear |", "|---|---|---|"]
    base = mdm_ppl_by_sched["linear"]
    for s in ("linear", "cosine", "cosine_inv"):
        ppl = mdm_ppl_by_sched[s]
        delta = (ppl - base) / base * 100 if base > 0 else 0.0
        marker = " <-- best" if s == best else ""
        sched_rows.append(f"| {s} | {ppl:.2f}{marker} | {delta:+.1f}% |")
    sched_md = "\n".join(sched_rows)

    # ----- block-wise sweep table -----
    bw_rows = ["| Block setup | NFE total | PPL under AR |", "|---|---|---|"]
    for (block_len, sub_steps), ppl in blockwise_results.items():
        n_blocks = (args.length + block_len - 1) // block_len
        if n_blocks == 1:
            label = f"full-seq (no blocks), {sub_steps} steps"
        else:
            label = f"{n_blocks} blocks x {sub_steps} steps (block_len={block_len})"
        bw_rows.append(f"| {label} | {n_blocks * sub_steps} | {ppl:.2f} |")
    bw_md = "\n".join(bw_rows)

    print("\n" + "=" * 72)
    print(md)
    print("\n" + sched_md)
    print("\n" + bw_md)
    print("=" * 72)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# nanoDLM eval: MDM vs AR\n\n")
        f.write(md + "\n\n")
        f.write("## MDM inference (denoising) schedule sweep (PPL under AR scorer)\n\n")
        f.write(sched_md + "\n\n")
        f.write("## MDM block-wise sampling sweep (Mercury-style semi-AR)\n\n")
        f.write(bw_md + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

"""Frequency-weighted (Zipf) masking schedules for masked-diffusion training.

Standard MDLM masks every token with the SAME probability 1 - alpha_t at noise
level t. This module replaces that with a PER-TOKEN schedule whose rate depends
on how frequent the token's identity is in the corpus, so that the order in
which information is destroyed follows the token-frequency ("spectral") axis.

The single knob is a per-token-identity exponent c_i applied to the survival
probability:

    survival_i(t) = alpha_t ** c_i ,     alpha_t = 1 - t   (linear schedule)
    mask_prob_i(t) = 1 - alpha_t ** c_i

c_i = 1 recovers vanilla MDLM (survival = 1 - t, mask prob = t). c_i > 1 means
the token is absorbed FASTER (masked earlier as t grows); c_i < 1 means it
survives LONGER. Writing the rate as an exponent (not a multiplier) guarantees
survival -> 0 as t -> 1 for every token, so the forward process always reaches a
fully-masked "blank canvas" regardless of frequency. That was the original
sticking point with a naive multiplicative frequency weighting.

The matching continuous-time ELBO weight for token i is

    w_i(t) = -d/dt[survival_i] / (1 - survival_i)
           = c_i * alpha_t**(c_i - 1) / (1 - alpha_t**c_i)

which reduces to the familiar 1/t when c_i = 1. Derivation: the absorbing-state
NELBO factorises over independent token positions, so each position carries its
own single-token MDLM term (Sahoo et al. 2024 / Shi et al. 2024, applied
per-coordinate). train.loss_fn uses exactly this weight.

Direction conventions (u = normalised log-rank in [0, 1], 0 = most frequent,
1 = rarest):

    rare_first      c_i = exp(beta * (u - 0.5))   rare tokens absorbed first
                    (the spectral analog: rare = low-"power" = degraded first)
    frequent_first  c_i = exp(beta * (0.5 - u))   frequent tokens absorbed first
                    (keep the rare/content tokens as the surviving skeleton)
    uniform         c_i = 1                        vanilla MDLM

beta is the strength; beta = 0 collapses every schedule back to uniform.
"""
from __future__ import annotations

import numpy as np


VALID_SCHEDULES = ("uniform", "rare_first", "frequent_first")


def survival_and_weight(t, c):
    """Per-token survival and the matching continuous-time MDLM ELBO weight.

    survival_i(t) = (1 - t) ** c_i
    weight_i(t)   = c_i * (1 - t) ** (c_i - 1) / (1 - (1 - t) ** c_i)

    Operator-based so it works with either numpy arrays or torch tensors and
    broadcasts t of shape (B, 1) against c of shape (B, T). With c = 1 this is
    the vanilla MDLM survival 1 - t and weight 1 / t.

    Exact invariant (the reason this is a fair reweighting and a valid ELBO):

        integral_0^1 weight_i(t) * maskprob_i(t) dt
          = integral_0^1 c_i (1 - t) ** (c_i - 1) dt = 1     for every c_i > 0,

    so each token's EXPECTED total loss weight is exactly 1 regardless of its
    frequency. The schedule changes *at which noise level* a token is
    supervised, not the total amount of supervision it receives.

    NOTE: this holds exactly in continuous time. train.loss_fn clamps t to
    [eps, 1-eps_hi], which trims the high-noise tail of the integral; for c < 1
    that slightly UNDER-counts (deficit ~ (eps_hi)**c at the smallest c), so the
    realised expected weight is ~1 (within ~2% at the default beta=2, eps_hi=1e-4)
    rather than exactly 1. The deficit grows with beta and is symmetric across
    the rare_first/frequent_first arms (same c multiset), so it does not bias
    their head-to-head; report it when pushing beta high.
    """
    alpha = 1.0 - t
    surv = alpha ** c
    weight = c * alpha ** (c - 1.0) / (1.0 - surv)
    return surv, weight


def corpus_ranks(train_bin_path: str, vocab_size: int):
    """Count token ids in a uint16 .bin corpus and return (counts, ranks).

    counts[i] = number of occurrences of token id i in the corpus.
    ranks[i]  = frequency rank of token id i, 1 = most frequent. Ties are
    broken by token id (stable sort); tokens that never appear (count 0) land
    at the rarest end, which is the desired fallback for OOV / val-only ids.
    """
    data = np.memmap(train_bin_path, dtype=np.uint16, mode="r")
    counts = np.bincount(np.asarray(data, dtype=np.int64), minlength=vocab_size)
    counts = counts[:vocab_size].astype(np.float64)
    # argsort descending by count; stable so equal-count ids keep id order.
    order = np.argsort(-counts, kind="stable")
    ranks = np.empty(vocab_size, dtype=np.float64)
    ranks[order] = np.arange(1, vocab_size + 1, dtype=np.float64)
    return counts, ranks


def exponents_from_ranks(ranks: np.ndarray, schedule: str, beta: float) -> np.ndarray:
    """Map frequency ranks (1 = most frequent) to per-token exponents c_i."""
    if schedule not in VALID_SCHEDULES:
        raise ValueError(f"schedule must be one of {VALID_SCHEDULES}, got {schedule!r}")
    vocab_size = len(ranks)
    if schedule == "uniform" or beta == 0.0:
        return np.ones(vocab_size, dtype=np.float32)

    # Normalised log-rank in [0, 1]: 0 for the most frequent token (rank 1,
    # log 1 = 0), 1 for the rarest (rank = vocab_size). log-rank rather than
    # raw rank keeps the Zipf head from dominating the whole [0, 1] range.
    denom = np.log(vocab_size) if vocab_size > 1 else 1.0
    u = np.log(ranks) / denom

    if schedule == "rare_first":
        c = np.exp(beta * (u - 0.5))
    else:  # frequent_first
        c = np.exp(beta * (0.5 - u))
    return c.astype(np.float32)


def matched_constant_exponent(counts: np.ndarray, target_c: np.ndarray) -> float:
    """Constant exponent c* whose corpus-weighted, time-integrated mask rate
    matches that of a per-token schedule `target_c`.

    For constant c, the time-integrated (t~U[0,1]) mask rate of a token is
    integral_0^1 (1-(1-t)**c) dt = c/(c+1). The corpus-weighted target rate is
    M = sum_i p_i * c_i/(c_i+1). Inverting c/(c+1)=M gives c* = M/(1-M). A model
    trained with this single c* for EVERY token sees the same average noise as
    the target arm but applies ZERO frequency ordering (pure global time-warp),
    which is the clean control for "was it the ordering or just the noise level".
    """
    p = counts / max(counts.sum(), 1.0)
    M = float((p * (target_c / (target_c + 1.0))).sum())
    M = min(max(M, 1e-6), 1.0 - 1e-6)
    return M / (1.0 - M)


def resolve_exponents(cfg, train_bin_path: str):
    """Single source of truth for which exponent vector an arm uses, shared by
    train.py and eval.py so they never drift. Precedence:

        match_noise  -> constant c* matched to that schedule's mean noise (control)
        const_exp>0  -> a fixed constant exponent (manual control)
        uniform / zipf_beta==0 -> None (vanilla MDLM, the exact baseline path)
        else         -> the rare_first / frequent_first frequency schedule

    Returns (exponents_or_None, label, counts, ranks). exponents is float32 of
    length vocab_size, or None to signal the byte-identical vanilla code path.
    """
    counts, ranks = corpus_ranks(train_bin_path, cfg.vocab_size)
    match = getattr(cfg, "match_noise", "") or ""
    const = float(getattr(cfg, "const_exp", 0.0) or 0.0)
    if match:
        target = exponents_from_ranks(ranks, match, cfg.zipf_beta)
        cstar = matched_constant_exponent(counts, target)
        c = np.full(cfg.vocab_size, cstar, dtype=np.float32)
        return c, f"match_{match}(c*={cstar:.3f}, beta={cfg.zipf_beta})", counts, ranks
    if const > 0:
        c = np.full(cfg.vocab_size, const, dtype=np.float32)
        return c, f"const(c={const})", counts, ranks
    if cfg.schedule == "uniform" or cfg.zipf_beta == 0.0:
        return None, "uniform", counts, ranks
    c = exponents_from_ranks(ranks, cfg.schedule, cfg.zipf_beta)
    return c, f"{cfg.schedule}(beta={cfg.zipf_beta})", counts, ranks


def build_exponents(train_bin_path: str, vocab_size: int, schedule: str,
                    beta: float):
    """Convenience: corpus_ranks + exponents_from_ranks.

    Returns (exponents, counts, ranks). exponents is float32, length vocab_size,
    indexable by token id. Tokens never appear as the [MASK] id, so no exponent
    is needed for it.
    """
    counts, ranks = corpus_ranks(train_bin_path, vocab_size)
    c = exponents_from_ranks(ranks, schedule, beta)
    return c, counts, ranks


def summarize(c: np.ndarray, counts: np.ndarray, ranks: np.ndarray,
              itos: dict | None = None, k: int = 8) -> str:
    """A short human-readable report of the exponent assignment, for logging."""
    lines = [
        f"exponents: min={c.min():.3f} max={c.max():.3f} "
        f"geo-mean={np.exp(np.log(np.clip(c, 1e-9, None)).mean()):.3f} "
        f"arith-mean={c.mean():.3f}",
    ]

    def label(i: int) -> str:
        if itos is None:
            return f"id{i}"
        ch = itos.get(i, f"id{i}")
        return repr(ch) if isinstance(ch, str) else str(ch)

    most = np.argsort(ranks)[:k]          # rank 1..k = most frequent
    rare = np.argsort(-ranks)[:k]         # rarest
    lines.append("  most frequent: " + ", ".join(
        f"{label(int(i))}:c={c[i]:.2f}" for i in most))
    lines.append("  rarest:        " + ", ".join(
        f"{label(int(i))}:c={c[i]:.2f}" for i in rare))

    # Corpus-frequency-weighted effective noise: the average masked FRACTION a
    # real training batch sees at each t (weighting tokens by how often they
    # actually occur, NOT one-per-vocab-id). This is the honest "how much noise
    # is this arm really training at" number, and the main confound to watch:
    # arms that mask more on average are partly just training at higher noise.
    total = counts.sum()
    if total > 0:
        p = counts / total
        rates = []
        for tt in (0.1, 0.3, 0.5, 0.7, 0.9):
            surv = (1.0 - tt) ** c
            rates.append(float((p * (1.0 - surv)).sum()))
        lines.append("  corpus-weighted E[mask frac]: " + "  ".join(
            f"t={tt}:{r:.3f}" for tt, r in zip((0.1, 0.3, 0.5, 0.7, 0.9), rates)))
    return "\n".join(lines)

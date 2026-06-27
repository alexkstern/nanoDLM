from dataclasses import dataclass


@dataclass
class Config:
    # data
    data_dir: str = "data"
    block_size: int = 256
    vocab_size: int = 65          # set by prepare.py; +1 for [MASK] is added inside the model

    # model (~10M params at these settings on char-level)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False

    # optim
    batch_size: int = 64
    # MDMs need ~2-3x the compute of an AR at matched scale. 5000 steps is
    # ~82M tokens, below the AR-Chinchilla floor (20 tok/param × 10M = 200M)
    # and produces samples that have learned the unigram distribution but not
    # word-level structure. 20000 steps ≈ 330M tokens, where local English
    # structure starts to emerge on tiny Shakespeare.
    max_steps: int = 20000
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100

    # eval / logging
    eval_interval: int = 1000
    eval_iters: int = 50
    sample_interval: int = 2000

    # diffusion
    eps: float = 1e-3             # min mask ratio to avoid 1/t blowup (low-t clamp)
    eps_hi: float = 1e-4          # high-t clamp 1-eps_hi for the per-token weight;
                                  # smaller than eps so small-c tokens keep more of
                                  # their high-noise supervision (tail of the
                                  # equal-weight invariant). Only used by the
                                  # frequency schedule, not the uniform baseline.

    # frequency-weighted (Zipf) masking schedule. See schedule.py.
    #   uniform        -> vanilla MDLM, every token masked with prob t
    #   rare_first     -> rare tokens absorbed first (spectral analog)
    #   frequent_first -> frequent tokens absorbed first (rare/content survive)
    # zipf_beta is the strength; 0.0 collapses every schedule back to uniform.
    schedule: str = "uniform"
    zipf_beta: float = 2.0
    # Controls (resolve_exponents precedence: match_noise > const_exp > schedule):
    #   const_exp>0   -> every token gets this fixed exponent (no ordering)
    #   match_noise   -> constant exponent auto-tuned to match that arm's mean
    #                    corpus mask rate; the "was it ordering or just noise?" control
    const_exp: float = 0.0
    match_noise: str = ""

    # DUO-style hybrid training (Sahoo et al. 2024, "Diffusion Forcing for
    # Discrete Tokens"). With probability p_ar_mix per batch, replace the
    # random Bernoulli(t) mask with a contiguous-suffix mask: pick a per-
    # example pivot k and MASK all positions >= k. The intent is to give
    # the model AR-style supervision (predict x[k:] from x[:k]) on top of
    # the MDM objective, which recent papers claim closes most of the
    # MDM-vs-AR gap on val NLL while keeping infilling.
    #
    # NEGATIVE RESULT at this scale (10M params, char-level tiny shakespeare,
    # 20K steps): p_ar_mix=0.25 made every measured metric worse vs pure MDM:
    #   - val ELBO        1.617 -> 1.664 (+2.9%)
    #   - sample PPL/AR   3.35  -> 3.97  (+18%)
    #   - infill recovery 0.200 -> 0.144 (-28%)
    # We keep the code path as a one-line opt-in for users who want to try
    # other mixing rates or larger scales where the technique may help, but
    # the default is 0.0 (pure MDM).
    p_ar_mix: float = 0.0

    # system
    device: str = "cuda"          # falls back to cpu/mps in train.py
    dtype: str = "bfloat16"       # autocast dtype on cuda
    compile: bool = True
    seed: int = 1337
    out_dir: str = "out"

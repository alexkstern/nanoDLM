# Frequency-weighted ("Zipf") masking schedules for MDLM

This folder benchmarks a **per-token, frequency-weighted masking schedule**
against the vanilla uniform MDLM that nanoDLM already implements. The question:
does *which* tokens you destroy *when* — ordered by token frequency — train a
better (or meaningfully different) diffusion language model than masking every
token with the same probability?

## The idea in one paragraph

Vanilla MDLM masks every token with probability `t` at noise level `t`
(survival `1-t`). We instead give each token identity `i` an exponent `c_i`
derived from its corpus frequency rank, so its survival is `(1-t)**c_i` and its
mask probability is `1-(1-t)**c_i`. The matching continuous-time absorbing-state
ELBO weight is

```
w_i(t) = c_i * (1-t)**(c_i-1) / (1 - (1-t)**c_i)
```

which reduces to the familiar `1/t` when `c_i = 1`. This is a valid MDLM ELBO
because the objective factorises over independent token positions; each token
just carries its own single-token term. Writing the rate as an **exponent**
(not a multiplier) guarantees `survival -> 0` as `t -> 1`, so the forward
process always reaches a fully-masked canvas regardless of frequency.

**Exact invariant** (why it's a fair reweighting): for every `c_i`,
`integral_0^1 w_i(t)*maskprob_i(t) dt = 1`. Each token's *expected total* loss
weight is exactly 1 — the schedule changes *at which noise level* a token is
supervised, not the total supervision it gets.

Three arms (single strength knob `beta`, `beta=0` == uniform):

| arm | `c_i` | effect |
|---|---|---|
| `uniform` | `1` | vanilla MDLM (byte-identical baseline) |
| `rare_first` | `exp(beta*(u-0.5))` | rare tokens absorbed first (spectral analog) |
| `frequent_first` | `exp(beta*(0.5-u))` | frequent tokens first; rare/content survive |

`u = log(rank)/log(V)` in `[0,1]`, `rank=1` = most frequent.

## What changed vs upstream nanoDLM

- `schedule.py` (new) — exponent builder + `survival_and_weight`.
- `config.py` — `schedule`, `zipf_beta` fields.
- `train.py` / `train_ar.py` — per-token masking + weight in `loss_fn`; CLI
  overrides for sweeping schedule/seed/scale. **The uniform arm and all eval use
  the common uniform bound**, so checkpoint selection and cross-arm comparison
  are apples-to-apples.
- `prepare.py` — `--tokenizer {char,gpt2}` and `--data-dir`.
- `eval.py` — made device-portable (CUDA unchanged; CPU for local dry-runs).
- `experiments/` — run scripts, aggregator, and a correctness check.

## Run it

Fresh clone on the A100 (the run scripts stand up a uv venv themselves):

```sh
bash experiments/run_char.sh          # 3 arms x 3 seeds, char-level TinyStories
bash experiments/run_bpe.sh           # subword robustness run (gpt2 BPE)
```

Each writes a scoreboard to `runs/<exp>/scoreboard.md` (arm x metric, mean±std
across training seeds, plus delta-vs-uniform). Knobs are env vars, e.g.:

```sh
MAXSTEPS=20000 SEEDS="0 1 2" BETA=2.0 bash experiments/run_char.sh
BETA=1.0 OUTROOT=runs/char_b1 bash experiments/run_char.sh   # beta sweep
```

Local sanity (CPU, ~2 min, tiny everything — proves the pipeline, not science):

```sh
bash experiments/smoke.sh                 # fresh uv venv
SKIP_ENV=1 bash experiments/smoke.sh      # reuse the active venv
TOKENIZER=gpt2 bash experiments/smoke.sh  # exercise the BPE path
python experiments/check_schedule.py      # schedule math assertions only
```

Defaults: ~10M-param char model (6L/384d), 20k steps, TinyStories 50MB,
linear schedule, `beta=2.0`. On an A100-80GB expect ~10-15 min per model →
~2-2.5 h for the full char sweep (1 AR + 9 MDM + evals).

## Metrics (all computed identically for every arm)

- **sample PPL under an AR scorer** — **PRIMARY**. Fully schedule-independent
  (same sampler, same AR scorer, same NFE for every arm), so it does not favour
  any training schedule. Lead the "is Zipf a better generative model" claim with
  this.
- **val uniform-ELBO** — the standard MDLM absorbing-state bound under *uniform*
  masking. A valid bound for every arm and the common yardstick, **but it is the
  uniform arm's own training objective**, so it structurally favours uniform —
  report it, don't crown a winner by it. Quote it with seed error bars.
- **val own-schedule ELBO** — each arm's bound under *its own* masking. Compared
  with the uniform-ELBO it separates a true-NLL gap from mere bound-looseness.
- **distinct-2/3**, **infill recovery** — diversity and the MDM-only infill.

Checkpoint selection is **selection-free**: eval scores `ckpt_last.pt` (the
final-step checkpoint), the same fixed rule for every arm. The repo's best-val
`ckpt.pt` is also saved, but selecting by uniform-ELBO would hand the uniform arm
an extra home-field advantage and is an off-objective stop for the others.

## The caveat to keep honest

The arms do **not** train at identical average noise. Corpus-frequency-weighted
`E[mask fraction]` at `beta=2` (char TinyShakespeare): `frequent_first` masks
~5-7% more at mid/high `t`, `rare_first` ~5% less (the training log prints this
per run). So a raw win could be partly a "more/less effective noise" effect, not
the frequency *ordering*. Controls to disentangle, in priority order:

1. **beta sweep** (`BETA=0.5,1,2,4`) — `beta=0` is uniform; a monotone trend in
   `beta` argues for the ordering, not a single lucky noise level.
2. **mean-mask-rate-matched uniform** — a uniform arm retuned to each schedule's
   average mask rate, isolating ordering from total noise. (Not yet scripted;
   easiest as a follow-up arm.)

The **matched-noise control is scripted**: `CONTROLS=1 bash experiments/run_char.sh`
adds a constant-exponent arm per Zipf schedule (`--match-noise`), auto-tuned to
the same corpus-weighted mean mask rate but with zero frequency ordering. If a
Zipf arm beats its `match_*` control, the *ordering* helped; if not, it was the
noise level. Default off to keep the base run small — turn it on once the main
sweep shows an effect worth attributing. Report the effective-noise table (the
training log prints it) alongside every result.

```sh
CONTROLS=1 bash experiments/run_char.sh                    # + matched controls
for b in 0.5 1 2 4; do BETA=$b OUTROOT=runs/char_b$b bash experiments/run_char.sh; done  # beta sweep
```

## Note on the high-noise clamp

The per-token weight uses a two-sided clamp `t in [eps, 1-eps_hi]`
(`eps_hi=1e-4`). The equal-weight invariant (`int w_i*maskprob dt = 1`) is exact
in continuous time; the clamp trims the high-noise tail, so for small-`c`
(long-surviving) tokens the realised total weight is ~1 within ~2% at `beta=2`,
growing with `beta`. It is symmetric across the rare/frequent arms (same `c`
multiset) so it does not bias their head-to-head, but treat `beta>=4` results
with care and report the realised deficit.

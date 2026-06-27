"""Fast correctness checks for the frequency-weighted schedule (no training).

Run: python experiments/check_schedule.py
Exits nonzero on any failed assertion. Validates the parts that, if wrong,
would silently invalidate the whole experiment:

  1. survival/weight reduce to vanilla MDLM (1-t, 1/t) at c = 1.
  2. The exact ELBO invariant  integral_0^1 weight*maskprob dt = 1  for all c,
     i.e. every token's expected total loss weight is 1 regardless of frequency.
  3. The numpy and torch code paths agree.
  4. The exponent builder is monotonic in rank and points the right direction.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schedule import (survival_and_weight, exponents_from_ranks,  # noqa: E402
                      VALID_SCHEDULES)

ok = True


def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and bool(cond)


print("1) reduction to vanilla MDLM at c=1")
t = np.linspace(0.01, 0.99, 50)
surv, w = survival_and_weight(t, 1.0)
check("survival(c=1) == 1 - t", np.allclose(surv, 1 - t))
check("weight(c=1) == 1/t", np.allclose(w, 1.0 / t))

print("2) ELBO invariant: integral_0^1 weight(t,c)*maskprob(t,c) dt == 1")
# integrand = weight*maskprob = c*(1-t)^(c-1); integrate by midpoint on (0,1).
grid = (np.arange(1, 2_000_001) - 0.5) / 2_000_000          # midpoints in (0,1)
for c in (0.25, 0.5, 1.0, 2.0, 4.0):
    s, wt = survival_and_weight(grid, float(c))
    maskprob = 1.0 - s
    integrand = wt * maskprob
    integral = integrand.mean()                            # E_{t~U}[...] = integral
    # also confirm the pointwise algebra weight*maskprob == c*(1-t)^(c-1)
    closed = c * (1 - grid) ** (c - 1)
    # For c<1 the integrand is singular at t->1, so midpoint quadrature slightly
    # under-counts the tail (~1.5% at c=0.25). The invariant is EXACT — the
    # pointwise identity below plus the analytic antiderivative -(1-t)^c prove
    # integral == 1; the numeric value is a coarse sanity, hence the 2% band.
    check(f"c={c}: integral={integral:.4f} ~= 1", abs(integral - 1.0) < 0.02)
    check(f"c={c}: weight*maskprob == c(1-t)^(c-1)", np.allclose(integrand, closed))

print("3) numpy and torch paths agree")
try:
    import torch
    tc = torch.linspace(0.05, 0.95, 7)[:, None]            # (B,1)
    cc = torch.tensor([[0.3, 1.0, 3.0, 0.7]])              # (1,T) -> broadcast
    s_t, w_t = survival_and_weight(tc, cc)
    s_n, w_n = survival_and_weight(tc.numpy(), cc.numpy())
    check("survival numpy==torch", np.allclose(s_t.numpy(), s_n, atol=1e-5))
    check("weight numpy==torch", np.allclose(w_t.numpy(), w_n, atol=1e-5))
    check("all weights finite & positive", bool(torch.isfinite(w_t).all() and (w_t > 0).all()))
except ImportError:
    print("  [skip] torch not installed")

print("4) exponent builder direction & monotonicity (beta=2.0)")
V = 5000
ranks = np.arange(1, V + 1, dtype=np.float64)             # 1=most frequent
c_uniform = exponents_from_ranks(ranks, "uniform", 2.0)
c_rare = exponents_from_ranks(ranks, "rare_first", 2.0)
c_freq = exponents_from_ranks(ranks, "frequent_first", 2.0)
check("uniform -> all c==1", np.allclose(c_uniform, 1.0))
check("rare_first increasing in rank (rarer absorbed first)",
      bool(np.all(np.diff(c_rare) >= -1e-9)))
check("frequent_first decreasing in rank", bool(np.all(np.diff(c_freq) <= 1e-9)))
check("rare_first: most-freq token c<1 < rarest token c", c_rare[0] < 1.0 < c_rare[-1])
check("frequent_first: most-freq token c>1 > rarest token c", c_freq[0] > 1.0 > c_freq[-1])
check("rare_first and frequent_first are mirror images",
      np.allclose(c_rare * c_freq, 1.0, atol=1e-5))
check("beta=0 collapses to uniform",
      np.allclose(exponents_from_ranks(ranks, "rare_first", 0.0), 1.0))

# transparency: realized mean mask fraction per arm across t (not asserted)
print("\nrealized E[mask fraction] over a uniform token mix, by t:")
for label, c in (("uniform", c_uniform), ("rare_first", c_rare), ("frequent_first", c_freq)):
    fracs = []
    for tt in (0.1, 0.3, 0.5, 0.7, 0.9):
        s, _ = survival_and_weight(np.full_like(c, tt), c)
        fracs.append((1 - s).mean())
    print(f"  {label:14s} " + "  ".join(f"t={tt}:{f:.3f}"
          for tt, f in zip((0.1, 0.3, 0.5, 0.7, 0.9), fracs)))

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)

"""Aggregate per-(schedule, seed) eval tables into one scoreboard.

Reads <root>/<arm>_s<seed>/eval_table.md produced by eval.py and reports, for
each metric, the mean +/- std across TRAINING seeds per schedule arm, the AR
reference, the best arm, and the delta of each arm vs the uniform baseline.

    python experiments/aggregate.py runs/char

This is the training-seed analog of eval_multi.py (which varies only the
sampling seed on a single checkpoint).
"""
import glob
import os
import re
import sys

import numpy as np

# canonical metric key -> (substring to match in the row label, lower_is_better)
# Order = display order; the schedule-independent sample-PPL is the PRIMARY
# cross-arm metric. uniform-ELBO is a common bound that structurally favours the
# uniform arm, so it is reported but not treated as the headline; own-ELBO is
# each arm's own bound.
METRICS = [
    ("sample PPL/AR (PRIMARY)", "Sample PPL under AR", True),
    ("val uniform-ELBO (bound)", "uniform-ELBO", True),
    ("val own-sched ELBO",      "own-schedule ELBO", True),
    ("distinct-2",              "distinct-2", False),
    ("distinct-3",              "distinct-3", False),
    ("infill recovery",         "Infill recovery", False),
]
ARM_ORDER = ["uniform", "rare_first", "frequent_first"]


def parse_float(s):
    m = re.search(r"-?\d+\.\d+", s)
    return float(m.group()) if m else None


def parse_eval_table(path):
    """Return {canonical_metric: (mdm_float, ar_float)} from one eval_table.md."""
    out = {}
    in_headline = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("| Metric | MDM | AR |"):
                in_headline = True
                continue
            if in_headline and (not line.startswith("|") or line.startswith("##")):
                if not line.startswith("|"):
                    in_headline = False
                continue
            if in_headline and "---" in line:
                continue
            if in_headline and line.startswith("|"):
                parts = [p.strip() for p in line.strip("|").split("|")]
                if len(parts) != 3:
                    continue
                label, mdm, ar = parts
                for key, needle, _ in METRICS:
                    if needle.lower() in label.lower():
                        out[key] = (parse_float(mdm), parse_float(ar))
    return out


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "runs/char"
    # arm -> metric -> list of mdm floats (one per seed)
    per_arm = {}
    ar_ref = {k: [] for k, _, _ in METRICS}
    n_runs = 0
    for path in sorted(glob.glob(os.path.join(root, "*_s*", "eval_table.md"))):
        run = os.path.basename(os.path.dirname(path))
        arm, _, seed = run.rpartition("_s")
        if not seed.isdigit():
            continue
        n_runs += 1
        table = parse_eval_table(path)
        bucket = per_arm.setdefault(arm, {k: [] for k, _, _ in METRICS})
        for key, _, _ in METRICS:
            mdm, ar = table.get(key, (None, None))
            bucket[key].append(mdm)
            if ar is not None:
                ar_ref[key].append(ar)

    if not per_arm:
        print(f"no eval_table.md found under {root}/*_s*/")
        return

    arms = [a for a in ARM_ORDER if a in per_arm] + \
           [a for a in per_arm if a not in ARM_ORDER]

    # ---- scoreboard table ----
    header = ["Metric"] + arms + ["AR"]
    print(f"# scoreboard: {root}  ({n_runs} runs, {len(arms)} arms)\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for key, _, lower_better in METRICS:
        cells = [fmt(per_arm[a][key]) for a in arms]
        ar_cell = fmt(ar_ref[key]) if ar_ref[key] else "n/a"
        # mark the best arm (by mean) with *
        means = []
        for a in arms:
            vs = [v for v in per_arm[a][key] if v is not None]
            means.append(np.mean(vs) if vs else (np.inf if lower_better else -np.inf))
        best = (np.argmin if lower_better else np.argmax)(means)
        cells[best] = "**" + cells[best] + "**"
        arrow = "v" if lower_better else "^"
        print(f"| {key} ({arrow}) | " + " | ".join(cells) + f" | {ar_cell} |")

    # ---- deltas vs uniform ----
    if "uniform" in per_arm:
        print("\n## delta vs uniform baseline (negative = better)\n")
        print("| Metric | " + " | ".join(a for a in arms if a != "uniform") + " |")
        print("|" + "|".join(["---"] * (len([a for a in arms if a != 'uniform']) + 1)) + "|")
        for key, _, lower_better in METRICS:
            base = [v for v in per_arm["uniform"][key] if v is not None]
            if not base:
                continue
            base_m = np.mean(base)
            row = []
            for a in arms:
                if a == "uniform":
                    continue
                vs = [v for v in per_arm[a][key] if v is not None]
                if not vs or base_m == 0:
                    row.append("n/a")
                    continue
                rel = (np.mean(vs) - base_m) / abs(base_m) * 100.0
                # flip sign so negative always means "better than uniform"
                signed = rel if lower_better else -rel
                row.append(f"{signed:+.1f}%")
            print(f"| {key} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()

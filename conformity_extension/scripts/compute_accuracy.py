#!/usr/bin/env python3
"""
compute_accuracy.py

Computes Round-1 and Round-2 accuracy from a run_studies_base.py output
JSONL, broken down by condition. Also reports the existing revision-based
metrics (revision%, harmful%, beneficial%, mean_dC) alongside, so you get
the full picture in one table.

Usage:
    python compute_accuracy.py path/to/results.jsonl

Output columns:
    n             - number of rows in the condition
    acc_r1        - Round-1 accuracy (before peer prompt)
    acc_r2        - Round-2 accuracy (after peer prompt)
    delta_acc     - acc_r2 - acc_r1 (positive = peers helped overall)
    revision%     - % of rows where the answer changed at all
    harmful%      - % of initially-correct rows that flipped to wrong
    beneficial%   - % of initially-wrong rows that flipped to correct
    mean_dC       - mean confidence change (Round2 - Round1)
"""

import json
import sys
from collections import defaultdict


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute(rows):
    buckets = defaultdict(lambda: {
        "n": 0,
        "r1_correct": 0,
        "r2_correct": 0,
        "revised": 0,
        "correct_before_n": 0,
        "harmful": 0,
        "wrong_before_n": 0,
        "beneficial": 0,
        "delta_c": [],
    })
    skipped = 0

    for row in rows:
        gold = row["answer"]["label"]
        a0 = row.get("judgment_before")
        af = row.get("judgment_after")
        c0 = row.get("confidence_before")
        cf = row.get("confidence_after")
        cond = row.get("condition_id", "unknown")

        if any(v is None for v in [a0, af, c0, cf]):
            skipped += 1
            continue

        b = buckets[cond]
        b["n"] += 1

        r1_correct = (a0 == gold)
        r2_correct = (af == gold)

        if r1_correct:
            b["r1_correct"] += 1
        if r2_correct:
            b["r2_correct"] += 1

        if af != a0:
            b["revised"] += 1

        if r1_correct:
            b["correct_before_n"] += 1
            if not r2_correct:
                b["harmful"] += 1
        else:
            b["wrong_before_n"] += 1
            if r2_correct:
                b["beneficial"] += 1

        b["delta_c"].append(cf - c0)

    return buckets, skipped


def pct(num, denom):
    if denom == 0:
        return "N/A"
    return f"{100 * num / denom:.1f}"


def mean(vals):
    if not vals:
        return "N/A"
    return f"{sum(vals) / len(vals):.2f}"


def delta_acc(b):
    if b["n"] == 0:
        return "N/A"
    d = (b["r2_correct"] - b["r1_correct"]) / b["n"] * 100
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}"


def print_report(buckets, skipped):
    totals = defaultdict(int)
    all_dc = []

    header = (f"{'condition':<22} {'n':>4}  {'acc_r1':>6}  {'acc_r2':>6}  "
              f"{'Δacc':>6}  {'rev%':>5}  {'harm%':>6}  {'ben%':>5}  {'mean_dC':>7}")
    print(header)
    print("-" * len(header))

    for cond in sorted(buckets):
        b = buckets[cond]
        all_dc.extend(b["delta_c"])
        for k in ["n", "r1_correct", "r2_correct", "revised",
                  "correct_before_n", "harmful", "wrong_before_n", "beneficial"]:
            totals[k] += b[k]

        # harmful% only meaningful when not all_correct; beneficial% only when not all_wrong
        harm = pct(b["harmful"], b["correct_before_n"])
        ben  = pct(b["beneficial"], b["wrong_before_n"])
        if b["wrong_before_n"] == 0:
            harm = harm  # keep as-is (N/A if denom=0 already)
            ben  = "N/A"
        if b["correct_before_n"] == 0:
            harm = "N/A"

        print(f"{cond:<22} {b['n']:>4}  "
              f"{pct(b['r1_correct'], b['n']):>6}  "
              f"{pct(b['r2_correct'], b['n']):>6}  "
              f"{delta_acc(b):>6}  "
              f"{pct(b['revised'], b['n']):>5}  "
              f"{harm:>6}  "
              f"{ben:>5}  "
              f"{mean(b['delta_c']):>7}")

    print("-" * len(header))
    t = totals
    print(f"{'ALL':<22} {t['n']:>4}  "
          f"{pct(t['r1_correct'], t['n']):>6}  "
          f"{pct(t['r2_correct'], t['n']):>6}  "
          f"{('+' if t['r2_correct'] >= t['r1_correct'] else '')}"
          f"{(t['r2_correct'] - t['r1_correct']) / t['n'] * 100:.1f}  "
          f"{pct(t['revised'], t['n']):>5}  "
          f"{pct(t['harmful'], t['correct_before_n']):>6}  "
          f"{pct(t['beneficial'], t['wrong_before_n']):>5}  "
          f"{mean(all_dc):>7}")

    if skipped:
        print(f"\n{skipped} row(s) skipped (missing judgment/confidence).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compute_accuracy.py <path_to_results.jsonl>")
        sys.exit(1)
    rows = load_rows(sys.argv[1])
    buckets, skipped = compute(rows)
    print_report(buckets, skipped)

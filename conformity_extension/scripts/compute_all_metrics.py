#!/usr/bin/env python3
"""
compute_all_metrics.py

Walks the results/ directory tree, finds every JSONL results file,
computes accuracy and revision metrics for each model × seed combination,
saves a CSV next to the source JSONL, and prints a consolidated summary.

Usage:
    python compute_all_metrics.py                        # auto-detects ./results/
    python compute_all_metrics.py --results_dir /path/to/results

Output:
    results/rq1/<model>/seed_<N>/<filename>.metrics.csv
    results/rq1/<model>/<model>_all_seeds.csv   (averaged across seeds)
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path


# ── metric computation (same logic as compute_accuracy.py) ──────────────────

def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute(rows):
    buckets = defaultdict(lambda: {
        "n": 0, "r1_correct": 0, "r2_correct": 0,
        "revised": 0, "correct_before_n": 0, "harmful": 0,
        "wrong_before_n": 0, "beneficial": 0, "delta_c": [],
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
        r1_ok = (a0 == gold)
        r2_ok = (af == gold)
        if r1_ok: b["r1_correct"] += 1
        if r2_ok: b["r2_correct"] += 1
        if af != a0: b["revised"] += 1
        if r1_ok:
            b["correct_before_n"] += 1
            if not r2_ok: b["harmful"] += 1
        else:
            b["wrong_before_n"] += 1
            if r2_ok: b["beneficial"] += 1
        b["delta_c"].append(cf - c0)
    return buckets, skipped


def pct_val(num, denom):
    return round(100 * num / denom, 2) if denom > 0 else None


def mean_val(vals):
    return round(sum(vals) / len(vals), 4) if vals else None


def buckets_to_rows(buckets, skipped, model, seed, source_file):
    """Convert computed buckets to a list of dicts (one per condition + ALL)."""
    rows = []
    totals = defaultdict(int)
    all_dc = []

    for cond in sorted(buckets):
        b = buckets[cond]
        all_dc.extend(b["delta_c"])
        for k in ["n", "r1_correct", "r2_correct", "revised",
                  "correct_before_n", "harmful", "wrong_before_n", "beneficial"]:
            totals[k] += b[k]

        d_acc = round((b["r2_correct"] - b["r1_correct"]) / b["n"] * 100, 2) if b["n"] else None
        rows.append({
            "model": model,
            "seed": seed,
            "source_file": source_file,
            "condition": cond,
            "n": b["n"],
            "acc_r1": pct_val(b["r1_correct"], b["n"]),
            "acc_r2": pct_val(b["r2_correct"], b["n"]),
            "delta_acc": d_acc,
            "rev_pct": pct_val(b["revised"], b["n"]),
            "harm_pct": pct_val(b["harmful"], b["correct_before_n"]),
            "ben_pct": pct_val(b["beneficial"], b["wrong_before_n"]),
            "mean_dC": mean_val(b["delta_c"]),
            "skipped": "",
        })

    t = totals
    d_acc_all = round((t["r2_correct"] - t["r1_correct"]) / t["n"] * 100, 2) if t["n"] else None
    rows.append({
        "model": model,
        "seed": seed,
        "source_file": source_file,
        "condition": "ALL",
        "n": t["n"],
        "acc_r1": pct_val(t["r1_correct"], t["n"]),
        "acc_r2": pct_val(t["r2_correct"], t["n"]),
        "delta_acc": d_acc_all,
        "rev_pct": pct_val(t["revised"], t["n"]),
        "harm_pct": pct_val(t["harmful"], t["correct_before_n"]),
        "ben_pct": pct_val(t["beneficial"], t["wrong_before_n"]),
        "mean_dC": mean_val(all_dc),
        "skipped": skipped,
    })
    return rows


CSV_FIELDS = [
    "model", "seed", "source_file", "condition",
    "n", "acc_r1", "acc_r2", "delta_acc",
    "rev_pct", "harm_pct", "ben_pct", "mean_dC", "skipped",
]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


# ── directory crawling ───────────────────────────────────────────────────────

def find_result_files(results_dir):
    """
    Yield (jsonl_path, model, seed) for every JSONL that looks like a
    run_studies_base.py output (has judgment_before in first line).
    Skips summary.json files.
    """
    results_dir = Path(results_dir)
    for jsonl_path in sorted(results_dir.rglob("*.jsonl")):
        # derive model and seed from the path structure
        # expected: results/rq1/<model>/seed_<N>/<file>.jsonl
        parts = jsonl_path.parts
        try:
            seed_idx = next(i for i, p in enumerate(parts) if p.startswith("seed_"))
            seed = parts[seed_idx].replace("seed_", "")
            model = parts[seed_idx - 1]
        except StopIteration:
            print(f"[skip] can't parse model/seed from path: {jsonl_path}")
            continue

        # quick check that it's a results file (has judgment_before)
        try:
            with open(jsonl_path, encoding="utf-8") as f:
                first = f.readline()
            if "judgment_before" not in first:
                continue
        except Exception:
            continue

        yield jsonl_path, model, seed


# ── averaging across seeds ───────────────────────────────────────────────────

def average_seeds(all_rows_for_model):
    """Average numeric metrics across seeds for the same condition."""
    by_cond = defaultdict(list)
    for r in all_rows_for_model:
        by_cond[r["condition"]].append(r)

    averaged = []
    for cond in sorted(by_cond, key=lambda c: (c != "ALL", c)):
        group = by_cond[cond]
        numeric_keys = ["n", "acc_r1", "acc_r2", "delta_acc",
                        "rev_pct", "harm_pct", "ben_pct", "mean_dC"]
        avg_row = {
            "model": group[0]["model"],
            "seed": f"avg({','.join(sorted(set(r['seed'] for r in group)))})",
            "source_file": "averaged",
            "condition": cond,
            "skipped": "",
        }
        for k in numeric_keys:
            vals = [r[k] for r in group if r[k] is not None]
            avg_row[k] = round(sum(vals) / len(vals), 2) if vals else None
        averaged.append(avg_row)
    return averaged


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default="./results_todos",
                   help="root results directory (default: ./results_todos)")
    p.add_argument("--filename_filter", default=None,
                   help="only process JSONL files whose name contains this string "
                        "(e.g. 'primevul' or 'ood_benchmark')")
    args = p.parse_args()

    results = list(find_result_files(args.results_dir))
    if args.filename_filter:
        results = [(p, m, s) for p, m, s in results
                   if args.filename_filter in p.name]
        print(f"[info] filename_filter='{args.filename_filter}': {len(results)} file(s) matched")
    if not results:
        print(f"[error] no result JSONL files found under {args.results_dir}")
        return

    print(f"[info] found {len(results)} result file(s)\n")

    # group by model for cross-seed averaging
    by_model = defaultdict(list)

    for jsonl_path, model, seed in results:
        rows_data = load_rows(jsonl_path)
        buckets, skipped = compute(rows_data)
        metric_rows = buckets_to_rows(buckets, skipped, model, seed,
                                       jsonl_path.name)

        # save per-file CSV next to the JSONL
        csv_path = jsonl_path.with_suffix(".metrics.csv")
        write_csv(csv_path, metric_rows)
        print(f"[saved] {csv_path}")

        by_model[model].extend(metric_rows)

    # save per-model averaged CSV one level up (in the model directory)
    print()
    for model, all_rows in sorted(by_model.items()):
        seeds = sorted(set(r["seed"] for r in all_rows if not r["seed"].startswith("avg")))
        if len(seeds) > 1:
            avg_rows = average_seeds(all_rows)
            model_dir = Path(args.results_dir) / "rq1" / model
            avg_path = model_dir / f"{model}_all_seeds.csv"
            write_csv(avg_path, avg_rows)
            print(f"[saved] {avg_path}  (averaged across seeds: {', '.join(seeds)})")
        else:
            print(f"[info]  {model}: only 1 seed found, no averaging CSV written")

    print("\nDone.")


if __name__ == "__main__":
    main()

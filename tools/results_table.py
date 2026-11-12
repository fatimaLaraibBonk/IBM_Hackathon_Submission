"""Print results/tables/summary.csv as a Markdown table and splice it into README.md.

Usage:  python tools/results_table.py [--write]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- RESULTS_TABLE_START -->", "<!-- RESULTS_TABLE_END -->"


def markdown_table() -> str:
    df = pd.read_csv(ROOT / "results" / "tables" / "summary.csv")
    rows = ["| Method | Mean gap | Median gap | Ties / losses vs exact | Valid | P(valid) | P(optimal) | "
            "Connected to base | Evaluations | Time / instance |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]

    def f(v, fmt):
        return "n/a" if pd.isna(v) else format(v, fmt)

    for _, r in df.iterrows():
        rows.append(f"| {r.method} | {r.gap_mean:.2%} | {r.gap_median:.2%} | {int(r.tie_vs_exact)} / "
                    f"{int(r.loss_vs_exact)} | {r.valid_rate:.0%} | {f(r.p_valid_mean, '.3f')} | "
                    f"{f(r.p_optimal_mean, '.4f')} | {r.connected_rate:.0%} | {r.evals_mean:.0f} | "
                    f"{r.runtime_mean_s:.3f} s |")
    return "\n".join(rows)


if __name__ == "__main__":
    table = markdown_table()
    print(table)
    if "--write" in sys.argv:
        readme = ROOT / "README.md"
        text = readme.read_text()
        a, b = text.index(START) + len(START), text.index(END)
        readme.write_text(text[:a] + "\n" + table + "\n" + text[b:])
        print("README.md updated")

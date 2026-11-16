"""Turn the long-format result rows into a CSV, a markdown summary, and plots."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import RESULTS_DIR

# One "headline" metric per task kind, used for the leaderboard.
HEADLINE = {
    "sts": "spearman",
    "retrieval": "ndcg@10",
    "reranking": "map",
    "classification": "f1_macro",
    "clustering": "v_measure",
}


def save_all(rows: list[dict], out_dir: Path | None = None) -> Path:
    out_dir = out_dir or (RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results_long.csv", index=False)

    _write_markdown(df, out_dir)
    _plots(df, out_dir)
    return out_dir


def _write_markdown(df: pd.DataFrame, out_dir: Path) -> None:
    lines = ["# Embedding model evaluation", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M}", ""]

    # --- Leaderboard: best config per model on each task's headline metric ---
    lines += ["## Leaderboard (best config per model, headline metric)", ""]
    scored = df[df["metric"].isin(HEADLINE.values())].copy()
    scored = scored[scored.apply(lambda r: HEADLINE.get(r["kind"]) == r["metric"], axis=1)]
    if not scored.empty:
        best = (scored.sort_values("value", ascending=False)
                      .groupby(["task", "model"], as_index=False).first())
        pivot = best.pivot_table(index="task", columns="model", values="value")
        lines.append(pivot.round(4).to_markdown())
        lines.append("")

    # --- Efficiency ---
    eff = df[df["kind"] == "efficiency"]
    if not eff.empty:
        lines += ["## Efficiency (per model x config)", ""]
        ep = eff.pivot_table(index=["model", "config"], columns="metric", values="value")
        lines.append(ep.round(2).to_markdown())
        lines.append("")

    # --- Effect of settings: avg headline score by normalize / precision ---
    if not scored.empty:
        lines += ["## Effect of settings (mean headline score)", ""]
        for axis in ("normalize", "precision", "batch_size", "effective_dim"):
            g = scored.groupby(["model", axis])["value"].mean().unstack()
            lines += [f"### by {axis}", "", g.round(4).to_markdown(), ""]

    (out_dir / "summary.md").write_text("\n".join(lines))


def _plots(df: pd.DataFrame, out_dir: Path) -> None:
    scored = df[df["metric"].isin(HEADLINE.values())].copy()
    scored = scored[scored.apply(lambda r: HEADLINE.get(r["kind"]) == r["metric"], axis=1)]
    if not scored.empty:
        best = (scored.sort_values("value", ascending=False)
                      .groupby(["task", "model"], as_index=False).first())
        pivot = best.pivot_table(index="task", columns="model", values="value")
        ax = pivot.plot(kind="bar", figsize=(12, 6))
        ax.set_ylabel("headline score (best config)")
        ax.set_title("Model comparison per task")
        plt.tight_layout()
        plt.savefig(out_dir / "comparison.png", dpi=130)
        plt.close()

    eff = df[(df["kind"] == "efficiency") & (df["metric"] == "texts_per_sec")]
    if not eff.empty:
        ax = eff.pivot_table(index="config", columns="model", values="value").plot(
            kind="bar", figsize=(12, 6))
        ax.set_ylabel("texts / sec")
        ax.set_title("Throughput per config")
        plt.tight_layout()
        plt.savefig(out_dir / "throughput.png", dpi=130)
        plt.close()

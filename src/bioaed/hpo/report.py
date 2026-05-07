"""HPO partial report generator.

Scans ``outputs/hpo/`` for Optuna SQLite studies, prints a summary table,
saves a Markdown report, and generates optimization-history plots.

Usage::

    python -m bioaed.hpo.report
    python -m bioaed.hpo.report --hpo-dir outputs/hpo --out reports/hpo_report.md
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

HPO_ROOT = Path("outputs/hpo")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

# Human-readable display names for directory-level keys
_DISPLAY_NAMES: dict[str, str] = {
    "inceptiontime": "InceptionTime",
    "audiospectrogramtransformer": "AST",
    "audiomamba": "AudioMamba (scratch)",
    "audiomamba_pretrained": "AudioMamba (pretrained)",
}


@dataclass
class StudySummary:
    model_key: str
    dataset: str
    completed: int
    total: int
    best_val_loss: float | None
    best_trial_id: int | None
    best_params: dict = field(default_factory=dict)

    @property
    def display_model(self) -> str:
        return _DISPLAY_NAMES.get(self.model_key, self.model_key)

    @property
    def progress_pct(self) -> float:
        return 100.0 * self.completed / self.total if self.total else 0.0


def _load_study(db_path: Path) -> StudySummary:
    """Read key stats directly from an Optuna SQLite database."""
    model_key = db_path.parts[-3]
    dataset = db_path.parts[-2]

    with sqlite3.connect(db_path, timeout=10) as conn:
        completed = conn.execute("SELECT COUNT(*) FROM trials WHERE state = 'COMPLETE'").fetchone()[
            0
        ]
        total = conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
        best_row = conn.execute(
            """
            SELECT t.trial_id, tv.value
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.state = 'COMPLETE'
            ORDER BY tv.value ASC
            LIMIT 1
            """
        ).fetchone()

    best_val = best_row[1] if best_row else None
    best_id = best_row[0] if best_row else None

    # Load best_params from JSON if available (written after every completed trial)
    best_params: dict = {}
    params_file = db_path.with_name("best_params.json")
    if params_file.exists():
        best_params = json.loads(params_file.read_text())

    return StudySummary(
        model_key=model_key,
        dataset=dataset,
        completed=completed,
        total=total,
        best_val_loss=best_val,
        best_trial_id=best_id,
        best_params=best_params,
    )


def _load_history(db_path: Path) -> pd.DataFrame:
    """Return a DataFrame with trial_id and value for all completed trials."""
    with sqlite3.connect(db_path, timeout=10) as conn:
        df = pd.read_sql_query(
            """
            SELECT t.trial_id, tv.value
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.state = 'COMPLETE'
            ORDER BY t.trial_id ASC
            """,
            conn,
        )
    if not df.empty:
        df["best_so_far"] = df["value"].cummin()
    return df


def plot_optimization_histories(
    summaries: list[StudySummary],
    hpo_root: Path = HPO_ROOT,
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot optimization history (val_loss per trial + running best) for all studies."""
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "hpo_optimization_history.pdf"

    datasets = sorted({s.dataset for s in summaries})
    model_keys = sorted({s.model_key for s in summaries})

    fig, axes = plt.subplots(
        len(datasets),
        len(model_keys),
        figsize=(5 * len(model_keys), 4 * len(datasets)),
        squeeze=False,
    )

    for row_i, dataset in enumerate(datasets):
        for col_i, model_key in enumerate(model_keys):
            ax = axes[row_i][col_i]
            db_path = hpo_root / model_key / dataset / "study.db"
            if not db_path.exists():
                ax.set_visible(False)
                continue

            hist = _load_history(db_path)
            display = _DISPLAY_NAMES.get(model_key, model_key)

            if hist.empty:
                ax.text(0.5, 0.5, "No trials yet", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.scatter(hist["trial_id"], hist["value"], alpha=0.4, s=18, label="trial")
                ax.plot(
                    hist["trial_id"], hist["best_so_far"], color="red", lw=1.5, label="best so far"
                )
                ax.legend(fontsize=7)

            ax.set_title(f"{display}\n{dataset}", fontsize=9)
            ax.set_xlabel("Trial", fontsize=8)
            ax.set_ylabel("Val loss", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle("HPO Optimization History (partial)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Optimization history plot saved to {output_path}")
    return output_path


def _build_markdown(summaries: list[StudySummary]) -> str:
    """Build a Markdown report string."""
    lines: list[str] = [
        "# HPO Partial Report",
        "",
        "## Progress Summary",
        "",
        "| Model | Dataset | Completed | Best Val Loss |",
        "|-------|---------|-----------|---------------|",
    ]

    for s in sorted(summaries, key=lambda x: (x.model_key, x.dataset)):
        best = f"{s.best_val_loss:.6f}" if s.best_val_loss is not None else "—"
        progress = f"{s.completed}/{s.total} ({s.progress_pct:.0f}%)"
        lines.append(f"| {s.display_model} | {s.dataset} | {progress} | {best} |")

    lines += ["", "## Best Hyperparameters per Study", ""]

    for s in sorted(summaries, key=lambda x: (x.model_key, x.dataset)):
        lines.append(f"### {s.display_model} × {s.dataset}")
        if not s.best_params:
            lines.append("_No best_params.json yet._")
        else:
            lines.append("")
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            for k, v in s.best_params.items():
                lines.append(f"| `{k}` | {v} |")
        lines.append("")

    return "\n".join(lines)


def generate_hpo_report(
    hpo_root: Path = HPO_ROOT,
    output_path: Path | None = None,
    plot: bool = True,
) -> list[StudySummary]:
    """Main entry point: load all studies, print summary, save report + plot.

    Args:
        hpo_root: Root directory containing per-model/dataset study.db files.
        output_path: Where to write the Markdown report.
        plot: Whether to generate the optimization-history figure.

    Returns:
        List of :class:`StudySummary` for all discovered studies.
    """
    if output_path is None:
        output_path = REPORTS_DIR / "hpo_report.md"

    db_paths = sorted(hpo_root.rglob("study.db"))
    if not db_paths:
        logger.warning(f"No study.db files found under {hpo_root}")
        return []

    summaries = [_load_study(p) for p in db_paths]

    # --- Console summary ---
    header = f"{'Model':<30} {'Dataset':<12} {'Progress':>12}  {'Best val_loss':>14}"
    print("\n" + "=" * len(header))
    print("HPO Progress Report")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for s in sorted(summaries, key=lambda x: (x.model_key, x.dataset)):
        best = f"{s.best_val_loss:.6f}" if s.best_val_loss is not None else "      —"
        progress = f"{s.completed}/{s.total} ({s.progress_pct:.0f}%)"
        print(f"{s.display_model:<30} {s.dataset:<12} {progress:>12}  {best:>14}")
    print("=" * len(header) + "\n")

    # --- Markdown report ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_build_markdown(summaries))
    logger.info(f"Markdown report saved to {output_path}")

    # --- Plot ---
    if plot:
        plot_optimization_histories(summaries, hpo_root=hpo_root)

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a partial HPO report.")
    parser.add_argument(
        "--hpo-dir",
        default=str(HPO_ROOT),
        help=f"Root HPO output directory (default: {HPO_ROOT})",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output Markdown file path (default: reports/hpo_report.md)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the optimization history figure",
    )
    args = parser.parse_args()

    generate_hpo_report(
        hpo_root=Path(args.hpo_dir),
        output_path=Path(args.out) if args.out else None,
        plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()

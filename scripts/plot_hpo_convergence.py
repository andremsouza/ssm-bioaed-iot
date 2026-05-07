"""Generate HPO convergence plot (E3).

For each study, compute the rolling best (best-so-far) validation mAP
across trials and plot convergence curves per model × dataset.
Saves to reports/figures/hpo_convergence.pdf.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STUDIES = [
    # (label, model_dir, dataset, linestyle)
    ("AST", "audiospectrogramtransformer", "anuraset", "-", "tab:blue"),
    ("AST", "audiospectrogramtransformer", "aswine", "-", "tab:blue"),
    ("SSAMBA", "audiomamba_pretrained", "anuraset", "--", "tab:orange"),
    ("SSAMBA", "audiomamba_pretrained", "aswine", "--", "tab:orange"),
    ("Mamba (scratch)", "audiomamba", "anuraset", "-.", "tab:green"),
    ("Mamba (scratch)", "audiomamba", "aswine", "-.", "tab:green"),
]


def best_so_far(values):
    """Running minimum — HPO objective is val_loss (minimize)."""
    best, out = np.inf, []
    for v in values:
        if v < best:
            best = v
        out.append(best)
    return np.array(out)


def load_map_sequence(model_dir, dataset):
    path = os.path.join(BASE, "outputs", "hpo", model_dir, dataset, "study.db")
    if not os.path.exists(path):
        return None
    storage = f"sqlite:///{path}"
    names = optuna.get_all_study_names(storage=storage)
    study = optuna.load_study(study_name=names[0], storage=storage)
    completed = sorted(
        [t for t in study.trials if t.state.name == "COMPLETE"],
        key=lambda t: t.number,
    )
    # objective is val_loss (minimise)
    values = [t.value for t in completed]
    return np.array(values)


fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=False)
DATASETS = ["anuraset", "aswine"]
DS_LABELS = {"anuraset": "AnuraSet", "aswine": "aSwine"}

plotted_labels = set()

for ax, ds in zip(axes, DATASETS):
    for label, model_dir, dataset, ls, color in STUDIES:
        if dataset != ds:
            continue
        vals = load_map_sequence(model_dir, dataset)
        if vals is None:
            continue
        bsf = best_so_far(vals)
        trial_nums = np.arange(1, len(bsf) + 1)
        lbl = label if label not in plotted_labels else None
        ax.plot(trial_nums, bsf, linestyle=ls, color=color, linewidth=1.8, label=lbl)
        # also show individual trial values as faint scatter
        ax.scatter(trial_nums, vals, color=color, s=8, alpha=0.35, zorder=2)
        if lbl:
            plotted_labels.add(label)

    ax.set_title(DS_LABELS[ds], fontsize=11)
    ax.set_xlabel("Trial number", fontsize=10)
    ax.set_ylabel(r"Best-so-far val.\ loss $\downarrow$", fontsize=10)
    ax.set_xlim(0.5, 30.5)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax.grid(True, linestyle=":", alpha=0.5)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.01),
    fontsize=9,
    frameon=False,
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = os.path.join(BASE, "reports", "figures", "hpo_convergence.pdf")
plt.savefig(out_path, bbox_inches="tight")
print(f"Saved → {out_path}")

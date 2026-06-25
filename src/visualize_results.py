import sys
import os
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
sys.path.insert(0, os.path.dirname(__file__))
from config import OUTPUT_DIR

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    7.5,
    "legend.title_fontsize": 8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.7,
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.4,
    "grid.alpha":         0.45,
    "lines.linewidth":    1.4,
    "lines.markersize":   5,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.04,
})

SHOTS         = [1, 2, 4, 8, 16]
CTX_LENGTHS   = [4, 8, 16]
DATASET_ORDER = ["Oxford Pets", "DTD", "EuroSAT", "Oxford Flowers",
                 "Caltech-101", "Food-101", "UCF-101", "FGVC-Aircraft"]
GRID_ROWS, GRID_COLS = 2, 4

COOP_METHODS = [
    "CoOp (best seed)",
    "Snapshot Ensemble",
    "Multi-Seed Ensemble",
    "Full Ensemble (seeds × snapshots)",
]
METHOD_LABELS = {
    "CoOp (best seed)":                   "CoOp (single)",
    "Snapshot Ensemble":                  "Snapshot Ens.",
    "Multi-Seed Ensemble":                "Multi-Seed Ens.",
    "Full Ensemble (seeds × snapshots)":  "Full Ensemble",
}
METHOD_STYLE = {
    "CoOp (best seed)":                   {"color": "#2166ac", "marker": "o", "linestyle": "-"},
    "Snapshot Ensemble":                  {"color": "#4dac26", "marker": "s", "linestyle": "-"},
    "Multi-Seed Ensemble":                {"color": "#d6604d", "marker": "^", "linestyle": "-"},
    "Full Ensemble (seeds × snapshots)":  {"color": "#8e4585", "marker": "D", "linestyle": "-"},
}
ZEROSHOT_STYLE = {"color": "#888888", "linestyle": "--", "linewidth": 1.2}


def load_ensemble_results(csv_path):
    """data[dataset][k_shot][ctx][method] = accuracy"""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            ds     = row["dataset"]
            k_shot = int(row["k_shot"])
            ctx    = int(row["ctx_length"])
            method = row["method"]
            acc    = float(row["accuracy"])
            data[ds][k_shot][ctx][method] = acc
    return data


def load_zeroshot_summary(csv_path):
    zs = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            zs[row["dataset"]] = float(row["mean_accuracy"])
    return zs


# ── Single-dataset line plot ─────────────────────────────────────────

def plot_accuracy_vs_shots(dataset, ens_data, zs_acc, ctx, out_dir):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.axhline(zs_acc, label="Zero-Shot CLIP", **ZEROSHOT_STYLE)

    for method in COOP_METHODS:
        ys = [ens_data[dataset][k][ctx][method] for k in SHOTS]
        st = METHOD_STYLE[method]
        ax.plot(SHOTS, ys, color=st["color"], marker=st["marker"],
                linestyle=st["linestyle"], label=METHOD_LABELS[method])

    ax.set_xlabel("Training shots per class")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title(f"{dataset}  (M={ctx})")
    ax.set_xticks(SHOTS)
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    all_vals = [zs_acc] + [ens_data[dataset][k][ctx][m]
                            for k in SHOTS for m in COOP_METHODS]
    y_lo = max(0,  5 * ((min(all_vals) - 3) // 5))
    y_hi = min(100, 5 * ((max(all_vals) + 4) // 5))
    ax.set_ylim(y_lo, y_hi)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="0.8")
    fig.tight_layout()

    slug = dataset.lower().replace(" ", "_").replace("-", "")
    save_path = out_dir / f"accuracy_vs_shots_{slug}_M{ctx}.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved → {save_path.name}")


# ── Combined 2x4 grid figure ─────────────────────────────────────────

def plot_accuracy_vs_shots_grid(ens_data, zs_data, ctx, out_dir):
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(14.0, 6.5))
    axes = axes.flatten()

    handles_seen = None
    labels_seen  = None

    for idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[idx]
        zs_acc = zs_data[dataset]
        ax.axhline(zs_acc, label="Zero-Shot CLIP", **ZEROSHOT_STYLE)

        for method in COOP_METHODS:
            ys = [ens_data[dataset][k][ctx][method] for k in SHOTS]
            st = METHOD_STYLE[method]
            ax.plot(SHOTS, ys, color=st["color"], marker=st["marker"],
                    linestyle=st["linestyle"], label=METHOD_LABELS[method])

        ax.set_xticks(SHOTS)
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_title(dataset, fontsize=10, fontweight="bold")

        if idx % GRID_COLS == 0:
            ax.set_ylabel("Top-1 accuracy (%)")
        if idx >= GRID_COLS:
            ax.set_xlabel("Training shots per class")

        all_vals = [zs_acc] + [ens_data[dataset][k][ctx][m]
                                for k in SHOTS for m in COOP_METHODS]
        y_lo = max(0,  5 * ((min(all_vals) - 3) // 5))
        y_hi = min(100, 5 * ((max(all_vals) + 4) // 5))
        ax.set_ylim(y_lo, y_hi)

        if handles_seen is None:
            handles_seen, labels_seen = ax.get_legend_handles_labels()

    for j in range(len(DATASET_ORDER), GRID_ROWS * GRID_COLS):
        axes[j].axis("off")

    fig.suptitle(f"Accuracy vs Training Shots  (M={ctx})",
                 fontsize=11, fontweight="bold", y=1.00)
    fig.legend(handles_seen, labels_seen,
               loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.03), fontsize=8.5)

    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    save_path = out_dir / f"accuracy_vs_shots_grid_M{ctx}.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path.name}")


# ── Grouped bar at 16-shot ───────────────────────────────────────────

def plot_grouped_bar_16shot(ens_data, zs_data, ctx, out_dir):
    all_methods   = ["Zero-Shot CLIP"] + COOP_METHODS
    method_colors = {
        "Zero-Shot CLIP": ZEROSHOT_STYLE["color"],
        **{m: METHOD_STYLE[m]["color"] for m in COOP_METHODS},
    }
    method_labels = {"Zero-Shot CLIP": "Zero-Shot CLIP", **METHOD_LABELS}

    n_datasets = len(DATASET_ORDER)
    n_methods  = len(all_methods)
    group_width = 0.78
    bar_w       = group_width / n_methods
    x_centres   = np.arange(n_datasets, dtype=float)

    fig, ax = plt.subplots(figsize=(11.0, 3.8))
    for m_idx, method in enumerate(all_methods):
        offsets = (m_idx - (n_methods - 1) / 2) * bar_w
        vals = []
        for ds in DATASET_ORDER:
            if method == "Zero-Shot CLIP":
                vals.append(zs_data[ds])
            else:
                vals.append(ens_data[ds][16][ctx][method])
        bars = ax.bar(x_centres + offsets, vals,
                      width=bar_w * 0.92,
                      color=method_colors[method],
                      label=method_labels[method],
                      linewidth=0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{v:.1f}",
                    ha="center", va="bottom",
                    fontsize=5.5, color="0.25")

    ax.set_xticks(x_centres)
    ax.set_xticklabels(DATASET_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title(f"Method comparison at 16-shot  (M={ctx})")

    all_vals = [zs_data[ds] for ds in DATASET_ORDER] + [
        ens_data[ds][16][ctx][m] for ds in DATASET_ORDER for m in COOP_METHODS
    ]
    y_lo = max(0, 5 * ((min(all_vals) - 6) // 5))
    ax.set_ylim(y_lo, None)
    ax.legend(ncol=2, loc="lower right", framealpha=0.9, edgecolor="0.8")

    fig.tight_layout()
    save_path = out_dir / f"grouped_bar_16shot_M{ctx}.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved → {save_path.name}")


# ── Improvement heatmap ──────────────────────────────────────────────

def plot_improvement_heatmap(ens_data, ctx, out_dir):
    matrix = np.zeros((len(DATASET_ORDER), len(SHOTS)))
    for r, ds in enumerate(DATASET_ORDER):
        for c, k in enumerate(SHOTS):
            full   = ens_data[ds][k][ctx]["Full Ensemble (seeds × snapshots)"]
            single = ens_data[ds][k][ctx]["CoOp (best seed)"]
            matrix[r, c] = full - single

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   vmin=-1.0, vmax=matrix.max() + 0.5)
    ax.set_xticks(range(len(SHOTS)))
    ax.set_xticklabels([f"{k}" for k in SHOTS])
    ax.set_yticks(range(len(DATASET_ORDER)))
    ax.set_yticklabels(DATASET_ORDER)
    ax.set_xlabel("Training shots per class")
    ax.set_title(f"Full Ensemble gain over CoOp (single) — pp  (M={ctx})")

    span = matrix.max() + 0.5 - (-1.0)
    for r in range(len(DATASET_ORDER)):
        for c in range(len(SHOTS)):
            val = matrix[r, c]
            norm_val = (val - (-1.0)) / span
            text_col = "white" if norm_val > 0.78 or norm_val < 0.18 else "black"
            ax.text(c, r, f"{val:+.2f}",
                    ha="center", va="center",
                    fontsize=8, color=text_col, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("pp improvement", fontsize=7.5)
    ax.grid(False)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    fig.tight_layout()
    save_path = out_dir / f"improvement_heatmap_M{ctx}.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved → {save_path.name}")


# ── M-ablation plot at 16-shot ───────────────────────────────────────

def plot_M_ablation_16shot(ens_data, out_dir):
    """Bar chart: Full Ensemble accuracy vs M for each dataset at 16-shot."""
    M_colors = {4: "#fdae61", 8: "#abd9e9", 16: "#74add1"}

    n_datasets = len(DATASET_ORDER)
    n_M        = len(CTX_LENGTHS)
    group_width = 0.7
    bar_w       = group_width / n_M
    x_centres   = np.arange(n_datasets, dtype=float)

    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    for m_idx, M in enumerate(CTX_LENGTHS):
        offsets = (m_idx - (n_M - 1) / 2) * bar_w
        vals = [ens_data[ds][16][M]["Full Ensemble (seeds × snapshots)"]
                for ds in DATASET_ORDER]
        bars = ax.bar(x_centres + offsets, vals,
                      width=bar_w * 0.92,
                      color=M_colors[M],
                      label=f"M={M}",
                      linewidth=0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{v:.1f}",
                    ha="center", va="bottom",
                    fontsize=6.5, color="0.25")

    ax.set_xticks(x_centres)
    ax.set_xticklabels(DATASET_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Effect of context length M on Full Ensemble  (16-shot)")

    all_vals = [ens_data[ds][16][M]["Full Ensemble (seeds × snapshots)"]
                 for ds in DATASET_ORDER for M in CTX_LENGTHS]
    y_lo = max(0, 5 * ((min(all_vals) - 6) // 5))
    ax.set_ylim(y_lo, None)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="0.8")

    fig.tight_layout()
    save_path = out_dir / "M_ablation_16shot.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved → {save_path.name}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ens_csv = OUTPUT_DIR / "coop" / "ensemble" / "ensemble_results.csv"
    zs_csv  = OUTPUT_DIR / "summary_results.csv"
    out_dir = OUTPUT_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in (ens_csv, zs_csv):
        if not p.exists():
            raise FileNotFoundError(f"Required CSV not found: {p}")

    ens_data = load_ensemble_results(ens_csv)
    zs_data  = load_zeroshot_summary(zs_csv)
    print("Datasets in ensemble CSV:", sorted(ens_data.keys()))
    print("Datasets in zero-shot CSV:", sorted(zs_data.keys()))

    for ctx in [8, 16]:
        print(f"\n{'='*50}")
        print(f"  Generating plots for M={ctx}")
        print(f"{'='*50}")

        print("\n[1] Per-dataset line plots")
        for ds in DATASET_ORDER:
            plot_accuracy_vs_shots(ds, ens_data, zs_data[ds], ctx, out_dir)

        print("\n[2] Combined 2x4 grid line plot")
        plot_accuracy_vs_shots_grid(ens_data, zs_data, ctx, out_dir)

        print("\n[3] Grouped bar chart at 16-shot")
        plot_grouped_bar_16shot(ens_data, zs_data, ctx, out_dir)

        print("\n[4] Improvement heatmap")
        plot_improvement_heatmap(ens_data, ctx, out_dir)

    print(f"\n{'='*50}")
    print("  M-ablation plot")
    print(f"{'='*50}")
    plot_M_ablation_16shot(ens_data, out_dir)

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()

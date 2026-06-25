import sys
import os
import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
sys.path.insert(0, os.path.dirname(__file__))
from config import CLIP_MODEL, OUTPUT_DIR, BATCH_SIZE
from datasets import SplitDataset, get_class_names
from coop_model import CoOpCLIP
from prompts import PROMPT_TEMPLATES
try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Note: umap-learn not installed — UMAP plots will be skipped.")

DATASET_NAMES = ["oxford_flowers", "fgvc_aircraft"]
MAX_CLASSES      = 10
K_SHOT           = 16
SEED             = 1
N_CTX            = 16
CANDIDATE_EPOCHS = [10, 50, 100, 200]

DISPLAY_NAMES = {
    "oxford_pets":    "Oxford Pets",
    "dtd":            "DTD",
    "eurosat":        "EuroSAT",
    "oxford_flowers": "Oxford Flowers",
    "caltech101":     "Caltech-101",
    "food101":        "Food-101",
    "ucf101":         "UCF-101",
    "fgvc_aircraft":  "FGVC-Aircraft",
}
_PALETTE = plt.get_cmap("tab10").colors


def select_class_subset(dataset_name: str, max_cls: int):
    all_names = get_class_names(dataset_name)
    n = min(len(all_names), max_cls)
    return list(range(n)), all_names[:n]


def extract_image_features(model, dataset_name, selected_ids, device):
    test_ds = SplitDataset(dataset_name, split="test", transform=model.preprocess)
    id_set  = set(selected_ids)
    indices = [i for i, (_, cid, _) in enumerate(test_ds.samples) if cid in id_set]
    subset  = Subset(test_ds, indices)
    loader  = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    id_to_local = {cid: idx for idx, cid in enumerate(selected_ids)}
    all_feats, all_labels = [], []
    with torch.no_grad():
        for images, class_ids, _ in loader:
            feats = model.encode_image(images.to(device))
            all_feats.append(feats.cpu().numpy())
            all_labels.append(
                np.array([id_to_local[cid.item()] for cid in class_ids])
            )
    return np.concatenate(all_feats), np.concatenate(all_labels)


def extract_zeroshot_text_features(clip_model, class_names, dataset_name, device):
    template = PROMPT_TEMPLATES[dataset_name][0]
    prompts  = [template.format(name) for name in class_names]
    tokens   = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        feats = clip_model.encode_text(tokens).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def extract_coop_text_features(model, ckpt_path, selected_ids, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.prompt.ctx.data.copy_(ckpt["ctx"].to(device))
    model.eval()
    with torch.no_grad():
        all_feats = model.prompt.encode_text()
    return all_feats[selected_ids].cpu().numpy()


def find_snapshot_paths(ckpt_dir, dataset_name, k_shot, seed, n_ctx, candidate_epochs):
    found = []
    # Try ctx-aware filename first; fall back to legacy (no ctx in filename) for M=16
    for ep in candidate_epochs:
        p_new    = ckpt_dir / f"{dataset_name}_seed{seed}_shot{k_shot}_ctx{n_ctx}_ep{ep}.pt"
        p_legacy = ckpt_dir / f"{dataset_name}_seed{seed}_shot{k_shot}_ep{ep}.pt"
        if p_new.exists():
            found.append((ep, p_new))
        elif n_ctx == 16 and p_legacy.exists():
            found.append((ep, p_legacy))
    return found


def run_tsne(data: np.ndarray, random_state: int = 42) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(data) - 1),
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    return tsne.fit_transform(data)


def run_umap(data: np.ndarray, random_state: int = 42) -> np.ndarray:
    reducer = umap_lib.UMAP(n_components=2, random_state=random_state)
    return reducer.fit_transform(data)


def scatter_embeddings(ax, img_2d, img_labels, txt_2d, class_names,
                       title, alpha=0.40, img_s=12, txt_s=180):
    for c, name in enumerate(class_names):
        col  = _PALETTE[c % 10]
        mask = img_labels == c
        ax.scatter(img_2d[mask, 0], img_2d[mask, 1],
                   c=[col], s=img_s, alpha=alpha, linewidths=0, rasterized=True)
        ax.scatter(txt_2d[c, 0], txt_2d[c, 1],
                   c=[col], s=txt_s, marker="*",
                   edgecolors="black", linewidths=0.6, zorder=5)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


def _build_legend_handles(class_names):
    handles = []
    for c, name in enumerate(class_names):
        handles.append(
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=_PALETTE[c % 10], markersize=8, label=name)
        )
    handles.append(
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
                   markersize=12, markeredgecolor="black", label="Text prototype (★)")
    )
    return handles


def make_image_embeddings_figure(
    dataset_name, display, img_feats, img_labels, class_names,
    zs_txt, coop_txt, out_dir
):
    n_img = img_feats.shape[0]
    n_cls = len(class_names)
    combined = np.concatenate([img_feats, zs_txt, coop_txt], axis=0)

    print("    t-SNE …", end=" ", flush=True)
    coords_tsne = run_tsne(combined)
    print("done")
    img_tsne  = coords_tsne[:n_img]
    zs_tsne   = coords_tsne[n_img:n_img + n_cls]
    coop_tsne = coords_tsne[n_img + n_cls:]

    if HAS_UMAP:
        print("    UMAP  …", end=" ", flush=True)
        coords_umap = run_umap(combined)
        print("done")
        img_umap  = coords_umap[:n_img]
        zs_umap   = coords_umap[n_img:n_img + n_cls]
        coop_umap = coords_umap[n_img + n_cls:]
        n_data_rows = 2
    else:
        n_data_rows = 1

    n_rows = n_data_rows + 1
    height_ratios = [1] * n_data_rows + [0.18]
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(12, 5 * n_data_rows + 0.9),
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.suptitle(
        f"{display} — Embedding Space: Zero-Shot CLIP vs CoOp ({K_SHOT}-shot)",
        fontsize=12, fontweight="bold",
    )
    scatter_embeddings(axes[0, 0], img_tsne, img_labels, zs_tsne,
                       class_names, "t-SNE  ·  Zero-Shot CLIP")
    scatter_embeddings(axes[0, 1], img_tsne, img_labels, coop_tsne,
                       class_names, f"t-SNE  ·  CoOp  ({K_SHOT}-shot, seed {SEED})")
    if HAS_UMAP:
        scatter_embeddings(axes[1, 0], img_umap, img_labels, zs_umap,
                           class_names, "UMAP  ·  Zero-Shot CLIP")
        scatter_embeddings(axes[1, 1], img_umap, img_labels, coop_umap,
                           class_names, f"UMAP  ·  CoOp  ({K_SHOT}-shot, seed {SEED})")

    for ax in axes[-1]:
        ax.axis("off")
    axes[-1, 0].legend(
        handles=_build_legend_handles(class_names),
        ncol=min(6, n_cls + 1),
        loc="center", frameon=False, fontsize=9,
    )

    plt.tight_layout()
    save_path = out_dir / f"{dataset_name}_image_embeddings.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {save_path.name}")


def make_evolution_figure(
    dataset_name, display, img_feats, img_labels, class_names,
    selected_ids, model, snapshots, zs_txt, out_dir, device
):
    if not snapshots:
        print("    No snapshots available — skipping evolution figure.")
        return

    print("    Extracting snapshot text features …", end=" ", flush=True)
    snap_txts = {}
    for ep, path in snapshots:
        snap_txts[ep] = extract_coop_text_features(model, path, selected_ids, device)
    print("done")
    all_txt_ordered = [zs_txt] + [snap_txts[ep] for ep, _ in snapshots]
    combined = np.concatenate([img_feats] + all_txt_ordered, axis=0)

    print("    t-SNE (joint) …", end=" ", flush=True)
    coords = run_tsne(combined)
    print("done")

    n_img = img_feats.shape[0]
    n_cls = len(class_names)
    img_2d = coords[:n_img]
    proto_coords = {}
    offset = n_img
    proto_coords[None] = coords[offset:offset + n_cls]
    offset += n_cls
    for ep, _ in snapshots:
        proto_coords[ep] = coords[offset:offset + n_cls]
        offset += n_cls
    cols = [None] + [ep for ep, _ in snapshots]
    col_titles = ["Zero-Shot"] + [f"Epoch {ep}" for ep, _ in snapshots]
    n_cols = len(cols)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5.5))
    if n_cols == 1:
        axes = [axes]
    fig.suptitle(
        f"{display} — Prompt Embedding Evolution  "
        f"({K_SHOT}-shot CoOp, seed {SEED})\n"
        "★ = text prototype  ·  image embeddings fixed across all columns",
        fontsize=11, fontweight="bold",
    )
    for col_idx, (ep, title) in enumerate(zip(cols, col_titles)):
        scatter_embeddings(
            axes[col_idx], img_2d, img_labels, proto_coords[ep],
            class_names, title, alpha=0.35, img_s=8, txt_s=140,
        )
    fig.legend(
        handles=_build_legend_handles(class_names),
        loc="lower center",
        ncol=min(6, n_cls + 1),
        fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, -0.07),
    )
    plt.tight_layout()
    save_path = out_dir / f"{dataset_name}_evolution.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {save_path.name}")


def visualise_dataset(dataset_name, device, ckpt_dir, out_dir):
    display = DISPLAY_NAMES[dataset_name]
    print(f"\n{'='*62}")
    print(f"  {display}")
    print(f"{'='*62}")
    selected_ids, selected_names = select_class_subset(dataset_name, MAX_CLASSES)
    n_cls = len(selected_names)
    print(f"  Classes  : {n_cls}  ({', '.join(selected_names)})")

    all_class_names = get_class_names(dataset_name)
    model = CoOpCLIP(all_class_names, n_ctx=N_CTX, device=device)
    model.eval()
    clip_model, _ = clip.load(CLIP_MODEL, device=device)
    clip_model.eval()

    print("  Extracting image features …", end=" ", flush=True)
    img_feats, img_labels = extract_image_features(
        model, dataset_name, selected_ids, device
    )
    print(f"{img_feats.shape[0]} images")

    print("  Building zero-shot text features …", end=" ", flush=True)
    zs_txt = extract_zeroshot_text_features(
        clip_model, selected_names, dataset_name, device
    )
    print("done")

    snapshots = find_snapshot_paths(ckpt_dir, dataset_name, K_SHOT, SEED,
                                    N_CTX, CANDIDATE_EPOCHS)
    if not snapshots:
        print(f"  WARNING: no checkpoints found for {dataset_name} "
              f"shot={K_SHOT} seed={SEED} ctx={N_CTX}.  Skipping.")
        return

    final_ep, final_path = snapshots[-1]
    print(f"  CoOp final checkpoint : {final_path.name}")
    coop_txt = extract_coop_text_features(model, final_path, selected_ids, device)

    print("  Figure 1 — image embeddings comparison")
    make_image_embeddings_figure(
        dataset_name, display, img_feats, img_labels, selected_names,
        zs_txt, coop_txt, out_dir,
    )
    print("  Figure 2 — embedding evolution")
    make_evolution_figure(
        dataset_name, display, img_feats, img_labels, selected_names,
        selected_ids, model, snapshots, zs_txt, out_dir, device,
    )


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device : {device}")
    print(f"Model  : {CLIP_MODEL}")
    print(f"UMAP   : {'available' if HAS_UMAP else 'not installed — skipped'}")

    out_dir  = OUTPUT_DIR / "plots" / "embeddings"
    ckpt_dir = OUTPUT_DIR / "coop" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in DATASET_NAMES:
        try:
            visualise_dataset(ds_name, device, ckpt_dir, out_dir)
        except Exception as e:
            print(f"  ERROR on {ds_name}: {e} — skipping.")

    print(f"\n{'='*62}")
    print(f"  Visualisation complete.")
    print(f"  Plots saved to: {out_dir}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()

import sys, os, csv, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(__file__))
from config import CLIP_MODEL, OUTPUT_DIR, BATCH_SIZE
from datasets import SplitDataset, get_class_names
from coop_model import CoOpCLIP

DATASET_NAMES = [
    "oxford_pets",
    "dtd",
    "eurosat",
    "oxford_flowers",
    "food101",
    "caltech101",
    "ucf101",
    "fgvc_aircraft",
]

DISPLAY_NAMES = {
    "oxford_pets": "Oxford Pets",
    "dtd": "DTD",
    "eurosat": "EuroSAT",
    "oxford_flowers": "Oxford Flowers",
    "food101": "Food-101",
    "caltech101": "Caltech-101",
    "ucf101": "UCF-101",
    "fgvc_aircraft": "FGVC-Aircraft",
}
SHOTS = [1, 2, 4, 8, 16]
SEEDS = [1, 2, 3]
CTX_LENGTHS = [4, 8, 16]

def load_checkpoint(ckpt_path, model, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.prompt.ctx.data.copy_(ckpt["ctx"].to(device))
    model.eval()
    return model

def compute_all_logits(model, loader, device):
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, class_ids, _ in loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(class_ids.numpy())

    return np.concatenate(all_logits), np.concatenate(all_labels)

def metrics_from_logits(logits, labels):
    preds = logits.argmax(axis=1)
    accuracy = (preds == labels).mean() * 100.0
    macro_f1 = f1_score(labels, preds, average="macro") * 100.0
    return accuracy, macro_f1

def evaluate_soup(ctx_tensors, model, loader, device):
    if not ctx_tensors:
        return 0.0, 0.0, None, None
    soup_ctx = torch.stack(ctx_tensors).mean(dim=0).to(device)
    model.prompt.ctx.data.copy_(soup_ctx)
    model.eval()
    logits, labels = compute_all_logits(model, loader, device)
    acc, f1 = metrics_from_logits(logits, labels)
    return acc, f1, logits, labels

def discover_checkpoints(ckpt_dir, ds_name, k_shot, n_ctx):
    # Try new format first
    pattern_new    = f"{ds_name}_seed*_shot{k_shot}_ctx{n_ctx}_ep*.pt"
    paths_new      = sorted(glob.glob(str(ckpt_dir / pattern_new)))
    paths_legacy   = []
    if n_ctx == 16:
        pattern_legacy = f"{ds_name}_seed*_shot{k_shot}_ep*.pt"
        candidates     = sorted(glob.glob(str(ckpt_dir / pattern_legacy)))
        for p in candidates:
            if "_ctx" not in Path(p).stem:
                paths_legacy.append(p)

    paths = paths_new if paths_new else paths_legacy
    seed_groups = defaultdict(list)
    for p in paths:
        name  = Path(p).stem
        parts = name.split("_")
        seed_val  = None
        epoch_val = None
        for part in parts:
            if part.startswith("seed"):
                seed_val  = int(part.replace("seed", ""))
            if part.startswith("ep"):
                epoch_val = int(part.replace("ep", ""))
        if seed_val is not None and epoch_val is not None:
            seed_groups[seed_val].append((epoch_val, p))
    for seed in seed_groups:
        seed_groups[seed].sort(key=lambda x: x[0])

    return dict(seed_groups)

def evaluate_all_modes(ds_name, k_shot, n_ctx, device, ckpt_dir):
    class_names = get_class_names(ds_name)
    model = CoOpCLIP(class_names, n_ctx=n_ctx, device=device)
    model.eval()
    preprocess = model.preprocess
    test_dataset = SplitDataset(ds_name, split="test", transform=preprocess)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    val_dataset  = SplitDataset(ds_name, split="val", transform=preprocess)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    seed_groups = discover_checkpoints(ckpt_dir, ds_name, k_shot, n_ctx)

    if not seed_groups:
        print(f"No checkpoints found for {ds_name} shot={k_shot} ctx={n_ctx}")
        return {}
    results = {}
    single_accs = []
    single_f1s  = []
    final_logits_per_seed = {}
    final_ctx_per_seed    = {}
    for seed in sorted(seed_groups.keys()):
        snapshots = seed_groups[seed]
        final_epoch, final_path = snapshots[-1] 
        ckpt = torch.load(final_path, map_location="cpu")
        final_ctx_per_seed[seed] = ckpt["ctx"].clone()
        load_checkpoint(final_path, model, device)
        logits, labels = compute_all_logits(model, test_loader, device)
        acc, f1 = metrics_from_logits(logits, labels)
        single_accs.append(acc)
        single_f1s.append(f1)
        final_logits_per_seed[seed] = logits
    best_idx = int(np.argmax(single_accs))
    results["single_best"] = (single_accs[best_idx], single_f1s[best_idx])
    results["single_mean"] = (np.mean(single_accs), np.mean(single_f1s))
    snap_accs = []
    snap_f1s  = []
    all_snapshot_logits = []   
    all_snapshot_ctxs   = []
    for seed in sorted(seed_groups.keys()):
        snapshots = seed_groups[seed]
        seed_logits_list = []

        for epoch, path in snapshots:
            ckpt = torch.load(path, map_location="cpu")
            all_snapshot_ctxs.append(ckpt["ctx"].clone())

            load_checkpoint(path, model, device)
            logits, labels = compute_all_logits(model, test_loader, device)
            seed_logits_list.append(logits)
            all_snapshot_logits.append(logits)
        avg_logits = np.mean(seed_logits_list, axis=0)
        acc, f1 = metrics_from_logits(avg_logits, labels)
        snap_accs.append(acc)
        snap_f1s.append(f1)
    results["snapshot_ens"] = (np.mean(snap_accs), np.mean(snap_f1s))
    multiseed_logits = list(final_logits_per_seed.values())
    avg_logits = np.mean(multiseed_logits, axis=0)
    acc, f1 = metrics_from_logits(avg_logits, labels)
    results["multiseed_ens"] = (acc, f1)
    avg_logits = np.mean(all_snapshot_logits, axis=0)
    acc, f1 = metrics_from_logits(avg_logits, labels)
    results["full_ens"] = (acc, f1)
    uniform_ctxs = list(final_ctx_per_seed.values())
    acc, f1, _, _ = evaluate_soup(uniform_ctxs, model, test_loader, device)
    results["uniform_soup"] = (acc, f1)
    candidate_list = [] 
    for seed in sorted(seed_groups.keys()):
        ctx_t = final_ctx_per_seed[seed]
        soup_ctx = ctx_t.to(device)
        model.prompt.ctx.data.copy_(soup_ctx)
        model.eval()
        val_logits, val_labels = compute_all_logits(model, val_loader, device)
        v_acc, _ = metrics_from_logits(val_logits, val_labels)
        candidate_list.append((v_acc, ctx_t, seed))

    candidate_list.sort(key=lambda x: x[0], reverse=True)
    ingredients = []
    best_val_acc = -1.0

    for v_acc_cand, ctx_t, seed in candidate_list:
        trial = ingredients + [ctx_t]
        t_acc, _, _, _ = evaluate_soup(trial, model, val_loader, device)
        if t_acc >= best_val_acc:
            best_val_acc = t_acc
            ingredients  = trial
    if ingredients:
        acc, f1, _, _ = evaluate_soup(ingredients, model, test_loader, device)
    else:
        acc, f1 = 0.0, 0.0
    results["greedy_soup"] = (acc, f1)
    acc, f1, _, _ = evaluate_soup(all_snapshot_ctxs, model, test_loader, device)
    results["full_uniform_soup"] = (acc, f1)

    return results

def discover_ctx_lengths(ckpt_dir, ds_name, k_shot):
    found = set()
    for p in glob.glob(str(ckpt_dir / f"{ds_name}_seed*_shot{k_shot}_ctx*_ep*.pt")):
        stem  = Path(p).stem
        for part in stem.split("_"):
            if part.startswith("ctx"):
                try:
                    found.add(int(part.replace("ctx", "")))
                except ValueError:
                    pass
    for p in glob.glob(str(ckpt_dir / f"{ds_name}_seed*_shot{k_shot}_ep*.pt")):
        if "_ctx" not in Path(p).stem:
            found.add(16)

    return sorted(found)

MODE_LABELS = {
    "single_best":       "CoOp (best seed)",
    "single_mean":       "CoOp (mean ± std)",
    "snapshot_ens":      "Snapshot Ensemble",
    "multiseed_ens":     "Multi-Seed Ensemble",
    "full_ens":          "Full Ensemble (seeds × snapshots)",
    "uniform_soup":      "Uniform Soup (final seeds)",
    "greedy_soup":       "Greedy Soup (final seeds)",
    "full_uniform_soup": "Full Uniform Soup (seeds × snapshots)",
}

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device : {device}")
    print(f"Model  : {CLIP_MODEL}")
    coop_root = OUTPUT_DIR / "coop"
    ckpt_dir  = coop_root / "checkpoints"
    ens_dir   = coop_root / "ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for ds_name in DATASET_NAMES:
        display = DISPLAY_NAMES[ds_name]
        print(f"\n{'='*60}")
        print(f"  {display}")
        print(f"{'='*60}")
        for k_shot in SHOTS:
            print(f"\n  --- {k_shot}-shot ---")
            ctx_lengths_found = discover_ctx_lengths(ckpt_dir, ds_name, k_shot)
            if not ctx_lengths_found:
                print(f"  No checkpoints found for {ds_name} shot={k_shot}")
                continue
            for n_ctx in ctx_lengths_found:
                print(f"\n  ctx={n_ctx}")
                results = evaluate_all_modes(
                    ds_name, k_shot, n_ctx, device, ckpt_dir)
                for mode, (acc, f1) in results.items():
                    label = MODE_LABELS.get(mode, mode)
                    print(f"    {label:45s}  acc={acc:.2f}%  f1={f1:.2f}%")
                    all_rows.append({
                        "dataset":    display,
                        "k_shot":     k_shot,
                        "ctx_length": n_ctx,
                        "method":     label,
                        "accuracy":   f"{acc:.2f}",
                        "macro_f1":   f"{f1:.2f}",
                    })
    csv_path = ens_dir / "ensemble_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "k_shot", "ctx_length",
                     "method", "accuracy", "macro_f1"])
        for r in all_rows:
            w.writerow([r["dataset"], r["k_shot"], r["ctx_length"],
                         r["method"], r["accuracy"], r["macro_f1"]])
    print(f"\n{'='*60}")
    print(f"  Ensemble + soup evaluation complete.")
    print(f"  Results saved to: {csv_path}")
    print(f"{'='*60}")
if __name__ == "__main__":
    main()

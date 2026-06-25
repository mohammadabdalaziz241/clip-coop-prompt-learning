import sys, os, csv, copy, math, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, os.path.dirname(__file__))
from config import CLIP_MODEL, SEED, OUTPUT_DIR, BATCH_SIZE as EVAL_BATCH_SIZE
from datasets import SplitDataset, get_class_names
from coop_model import CoOpCLIP

DATASET_NAMES = ["oxford_pets", "dtd", "eurosat", "oxford_flowers",
                 "food101", "caltech101", "ucf101","fgvc_aircraft"]
CTX_LENGTHS = [4, 8, 16]
TRAIN_BATCH_SIZE = 32
SHOTS = [1, 2, 4, 8, 16]
EPOCH_MAP = {1: 50, 2: 100, 4: 100, 8: 200, 16: 200}
SEEDS = [1, 2, 3]
LR = 0.002
WARMUP_LR = 1e-5
WARMUP_EPOCHS = 1
BASE_SNAPSHOT_EPOCHS = [10, 20, 30, 50]

def sample_few_shot(dataset, k_shot, rng):
    class_to_indices = {}
    for idx, (_, class_id, _) in enumerate(dataset.samples):
        class_to_indices.setdefault(class_id, []).append(idx)

    selected = []
    for class_id in sorted(class_to_indices.keys()):
        indices = class_to_indices[class_id]
        if len(indices) < k_shot:
            selected.extend(indices)
        else:
            chosen = rng.choice(indices, size=k_shot, replace=False)
            selected.extend(chosen.tolist())
    return Subset(dataset, selected)

def get_snapshot_epochs(max_epoch):
    snapshots = set()
    for ep in BASE_SNAPSHOT_EPOCHS:
        if ep <= max_epoch:
            snapshots.add(ep)
    ep = 100
    while ep <= max_epoch:
        snapshots.add(ep)
        ep += 50
    snapshots.add(max_epoch)

    return sorted(snapshots)

def adjust_lr(optimizer, epoch, max_epoch):
    if epoch < WARMUP_EPOCHS:
        lr = WARMUP_LR
    else:
        progress = (epoch - WARMUP_EPOCHS) / max(1, max_epoch - WARMUP_EPOCHS)
        lr = 0.5 * LR * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer["param_groups"]:
        param_group["lr"] = lr

    return lr

def train_one_config(ds_name, seed, k_shot, n_ctx, device, out_root):
    tag = f"{ds_name}_seed{seed}_shot{k_shot}_ctx{n_ctx}"
    max_epoch = EPOCH_MAP[k_shot]
    snapshot_epochs = get_snapshot_epochs(max_epoch)
    print(f"\n{'─'*50}")
    print(f"  Training: {tag}  |  epochs={max_epoch}  |  snapshots={snapshot_epochs}")
    print(f"{'─'*50}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    class_names = get_class_names(ds_name)
    model = CoOpCLIP(class_names, n_ctx=n_ctx, device=device)
    preprocess = model.preprocess
    model.train()
    full_train = SplitDataset(ds_name, split="train", transform=preprocess)
    train_subset = sample_few_shot(full_train, k_shot, rng)
    train_loader = DataLoader(train_subset, batch_size=TRAIN_BATCH_SIZE,
                              shuffle=True, num_workers=0, drop_last=False)
    print(f"  Train samples: {len(train_subset)}  "
          f"({k_shot} shots × {len(class_names)} classes)  "
          f"ctx={n_ctx}")
    optimizer = torch.optim.SGD(
        [model.prompt.ctx],
        lr=LR,
        momentum=0.9,
        weight_decay=0.0,
    )
    criterion = nn.CrossEntropyLoss()
    ckpt_dir = out_root / "checkpoints"
    log_dir = out_root / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_rows = []
    saved_checkpoints = []

    for epoch in range(max_epoch):
        if epoch < WARMUP_EPOCHS:
            lr = WARMUP_LR
        else:
            progress = (epoch - WARMUP_EPOCHS) / max(1, max_epoch - WARMUP_EPOCHS)
            lr = 0.5 * LR * (1.0 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        epoch_loss = 0.0
        n_batches = 0

        for images, class_ids, _ in train_loader:
            images = images.to(device)
            labels = class_ids.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        log_rows.append((epoch + 1, f"{avg_loss:.6f}", f"{lr:.8f}"))
        if (epoch + 1) % 10 == 0 or (epoch + 1) == max_epoch:
            print(f"  Epoch {epoch+1:>3d}/{max_epoch}  "
                  f"loss={avg_loss:.4f}  lr={lr:.6f}")
        if (epoch + 1) in snapshot_epochs:
            ckpt_path = ckpt_dir / f"{tag}_ep{epoch+1}.pt"
            torch.save({
                "ctx": model.prompt.ctx.detach().cpu(),
                "epoch": epoch + 1,
                "seed": seed,
                "k_shot": k_shot,
                "dataset": ds_name,
                "n_ctx": n_ctx,
                "loss": avg_loss,
            }, ckpt_path)
            saved_checkpoints.append(str(ckpt_path))
            print(f"  → Saved snapshot: {ckpt_path.name}")

    log_path = log_dir / f"{tag}.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "learning_rate"])
        w.writerows(log_rows)

    model.eval()
    test_dataset = SplitDataset(ds_name, split="test", transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE,
                             shuffle=False, num_workers=0)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, class_ids, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(class_ids.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    from sklearn.metrics import f1_score
    accuracy = (all_preds == all_labels).mean() * 100.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro") * 100.0
    print(f"\n  Test accuracy: {accuracy:.2f}%  |  Macro-F1: {macro_f1:.2f}%")
    final_dir = out_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    result_path = final_dir / f"{tag}_results.csv"
    with open(result_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "seed", "k_shot", "n_ctx", "max_epoch",
                     "accuracy", "macro_f1"])
        w.writerow([ds_name, seed, k_shot, n_ctx, max_epoch,
                     f"{accuracy:.2f}", f"{macro_f1:.2f}"])
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "saved_checkpoints": saved_checkpoints,
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description="CoOp training — few-shot prompt tuning for CLIP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=DATASET_NAMES,
        help="Dataset to train on. Omit to run all datasets.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed. Omit to run all seeds.",
    )
    parser.add_argument(
        "--shot", type=int, default=None,
        choices=SHOTS,
        help="Number of training shots per class. Omit to run all shot counts.",
    )
    parser.add_argument(
        "--ctx", type=int, default=None,
        choices=CTX_LENGTHS,
        help="Number of learnable context tokens (M). Omit to run all ctx lengths.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    datasets = [args.dataset] if args.dataset else DATASET_NAMES
    seeds    = [args.seed]    if args.seed    else SEEDS
    shots    = [args.shot]    if args.shot    else SHOTS
    ctxs     = [args.ctx]     if args.ctx     else CTX_LENGTHS

    print(f"Device    : {device}")
    print(f"Model     : {CLIP_MODEL}")
    print(f"Datasets  : {datasets}")
    print(f"Seeds     : {seeds}")
    print(f"Shots     : {shots}")
    print(f"Ctx lengths: {ctxs}")
    out_root = OUTPUT_DIR / "coop"
    out_root.mkdir(parents=True, exist_ok=True)
    all_results = []

    for ds_name in datasets:
        for seed in seeds:
            for k_shot in shots:
                for n_ctx in ctxs:
                    result = train_one_config(
                        ds_name, seed, k_shot, n_ctx, device, out_root)
                    all_results.append({
                        "dataset": ds_name,
                        "seed": seed,
                        "k_shot": k_shot,
                        "n_ctx": n_ctx,
                        "accuracy": result["accuracy"],
                        "macro_f1": result["macro_f1"],
                    })
    summary_path = out_root / "coop_all_results.csv"
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["dataset", "seed", "k_shot", "n_ctx",
                         "accuracy", "macro_f1"])
        for r in all_results:
            w.writerow([r["dataset"], r["seed"], r["k_shot"], r["n_ctx"],
                         f"{r['accuracy']:.2f}", f"{r['macro_f1']:.2f}"])
    print(f"\n{'='*60}")
    print(f"  All CoOp training complete.")
    print(f"  Master summary: {summary_path}")
    print(f"{'='*60}")
if __name__ == "__main__":
    main()

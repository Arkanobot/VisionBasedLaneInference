"""Training entry point for the IDD-Lite semantic segmentation model."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from data.idd import IDDLite, class_frequencies
from eval.metrics import ConfusionMatrix, format_table
from models.segnet import SegLoss, build_model, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    """Linear warm-up then cosine decay to 1% of the base learning rate."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, loader, device, lane: bool = False) -> dict:
    model.eval()
    cm = ConfusionMatrix()
    lane_inter = lane_union = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        out = model(x)
        logits, lane_logits = out if isinstance(out, tuple) else (out, None)
        cm.update(logits.argmax(1).cpu().numpy(), y.numpy())
        if lane and lane_logits is not None and len(batch) > 2:
            pred = (torch.sigmoid(lane_logits)[:, 0] > 0.5).cpu().numpy()
            tgt = batch[2].numpy() > 0.5
            lane_inter += int((pred & tgt).sum())
            lane_union += int((pred | tgt).sum())
    summary = cm.summary()
    summary["lane_iou"] = (lane_inter / lane_union) if lane_union else float("nan")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Train IDD-Lite segmentation model")
    p.add_argument("--epochs", type=int, default=cfg.DEFAULT.train.epochs)
    p.add_argument("--batch-size", type=int, default=cfg.DEFAULT.train.batch_size)
    p.add_argument("--lr", type=float, default=cfg.DEFAULT.train.lr)
    p.add_argument("--weight-decay", type=float, default=cfg.DEFAULT.train.weight_decay)
    p.add_argument("--arch", default=cfg.DEFAULT.train.arch)
    p.add_argument("--seed", type=int, default=cfg.DEFAULT.train.seed)
    p.add_argument("--num-workers", type=int, default=cfg.DEFAULT.train.num_workers)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--run-name", default="seg")
    p.add_argument("--lane-head", action="store_true",
                   help="train the distilled lane-line head (needs laneFine/)")
    p.add_argument("--lane-weight", type=float, default=0.5)
    p.add_argument("--init-from", default=None,
                   help="warm-start the weights from another checkpoint, with a "
                        "fresh schedule. Used to fine-tune an existing model on "
                        "new augmentations without retraining from scratch.")
    p.add_argument("--no-dashcam-aug", action="store_true",
                   help="disable the wide-angle / low-light augmentations")
    p.add_argument("--resume", action="store_true",
                   help="continue from last.pt in the run directory if present")
    p.add_argument("--checkpoint-dir", default=None,
                   help="where to write checkpoints. Point this at Drive on "
                        "Colab: a session drop destroys local disk, and a "
                        "checkpoint that dies with the session cannot be resumed "
                        "from.")
    p.add_argument("--input-size", default=None,
                   help="HxW, e.g. 360x640; defaults to the config value")
    p.add_argument("--data-root", default=str(cfg.DATA_ROOT))
    args = p.parse_args()

    set_seed(args.seed)
    device = cfg.resolve_device()
    out_dir = (Path(args.checkpoint_dir) / args.run_name if args.checkpoint_dir
               else cfg.CHECKPOINT_ROOT / args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_size:
        H, W = (int(v) for v in args.input_size.lower().split("x"))
    else:
        H, W = cfg.DEFAULT.train.input_size
    train_ds = IDDLite(args.data_root, "train", (H, W), augment=True,
                       seed=args.seed, load_lane=args.lane_head,
                       dashcam_aug=not args.no_dashcam_aug)
    val_ds = IDDLite(args.data_root, "val", (H, W), augment=False,
                     load_lane=args.lane_head)

    common = dict(num_workers=args.num_workers, pin_memory=(device == "cuda"))
    if args.num_workers > 0:
        common["persistent_workers"] = True
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          drop_last=True, **common)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **common)

    model = build_model(args.arch, pretrained=not args.no_pretrained,
                        lane_head=args.lane_head).to(device)

    if args.init_from:
        state = torch.load(args.init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        print(f"warm-started from {args.init_from}")
        if missing or unexpected:
            print(f"  missing {len(missing)} / unexpected {len(unexpected)} keys")


    weights = None
    if not args.no_class_weights:
        freq = class_frequencies(args.data_root, "train")
        w = 1.0 / np.sqrt(np.maximum(freq, 1))
        w = w / w.mean()
        weights = torch.tensor(w, dtype=torch.float32, device=device)

    criterion = SegLoss(class_weights=weights,
                        dice_weight=cfg.DEFAULT.train.dice_loss_weight,
                        aux_weight=cfg.DEFAULT.train.aux_loss_weight,
                        lane_weight=args.lane_weight).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)

    steps_per_epoch = len(train_ld)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * cfg.DEFAULT.train.warmup_epochs

    print(f"device={device}  params={count_parameters(model)/1e6:.2f}M  "
          f"train={len(train_ds)}  val={len(val_ds)}  "
          f"steps/epoch={steps_per_epoch}")
    if weights is not None:
        print("class weights:", np.round(weights.cpu().numpy(), 3).tolist())

    history, best_miou, step, start_epoch = [], -1.0, 0, 0


    ckpt_last = out_dir / "last.pt"
    if args.resume and ckpt_last.is_file():
        state = torch.load(ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if "optim" in state:
            optim.load_state_dict(state["optim"])
        start_epoch = int(state.get("epoch", 0))
        best_miou = float(state.get("best_miou", state.get("miou", -1.0)))
        step = start_epoch * steps_per_epoch
        hist_file = out_dir / "history.json"
        if hist_file.is_file():
            history = json.loads(hist_file.read_text())[:start_epoch]
        print(f"resumed from epoch {start_epoch}, best mIoU so far {best_miou:.4f}")

    t_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, seen = 0.0, 0
        t_epoch = time.time()

        for batch in train_ld:
            lr = cosine_lr(step, total_steps, warmup_steps, args.lr)
            for g in optim.param_groups:
                g["lr"] = lr

            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)
            lane_t = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None

            optim.zero_grad(set_to_none=True)
            loss = criterion(model(x), y, lane_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            running += float(loss.detach()) * x.size(0)
            seen += x.size(0)
            step += 1

        train_loss = running / max(1, seen)
        summary = evaluate(model, val_ld, device, lane=args.lane_head)
        epoch_time = time.time() - t_epoch

        history.append({
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_loss,
            "val_miou": summary["miou"],
            "val_drivable_iou": summary["drivable_iou"],
            "val_pixel_acc": summary["pixel_accuracy"],
            "per_class_iou": [float(v) for v in summary["per_class_iou"]],
            "val_lane_iou": float(summary.get("lane_iou", float("nan"))),
            "seconds": epoch_time,
        })

        flag = ""
        if summary["miou"] > best_miou:
            best_miou = summary["miou"]
            torch.save({"model": model.state_dict(), "arch": args.arch,
                        "input_size": [H, W], "num_classes": cfg.NUM_CLASSES,
                        "lane_head": args.lane_head,
                        "epoch": epoch + 1, "miou": best_miou},
                       out_dir / "best.pt")
            flag = "  *best*"


        torch.save({"model": model.state_dict(), "optim": optim.state_dict(),
                    "arch": args.arch, "input_size": [H, W],
                    "num_classes": cfg.NUM_CLASSES, "lane_head": args.lane_head,
                    "epoch": epoch + 1, "miou": summary["miou"],
                    "best_miou": best_miou},
                   out_dir / "last.pt")

        lane_txt = (f"  lane {summary['lane_iou']:.4f}"
                    if args.lane_head and summary.get("lane_iou") == summary.get("lane_iou")
                    else "")
        print(f"epoch {epoch+1:3d}/{args.epochs}  loss {train_loss:.4f}  "
              f"mIoU {summary['miou']:.4f}  drivable {summary['drivable_iou']:.4f}  "
              f"pixAcc {summary['pixel_accuracy']:.4f}{lane_txt}  "
              f"{epoch_time:.1f}s{flag}", flush=True)

        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    print(f"\ntotal training time: {(time.time()-t_start)/60:.1f} min")
    print(f"best val mIoU: {best_miou:.4f}")
    print()
    print(format_table(summary, "Final-epoch validation metrics"))


if __name__ == "__main__":
    main()

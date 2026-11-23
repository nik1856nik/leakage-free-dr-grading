"""Generic supervised training loop for the end-to-end models."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.datasets.fundus import (CLASS_NAMES, TASKS, FundusDataset, balanced_sampler_weights,
                                 class_weights, load_split)
from src.datasets.sources import ROOT
from src.evaluation.metrics import compute_metrics
from src.models.zoo import REGISTRY, build, n_params


@dataclass
class TrainConfig:
    model: str
    dataset: str = "aptos"
    task: str = "5class"
    size: int | None = None            # None -> model default
    enhancement: str | None = None     # None -> model default
    aug: str = "standard"
    epochs: int = 30
    batch_size: int = 16
    accum: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"           # adamw | sgd | adam
    scheduler: str = "cosine"          # cosine | plateau | none
    warmup_epochs: int = 1
    loss: str = "wce"                  # ce | wce | focal
    class_weight_power: float = 0.5
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    sampler: str = "none"              # none | sqrt
    amp: bool = True
    patience: int = 8
    monitor: str = "auto"              # auto -> qwk+macro_f1 (multiclass) / auc (binary)
    num_workers: int = 8
    seed: int = 42
    init_from: str | None = None       # checkpoint path (e.g. P10 stage 2)
    freeze_body: bool = False
    run_name: str | None = None
    head: str = "cls"                  # cls (softmax over grades) | reg (one continuous grade score)
    channels_last: bool = True         # speed only (tensor-core friendly layout); applied to plain CNNs
    extra: dict = field(default_factory=dict)

    def resolved(self):
        _, size, enh, _ = REGISTRY[self.model]
        self.size = self.size or size
        self.enhancement = self.enhancement or enh
        if self.run_name is None:
            self.run_name = f"{self.model}__{self.dataset}__{self.task}__{self.enhancement}_{self.size}_s{self.seed}"
        return self


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma

    def forward(self, logits, y):
        logp = F.log_softmax(logits.float(), 1)
        ce = F.nll_loss(logp, y, weight=self.weight, reduction="none")
        pt = logp.gather(1, y[:, None]).squeeze(1).exp()
        return ((1 - pt) ** self.gamma * ce).mean()


def seed_everything(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def monitor_score(m: dict, n_classes: int, monitor: str) -> float:
    if monitor == "auto":
        return m["roc_auc"] if n_classes == 2 else 0.5 * (m["qwk"] + m["macro_f1"])
    return m[monitor]


# plain 4-D conv nets that benefit from channels_last (MIL / LSTM / ViT keep the default layout)
CL_MODELS = {"effb3_se", "inception_v3", "densenet169", "resnet50_rf", "p8_vgg16"}


def score_to_prob(s: torch.Tensor, n_cls: int = 5, sigma: float = 0.5) -> torch.Tensor:
    """Regression score -> class probabilities (Gaussian around the score; argmax = nearest grade).

    Used only so the usual metrics (AUC etc.) apply; decisions use the raw score + validation thresholds.
    """
    k = torch.arange(n_cls, device=s.device, dtype=torch.float32)
    return F.softmax(-((s.float().view(-1, 1) - k) ** 2) / (2 * sigma ** 2), 1)


def _out_to_prob(o: torch.Tensor) -> torch.Tensor:
    return score_to_prob(o) if o.shape[1] == 1 else F.softmax(o.float(), 1)


@torch.no_grad()
def predict(model, loader, device, amp=True, tta: int = 0, memory_format=torch.contiguous_format,
            return_raw: bool = False):
    """Return (targets, probabilities[, raw]). tta: 0 none, else number of flip variants (<=4).

    raw = the regression score for a 1-output model, else the expected grade sum_k k*p_k.
    """
    model.eval()
    ys, ps, rs = [], [], []
    flips = [lambda x: x, lambda x: x.flip(3), lambda x: x.flip(2), lambda x: x.flip(2).flip(3)]
    for x, y in loader:
        x = x.to(device, non_blocking=True, memory_format=memory_format)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            outs = [model(f(x)).float() for f in (flips[:tta] if tta else flips[:1])]
            p = torch.stack([_out_to_prob(o) for o in outs]).mean(0)
            if outs[0].shape[1] == 1:
                r = torch.stack(outs).mean(0).view(-1)
            else:
                r = (p * torch.arange(p.shape[1], device=p.device)).sum(1)
        ys.append(y.numpy()); ps.append(p.cpu().numpy()); rs.append(r.cpu().numpy())
    out = (np.concatenate(ys), np.concatenate(ps))
    return out + (np.concatenate(rs),) if return_raw else out


def make_loaders(cfg: TrainConfig, splits=("train", "val", "test")):
    out = {}
    for sp in splits:
        df = load_split(cfg.dataset, cfg.task, sp)
        ds = FundusDataset(df, cfg.size, train=(sp == "train"), enhancement=cfg.enhancement, aug=cfg.aug)
        sampler = None
        if sp == "train" and cfg.sampler == "sqrt":
            w = balanced_sampler_weights(ds.targets, 0.5)
            sampler = WeightedRandomSampler(torch.tensor(w), len(w), replacement=True)
        out[sp] = DataLoader(ds, batch_size=cfg.batch_size if sp == "train" else cfg.batch_size * 2,
                             shuffle=(sp == "train" and sampler is None), sampler=sampler,
                             num_workers=cfg.num_workers, pin_memory=True,
                             drop_last=(sp == "train"), persistent_workers=cfg.num_workers > 0)
    return out


def train(cfg: TrainConfig, evaluate_test: bool = False) -> dict:
    cfg.resolved()
    seed_everything(cfg.seed)
    device = torch.device("cuda")
    n_cls = TASKS[cfg.task][0]
    names = CLASS_NAMES[cfg.task]
    run_dir = ROOT / "experiments" / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ROOT / "checkpoints" / f"{cfg.run_name}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    loaders = make_loaders(cfg, ("train", "val") + (("test",) if evaluate_test else ()))
    reg = cfg.head == "reg"
    model = build(cfg.model, 1 if reg else n_cls)
    if cfg.init_from:
        sd = torch.load(cfg.init_from, map_location="cpu")["model"]
        own = model.state_dict()
        if cfg.freeze_body:   # P10 stage 2: new task, never reuse the old head
            sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
        # keep every tensor whose shape matches (a 5-class head is dropped for a 1-output regressor)
        sd = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        print("init_from:", model.load_state_dict(sd, strict=False))
    if cfg.freeze_body and hasattr(model, "reset_head"):
        model.reset_head(n_cls, freeze_body=True)
    model.to(device)
    mf = torch.channels_last if (cfg.channels_last and cfg.model in CL_MODELS) else torch.contiguous_format
    model.to(memory_format=mf)

    tr_targets = loaders["train"].dataset.targets
    w = class_weights(tr_targets, n_cls, cfg.class_weight_power).to(device) if cfg.loss in ("wce", "focal") else None
    if reg:
        crit = nn.SmoothL1Loss()
    else:
        crit = FocalLoss(w, cfg.focal_gamma) if cfg.loss == "focal" else \
            nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)
    crit.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, nesterov=True, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adam":
        opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(loaders["train"]) / cfg.accum)
    total, warm = cfg.epochs * steps_per_epoch, cfg.warmup_epochs * steps_per_epoch
    if cfg.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else
                                                  0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    elif cfg.scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.3, patience=3)
    else:
        sched = None
    scaler = torch.amp.GradScaler(enabled=cfg.amp)

    hist, best, best_ep, bad = [], -1e9, -1, 0
    t_start = time.time()
    for ep in range(cfg.epochs):
        model.train()
        t0, tot, correct, seen = time.time(), 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for it, (x, y) in enumerate(loaders["train"]):
            x, y = x.to(device, non_blocking=True, memory_format=mf), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
                logits = model(x)
                loss = (crit(logits.float().view(-1), y.float()) if reg else crit(logits.float(), y)) / cfg.accum
            scaler.scale(loss).backward()
            if (it + 1) % cfg.accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, 5.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if sched is not None and cfg.scheduler == "cosine":
                    sched.step()
            tot += loss.detach() * cfg.accum * len(y)          # stays on GPU: no per-step sync
            pred = logits.detach().view(-1).round().clamp(0, n_cls - 1).long() if reg else logits.argmax(1)
            correct += (pred == y).sum()
            seen += len(y)
        tot, correct = float(tot), int(correct)
        yv, pv = predict(model, loaders["val"], device, cfg.amp, memory_format=mf)
        mv = compute_metrics(yv, pv, names)
        val_loss = float(F.nll_loss(torch.log(torch.tensor(pv).clamp_min(1e-8)), torch.tensor(yv)).item())
        score = monitor_score(mv, n_cls, cfg.monitor)
        if sched is not None and cfg.scheduler == "plateau":
            sched.step(score)
        row = dict(epoch=ep + 1, train_loss=tot / seen, train_acc=correct / seen, val_loss=val_loss,
                   val_acc=mv["accuracy"], val_macro_f1=mv["macro_f1"], val_qwk=mv["qwk"],
                   val_auc=mv["roc_auc"], score=score, lr=opt.param_groups[0]["lr"],
                   epoch_time=time.time() - t0)
        hist.append(row)
        pd.DataFrame(hist).to_csv(run_dir / "history.csv", index=False)
        flag = ""
        if score > best:
            best, best_ep, bad, flag = score, ep + 1, 0, " *"
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": ep + 1,
                        "val_metrics": mv}, ckpt)
        else:
            bad += 1
        print(f"[{cfg.run_name}] ep {ep+1:02d} loss {row['train_loss']:.4f} acc {row['train_acc']:.3f} | "
              f"val acc {mv['accuracy']:.4f} mF1 {mv['macro_f1']:.4f} qwk {mv['qwk']:.4f} "
              f"auc {mv['roc_auc']:.4f} | {row['epoch_time']:.0f}s{flag}", flush=True)
        if bad >= cfg.patience:
            print(f"early stop at epoch {ep+1} (best {best_ep})")
            break
    train_time = time.time() - t_start

    # reload best, final val predictions
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    summary = dict(run_name=cfg.run_name, model=cfg.model, paper=REGISTRY[cfg.model][3],
                   dataset=cfg.dataset, task=cfg.task, best_epoch=best_ep, best_val_score=best,
                   train_time_s=train_time, params=n_params(model),
                   trainable_params=n_params(model, True))
    yv, pv, rv = predict(model, loaders["val"], device, cfg.amp, return_raw=True)
    summary["val"] = compute_metrics(yv, pv, names)
    np.savez(run_dir / "val_predictions.npz", y=yv, prob=pv, raw=rv,
             uid=loaders["val"].dataset.df.uid.values)
    if evaluate_test:
        summary["test"] = evaluate_split(model, loaders["test"], device, names, run_dir, "test", cfg.amp)
        summary["inference_ms_per_image"] = summary["test"].pop("_ms_per_image")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def evaluate_split(model, loader, device, names, run_dir, tag, amp=True, tta=0):
    torch.cuda.synchronize(); t0 = time.time()
    y, p, r = predict(model, loader, device, amp, tta, return_raw=True)
    torch.cuda.synchronize(); ms = 1000 * (time.time() - t0) / len(y)
    m = compute_metrics(y, p, names)
    m["_ms_per_image"] = ms
    np.savez(Path(run_dir) / f"{tag}_predictions.npz", y=y, prob=p, raw=r, uid=loader.dataset.df.uid.values)
    return m

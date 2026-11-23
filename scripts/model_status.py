"""Live status of the training queue: per-model progress bars and validation metrics.

    uv run python scripts/model_status.py                     # snapshot (APTOS queue)
    uv run python scripts/model_status.py --watch             # refresh every 10 s (Ctrl-C to quit)
    uv run python scripts/model_status.py --dataset eyepacs --watch
    uv run python scripts/model_status.py --tuning --watch    # tuning round vs. baselines
    uv run python scripts/model_status.py --best --watch      # best-model pipeline (current)

All metrics shown are VALIDATION metrics (the test split is untouched until final runs).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
G, Y, R, C, DIM, B, X = "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[2m", "\033[1m", "\033[0m"

# queue order (matches scripts/run_queue.sh): label, paper, queue-name prefix, run-dir globs
MODELS = [
    ("EfficientNetB3+SE", "P4", "effb3_se", ["effb3_se__{ds}__5class__*"]),
    ("Inception-v3", "P7", "inception_v3", ["inception_v3__{ds}__5class__*"]),
    ("DenseNet-169", "P14", "densenet169", ["densenet169__{ds}__5class__*"]),
    ("ResNet50 (CNN stage)", "P11", "resnet50_cnn", ["resnet50_rf__{ds}__5class__*"]),
    ("ResNet50 + RF", "P11", "resnet50_rf", ["resnet50_rf__{ds}__5class"]),
    ("GoogLeNet+ResNet18+SVM", "P6", "hybrid_svm", ["hybrid_svm__{ds}__5class"]),
    ("Light 2-stage CNN", "P10", "light2stage", ["light2stage__{ds}__binary__*", "light2stage__{ds}__severity4__*",
                                                 "light2stage__{ds}__5class_combined_*"]),
    ("MS-CNN+LSTM+Attn", "P15", "mscnn_lstm", ["mscnn_lstm__{ds}__5class__*"]),
    ("ViT-B/16", "P14", "vit_b16", ["vit_b16__{ds}__5class__*"]),
    ("Enhance(AGC)+VGG16", "P8", "p8_vgg16", ["p8_vgg16__{ds}__5class__*"]),
    ("Attention-MIL ResNet18", "P12", "mil_resnet18", ["mil_resnet18__{ds}__5class__*"]),
]
SPARK = "▁▂▃▄▅▆▇█"


def bar(frac, width=22):
    frac = max(0.0, min(1.0, frac))
    full = int(frac * width)
    part = " ▏▎▍▌▋▊▉"[int((frac * width - full) * 8)] if full < width else ""
    return "█" * full + part + " " * (width - full - len(part))


def spark(vals, n=16):
    vals = list(vals)[-n:]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    return "".join(SPARK[int((v - lo) / (hi - lo + 1e-9) * 7)] for v in vals)


def hms(s):
    if s is None:
        return "  --  "
    h, m = divmod(int(s) // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m:2d}m{int(s) % 60:02d}s"


def queue_events(ds):
    ev = {}
    log = EXP / f"queue_{ds}.log"
    if log.exists():
        for line in log.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[1] in ("START", "DONE", "FAIL"):
                ev[parts[2]] = (parts[1], parts[0].strip("[]"))
    return ev


def find(globs, ds):
    out = []
    for g in globs:
        out += sorted(p for p in EXP.glob(g.format(ds=ds)) if p.is_dir())
    return out


def per_class_recall(m, names=("Mild", "Severe", "PDR")):
    pc = m.get("per_class", {})
    return [pc.get(n, {}).get("recall") for n in names]


def fmt(v, best=None, w=6):
    if v is None or v != v:
        return f"{DIM}{'—':>{w}}{X}"
    s = f"{v:{w}.3f}"
    return f"{G}{B}{s}{X}" if best is not None and abs(v - best) < 1e-9 else s


def run_progress(d: Path):
    """(epoch, max_epochs, patience, best_epoch, last_row, score_hist, mean_epoch_s, mtime)."""
    h = d / "history.csv"
    cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    if not h.exists():
        return None, cfg
    df = pd.read_csv(h)
    if df.empty:
        return None, cfg
    best_ep = int(df.loc[df.score.idxmax(), "epoch"])
    ep_t = df.epoch_time.iloc[1:].mean() if len(df) > 1 else df.epoch_time.iloc[0]
    return dict(epoch=int(df.epoch.iloc[-1]), max_epochs=cfg.get("epochs", 0), patience=cfg.get("patience", 0),
                best_epoch=best_ep, last=df.iloc[-1], best=df.loc[df.score.idxmax()], scores=df.score.tolist(),
                epoch_s=ep_t, mtime=h.stat().st_mtime), cfg


def render(ds):
    ev = queue_events(ds)
    lines = [f"{B}DR_Detection — model training status · {ds.upper()} · five-class (validation metrics){X}"
             f"   {time.strftime('%H:%M:%S')}", ""]
    rows, running_block, n_done = [], [], 0
    for label, paper, qname, globs in MODELS:
        dirs = find(globs, ds)
        qstate = ev.get(f"{qname}_{ds}", (None, None))[0]
        summaries = [json.loads((d / "summary.json").read_text()) for d in dirs if (d / "summary.json").exists()]
        final_summary = None
        if qname == "light2stage":
            comb = [d for d in dirs if "combined" in d.name and (d / "summary.json").exists()]
            final_summary = json.loads((comb[0] / "summary.json").read_text()) if comb else None
        elif summaries:
            final_summary = summaries[-1]
        if qstate == "FAIL":
            status = "failed"
        elif final_summary is not None and (qstate in ("DONE", None) or qname != "light2stage"):
            status = "done"
        elif qstate == "START" or any((d / "history.csv").exists() for d in dirs):
            status = "running"
        else:
            status = "queued"
        if status == "done":
            n_done += 1
            v = final_summary.get("val", {})
            rows.append(dict(label=label, paper=paper, status=status, acc=v.get("accuracy"), mf1=v.get("macro_f1"),
                             qwk=v.get("qwk"), auc=v.get("roc_auc"), rec=per_class_recall(v),
                             params=final_summary.get("params"), t=final_summary.get("train_time_s"),
                             ep=final_summary.get("best_epoch")))
            continue
        rows.append(dict(label=label, paper=paper, status=status))
        if status == "running":
            # pick the active run dir (most recently modified history)
            active = [d for d in dirs if (d / "history.csv").exists() and not (d / "summary.json").exists()]
            stage = ""
            if qname == "light2stage" and active:
                stage = " · stage 1 (binary)" if "binary" in active[-1].name else " · stage 2 (severity, frozen body)"
            if not active:
                running_block += [f"{Y}▶ {label} ({paper}){stage}{X}  fitting classifier / extracting features…"]
                continue
            pr, cfg = run_progress(active[-1])
            if pr is None:
                running_block += [f"{Y}▶ {label} ({paper}){stage}{X}  starting (first epoch, cudnn warm-up)…"]
                continue
            e, E, pat, be = pr["epoch"], pr["max_epochs"], pr["patience"], pr["best_epoch"]
            left_cap = min(E - e, max(0, pat - (e - be)))  # early-stop bound
            last, best = pr["last"], pr["best"]
            idle = time.time() - pr["mtime"]
            running_block += [
                f"{Y}▶ {label} ({paper}){stage}{X}   {C}{cfg.get('size')}px · {cfg.get('enhancement')} · "
                f"{cfg.get('optimizer')} lr {cfg.get('lr')} · bs {cfg.get('batch_size')} · {cfg.get('loss')}{X}",
                f"  epoch {e:3d}/{E:<3d} {bar(e / max(E, 1))} {100 * e / max(E, 1):5.1f}%   "
                f"{hms(pr['epoch_s'])}/epoch   ETA {hms(left_cap * pr['epoch_s']).strip()}–"
                f"{hms((E - e) * pr['epoch_s']).strip()} "
                f"{DIM}(early stop after {pat} epochs w/o improvement; {e - be} so far){X}",
                f"  last  ep {e:3d}: acc {last.val_acc:.3f}  mF1 {last.val_macro_f1:.3f}  QWK {last.val_qwk:.3f}  "
                f"AUC {last.val_auc:.3f}  train-loss {last.train_loss:.3f}  val-loss {last.val_loss:.3f}",
                f"  best  ep {be:3d}: acc {best.val_acc:.3f}  mF1 {best.val_macro_f1:.3f}  {B}QWK {best.val_qwk:.3f}{X}  "
                f"AUC {best.val_auc:.3f}   score trend {C}{spark(pr['scores'])}{X}"
                + (f"   {R}no update for {hms(idle)}{X}" if idle > 600 else ""),
            ]

    done = [r for r in rows if r["status"] == "done"]
    bests = {k: max((r[k] for r in done if r.get(k) is not None), default=None) for k in ("acc", "mf1", "qwk", "auc")}
    bests_rec = [max((r["rec"][i] for r in done if r["rec"][i] is not None), default=None) for i in range(3)]
    lines.append(f"queue  {bar(n_done / len(MODELS), 30)} {n_done}/{len(MODELS)} finished")
    lines.append("")
    lines += running_block or [f"{DIM}no model currently training{X}"]
    lines.append("")
    hdr = (f"   {'Model':24s} {'Paper':5s} {'Acc':>6s} {'mF1':>6s} {'QWK':>6s} {'AUC':>6s} │ "
           f"{'Mild':>6s} {'Sev':>6s} {'PDR':>6s} │ {'best ep':>7s} {'time':>7s} {'params':>7s}")
    lines += [f"{B}{hdr}{X}", f"   {'':24s} {'':5s} {DIM}{'— validation —':^27s}{X}   {DIM}{'recall':^20s}{X}"]
    for r in rows:
        icon = {"done": f"{G}✔{X}", "running": f"{Y}▶{X}", "queued": f"{DIM}·{X}", "failed": f"{R}✘{X}"}[r["status"]]
        if r["status"] == "done":
            p = f"{r['params'] / 1e6:5.1f}M" if r.get("params") else "—"
            lines.append(f" {icon} {r['label']:24s} {r['paper']:5s} {fmt(r['acc'], bests['acc'])} {fmt(r['mf1'], bests['mf1'])} "
                         f"{fmt(r['qwk'], bests['qwk'])} {fmt(r['auc'], bests['auc'])} │ "
                         + " ".join(fmt(v, bests_rec[i]) for i, v in enumerate(r["rec"]))
                         + f" │ {str(r.get('ep') or '—'):>7s} {hms(r.get('t')):>7s} {p:>7s}")
        else:
            note = {"running": f"{Y}training…{X}", "queued": f"{DIM}queued{X}",
                    "failed": f"{R}failed — see experiments/logs/{r['label']}{X}"}[r["status"]]
            lines.append(f" {icon} {r['label']:24s} {r['paper']:5s} {note}")
    lines.append("")
    lines.append(f"{DIM}best value per column in {X}{G}{B}green{X}{DIM}. Clinical ranking uses macro-F1, QWK and "
                 f"minority-class recall, not accuracy alone.{X}")
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.split(",")
        lines.append(f"{DIM}GPU {gpu[0].strip()}% · {int(gpu[1]) / 1024:.1f}/{int(gpu[2]) / 1024:.1f} GB · "
                     f"{gpu[3].strip()}°C{X}")
    except Exception:  # noqa: BLE001
        pass
    return lines


TUNE_MODELS = [("ResNet50 (P11)", "resnet50_rf"), ("DenseNet-169 (P14)", "densenet169"), ("EffB3+SE (P4)", "effb3_se")]


def _score(v):
    return 0.5 * (v["qwk"] + v["macro_f1"]) if v else None


def _running_lines(d: Path, title: str):
    pr, cfg = run_progress(d)
    if pr is None:
        return [f"{Y}▶ {title}{X}  starting (first epoch, cudnn warm-up)…"]
    e, E, pat, be = pr["epoch"], pr["max_epochs"], pr["patience"], pr["best_epoch"]
    left_cap = min(E - e, max(0, pat - (e - be)))
    last, best = pr["last"], pr["best"]
    idle = time.time() - pr["mtime"]
    return [
        f"{Y}▶ {title}{X}   {C}{cfg.get('size')}px · {cfg.get('enhancement')} · {cfg.get('loss')} · "
        f"sampler {cfg.get('sampler')} · bs {cfg.get('batch_size')}×{cfg.get('accum')}{X}",
        f"  epoch {e:3d}/{E:<3d} {bar(e / max(E, 1))} {100 * e / max(E, 1):5.1f}%   {hms(pr['epoch_s'])}/epoch   "
        f"ETA {hms(left_cap * pr['epoch_s']).strip()}–{hms((E - e) * pr['epoch_s']).strip()}",
        f"  last  ep {e:3d}: acc {last.val_acc:.3f}  mF1 {last.val_macro_f1:.3f}  QWK {last.val_qwk:.3f}  AUC {last.val_auc:.3f}",
        f"  best  ep {be:3d}: acc {best.val_acc:.3f}  mF1 {best.val_macro_f1:.3f}  {B}QWK {best.val_qwk:.3f}{X}  "
        f"AUC {best.val_auc:.3f}   score trend {C}{spark(pr['scores'])}{X}"
        + (f"   {R}no update for {hms(idle)}{X}" if idle > 600 else ""),
    ]


def render_tuning(ds):
    lines = [f"{B}DR_Detection — tuning · {ds.upper()} · top-3 models (validation metrics){X}"
             f"   {time.strftime('%H:%M:%S')}", ""]
    tdirs = sorted(p for p in EXP.glob(f"tune__*__{ds}") if p.is_dir())
    n_done = sum((d / "summary.json").exists() for d in tdirs)
    lines += [f"queue  {bar(n_done / max(len(tdirs), 1), 30)} {n_done}/{len(tdirs)} finished", ""]
    active = [d for d in tdirs if not (d / "summary.json").exists()]
    run_block = []
    for d in active:
        title = d.name.replace("tune__", "").replace(f"__{ds}", "").replace("__", " · ")
        run_block += _running_lines(d, title) if (d / "history.csv").exists() else [f"{Y}▶ {title}{X}  starting…"]
    lines += run_block or [f"{DIM}no tuning run currently training{X}"]
    lines.append("")
    hdr = (f"   {'Model / variant':30s} {'Acc':>6s} {'mF1':>6s} {'QWK':>6s} {'AUC':>6s} │ {'Mild':>6s} {'Sev':>6s} "
           f"{'PDR':>6s} │ {'score':>6s} {'Δ base':>7s} │ {'ep':>3s} {'time':>7s}")
    lines += [f"{B}{hdr}{X}"]
    for label, m in TUNE_MODELS:
        base = find([f"{m}__{{ds}}__5class__*"], ds)
        bs = json.loads((base[-1] / "summary.json").read_text()) if base and (base[-1] / "summary.json").exists() else None
        b_score = _score(bs["val"]) if bs else None
        group = [(f"  baseline ({bs['config']['size'] if bs and 'config' in bs else ''})".rstrip(" ()"), bs)] if bs else []
        for d in sorted(EXP.glob(f"tune__{m}__*__{ds}")):
            var = d.name.split("__")[2]
            s = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else None
            group.append((f"  {var}", s if s else ("running" if (d / "history.csv").exists() else "starting")))
        scored = [g[1]["val"] for g in group if isinstance(g[1], dict)]
        best_sc = max((_score(v) for v in scored), default=None)
        lines.append(f" {B}{label}{X}")
        for name, s in group:
            if not isinstance(s, dict):
                lines.append(f" {Y}▶{X} {name:30s} {Y}{s}…{X}")
                continue
            v = s["val"]
            sc = _score(v)
            dl = "" if name.strip() == "baseline" else f"{(G if sc > b_score else R)}{sc - b_score:+7.3f}{X}"
            mark = f"{G}★{X}" if abs(sc - best_sc) < 1e-9 else " "
            lines.append(f" {mark} {name:30s} {fmt(v['accuracy'])} {fmt(v['macro_f1'])} {fmt(v['qwk'])} {fmt(v.get('roc_auc'))} │ "
                         + " ".join(fmt(r) for r in per_class_recall(v))
                         + f" │ {fmt(sc)} {dl:>7s} │ {str(s.get('best_epoch') or '—'):>3s} {hms(s.get('train_time_s')):>7s}")
        lines.append("")
    lines.append(f"{DIM}score = 0.5·(QWK + macro-F1), the model-selection criterion. ★ = best variant per model. "
                 f"Δ base = change vs. that model's baseline.{X}")
    return lines


def render_best(ds):
    """Best-model pipeline: EyePACS pretraining -> APTOS fine-tunes -> ensemble selection."""
    lines = [f"{B}DR_Detection — best-model pipeline · {ds.upper()} five-class (validation metrics){X}"
             f"   {time.strftime('%H:%M:%S')}", ""]
    pre = [("EffB3+SE", "effb3_se__eyepacs__5class__*"), ("DenseNet-169", "densenet169__eyepacs__5class__*"),
           ("ResNet50", "resnet50_rf__eyepacs__5class__*")]
    ft = [(f"{m} {h}", f"ft__{k}_{h}__{ds}") for m, k in (("ResNet50", "resnet50"), ("DenseNet-169", "densenet"),
                                                         ("EffB3+SE", "effb3")) for h in ("cls", "reg")]
    hdr = (f"   {'Run':28s} {'Acc':>6s} {'mF1':>6s} {'QWK':>6s} {'AUC':>6s} │ {'Mild':>6s} {'Sev':>6s} {'PDR':>6s} │ "
           f"{'score':>6s} {'ep':>3s} {'time':>7s}")
    running = []

    def row(label, d):
        if d is None:
            return f" {DIM}· {label:28s} queued{X}"
        if (d / "summary.json").exists():
            s = json.loads((d / "summary.json").read_text()); v = s["val"]
            return (f" {G}✔{X} {label:28s} {fmt(v['accuracy'])} {fmt(v['macro_f1'])} {fmt(v['qwk'])} "
                    f"{fmt(v.get('roc_auc'))} │ " + " ".join(fmt(r) for r in per_class_recall(v))
                    + f" │ {fmt(0.5 * (v['qwk'] + v['macro_f1']))} {str(s.get('best_epoch')):>3s} "
                    f"{hms(s.get('train_time_s')):>7s}")
        running.extend(_running_lines(d, label) if (d / "history.csv").exists() else [f"{Y}▶ {label}{X} starting…"])
        return f" {Y}▶{X} {label:28s} {Y}training…{X}"

    body = [f"{B}1 · pretraining on EyePACS (24.6k images){X}", f"{B}{hdr}{X}"]
    for label, g in pre:
        ds_ = sorted(p for p in EXP.glob(g) if p.is_dir())
        body.append(row(label, ds_[-1] if ds_ else None))
    body += ["", f"{B}2 · fine-tuning on APTOS from EyePACS weights{X}", f"{B}{hdr}{X}"]
    for label, name in ft:
        d = EXP / name
        body.append(row(label, d if d.exists() else None))
    body += ["", f"{B}3 · ensemble selection (5-fold cross-fitted validation, thresholds fitted on val){X}"]
    sel = EXP / f"best_{ds}" / "selection.json"
    log = EXP / f"best_{ds}.log"
    if sel.exists() and log.exists() and "SELECTION" in log.read_text():
        s = json.loads(sel.read_text()); v = s["val_cv"]
        body += [f"  {G}selected:{X} " + " + ".join(Path(m).stem for m in s["members"]),
                 f"  val (cross-fitted): acc {v['accuracy']:.3f}  mF1 {v['macro_f1']:.3f}  {B}QWK {v['qwk']:.3f}{X}  "
                 f"AUC {v['roc_auc']:.3f}  │ referable DR: sens {v['referable_sensitivity']:.3f}  "
                 f"spec {v['referable_specificity']:.3f}"]
        fr = EXP / f"best_{ds}" / "final_results.json"
        if fr.exists():
            for k, m in json.loads(fr.read_text())["results"].items():
                body.append(f"  {C}TEST {k:16s}{X} acc {m['accuracy']:.3f}  mF1 {m['macro_f1']:.3f}  QWK {m['qwk']:.3f}  "
                            f"│ referable: sens {m['referable_sensitivity']:.3f} spec {m['referable_specificity']:.3f}")
    else:
        body.append(f"  {DIM}waiting for the fine-tunes{X}")
    lines += (running or [f"{DIM}no run currently training{X}"]) + [""] + body
    lines += ["", f"{DIM}score = 0.5·(QWK + macro-F1). The test split is read only once, after selection.{X}"]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aptos")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=float, default=10)
    ap.add_argument("--tuning", action="store_true", help="show the tuning round instead of the baseline queue")
    ap.add_argument("--best", action="store_true", help="show the best-model pipeline")
    a = ap.parse_args()
    view = render_best if a.best else render_tuning if a.tuning else render
    if not a.watch:
        print("\n".join(view(a.dataset)))
        return
    try:
        while True:
            sys.stdout.write("\033[H\033[2J" + "\n".join(view(a.dataset)) + "\n")
            sys.stdout.flush()
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()

"""Progress bars for the dataset downloads (APTOS 2019, EyePACS train parts) + Messidor-2/IDRiD status.

    uv run python scripts/download_status.py            # one snapshot
    uv run python scripts/download_status.py --watch    # live view, refresh every 5 s (Ctrl-C to quit)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
GB = 1e9

APTOS_ZIP = RAW / "aptos2019" / "aptos2019-blindness-detection.zip"
APTOS_SIZE = 10_215_289_875                      # from Kaggle's download metadata
EYEPACS_PARTS = {f"train.zip.00{i}": 8_388_608_000 for i in range(1, 5)}
EYEPACS_PARTS["train.zip.005"] = 1_434_013_506   # sizes from `kaggle competitions files`

GREEN, YELLOW, DIM, BOLD, RESET = "\033[32m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"


def bar(frac: float, width: int = 32) -> str:
    frac = max(0.0, min(1.0, frac))
    full = int(frac * width)
    part = " ▏▎▍▌▋▊▉"[int((frac * width - full) * 8)] if full < width else ""
    return "█" * full + part + " " * (width - full - len(part))


def fmt_eta(sec: float | None) -> str:
    if sec is None or sec <= 0 or sec == float("inf"):
        return "  --:--"
    h, m = divmod(int(sec) // 60, 60)
    return f"{h:2d}h{m:02d}m" if h else f"   {m:2d}m"


def item_state(path: Path, expected: int) -> tuple[int, str]:
    """Return (bytes_on_disk, state) where state is done | downloading | unwrapping | queued.

    EyePACS parts arrive wrapped (train.zip.00N.zip, ~2.5 MB larger) and are then unwrapped.
    """
    wrapped = Path(str(path) + ".zip")
    if wrapped.exists():
        size = wrapped.stat().st_size
        if Path(str(wrapped) + ".kaggle-partial").exists():
            return min(size, expected), "downloading"
        return expected, "unwrapping"
    partial = Path(str(path) + ".kaggle-partial")
    if path.exists():
        size = path.stat().st_size
        if not partial.exists() and size >= expected:
            return size, "done"
        return size, "downloading"
    return 0, "queued"


def snapshot(prev: dict, dt: float) -> tuple[list[str], dict]:
    lines, cur = [], {}
    lines.append(f"{BOLD}DR_Detection — dataset downloads{RESET}   {time.strftime('%H:%M:%S')}")
    lines.append("")

    # ---------------- APTOS
    if (RAW / "aptos2019" / "train.csv").exists() and not APTOS_ZIP.exists():
        n = len(list((RAW / "aptos2019" / "train_images").glob("*.png")))
        lines.append(f"{GREEN}✔{RESET} APTOS 2019        {bar(1)} 100%  extracted: {n} train images")
        aptos_left = 0
    else:
        got, st = item_state(APTOS_ZIP, APTOS_SIZE)
        cur["aptos"] = got
        if st == "done":
            lines.append(f"{YELLOW}…{RESET} APTOS 2019        {bar(1)} 100%  downloaded, extracting")
            aptos_left = 0
        else:
            spd = (got - prev.get("aptos", got)) / dt if dt else 0
            cur["aptos_speed"] = spd
            aptos_left = APTOS_SIZE - got
            eta = aptos_left / spd if spd > 0 else None
            lines.append(f"{YELLOW}↓{RESET} APTOS 2019        {bar(got / APTOS_SIZE)} {100*got/APTOS_SIZE:5.1f}%  "
                         f"{got/GB:5.2f}/{APTOS_SIZE/GB:.2f} GB  {spd/1e6:5.2f} MB/s  ETA {fmt_eta(eta)}")

    # ---------------- EyePACS
    tot_exp = sum(EYEPACS_PARTS.values())
    tot_got, ep_speed, part_lines = 0, 0.0, []
    for name, exp in EYEPACS_PARTS.items():
        got, st = item_state(RAW / "eyepacs" / name, exp)
        tot_got += got
        cur[name] = got
        if st == "downloading":
            ep_speed = (got - prev.get(name, got)) / dt if dt else 0
        icon = {"done": f"{GREEN}✔{RESET}", "downloading": f"{YELLOW}↓{RESET}", "unwrapping": f"{YELLOW}…{RESET}", "queued": f"{DIM}·{RESET}"}[st]
        part_lines.append(f"    {icon} {name:14s}{bar(got / exp, 24)} {100*got/exp:5.1f}%  {got/GB:5.2f}/{exp/GB:.2f} GB"
                          + (f"  {ep_speed/1e6:5.2f} MB/s" if st == "downloading" else f"  {DIM}{st}{RESET}"))
    labels = (RAW / "eyepacs" / "trainLabels.csv").exists()
    speed_for_eta = ep_speed or cur.get("aptos_speed", 0) or prev.get("last_speed", 0)
    cur["last_speed"] = speed_for_eta
    ep_eta = (tot_exp - tot_got + aptos_left) / speed_for_eta if speed_for_eta > 0 else None
    state = f"{GREEN}✔{RESET}" if tot_got >= tot_exp else f"{YELLOW}↓{RESET}" if tot_got else f"{DIM}·{RESET}"
    lines.append(f"{state} EyePACS train     {bar(tot_got / tot_exp)} {100*tot_got/tot_exp:5.1f}%  "
                 f"{tot_got/GB:5.2f}/{tot_exp/GB:.2f} GB  ETA {fmt_eta(ep_eta)}"
                 f"{'' if ep_speed else f'  {DIM}(after APTOS){RESET}' if tot_got < tot_exp else ''}")
    lines.extend(part_lines)
    lines.append(f"    {GREEN + '✔' + RESET if labels else DIM + '·' + RESET} trainLabels.csv")

    # ---------------- already-local datasets
    m2 = RAW / "messidor2"
    n_m2 = len(os.listdir(m2 / "IMAGES")) if (m2 / "IMAGES").exists() else 0
    ok_m2 = n_m2 == 1748 and (m2 / "messidor_data.csv").exists()
    lines.append(f"{GREEN + '✔' + RESET if ok_m2 else YELLOW + '!' + RESET} Messidor-2        {bar(n_m2 / 1748)} "
                 f"{100*n_m2/1748:5.1f}%  {n_m2}/1748 images, grades {'✔' if (m2 / 'messidor_data.csv').exists() else '✘'}")
    idr = RAW / "idrid" / "B. Disease Grading" / "1. Original Images"
    n_id = sum(len(os.listdir(p)) for p in idr.iterdir()) if idr.exists() else 0
    lines.append(f"{GREEN + '✔' + RESET if n_id == 516 else YELLOW + '!' + RESET} IDRiD             "
                 f"{bar(n_id / 516)} {100*n_id/516:5.1f}%  {n_id}/516 images")

    free = shutil.disk_usage(RAW).free / GB
    lines.append("")
    lines.append(f"{DIM}disk free: {free:.1f} GB   (speeds averaged over the refresh interval){RESET}")
    return lines, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=float, default=5.0)
    a = ap.parse_args()
    prev, t_prev = {}, time.time()
    if not a.watch:  # single snapshot: sample twice to measure speed
        _, prev = snapshot({}, 0)
        time.sleep(3)
        lines, _ = snapshot(prev, time.time() - t_prev)
        print("\n".join(lines))
        return
    try:
        while True:
            now = time.time()
            lines, prev = snapshot(prev, now - t_prev)
            t_prev = now
            sys.stdout.write("\033[H\033[2J" + "\n".join(lines) + "\n")
            sys.stdout.flush()
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()

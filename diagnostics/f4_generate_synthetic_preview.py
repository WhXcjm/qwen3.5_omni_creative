#!/usr/bin/env python3
"""Generate deterministic fake MindSpeed-MM logs for the F4 HTML preview.

These curves are diagnostic fixtures, not experimental evidence. Their common
backbone is ``3.05 + 14 / (iteration + 30) ** 0.55``. Run-specific offsets
make F1 overtake AdamW around step 100 and F2 overtake F1 around step 177;
F2's advantage peaks near step 300 and then decays. Shared and run-local noise
exercise the smoothing and equal-loss threshold logic.
"""

from __future__ import annotations

import argparse
import math
import random
from datetime import datetime, timedelta
from pathlib import Path


def learning_rate(iteration: int) -> float:
    if iteration <= 50:
        return 4e-4 * iteration / 50
    progress = (iteration - 50) / (5000 - 50)
    return 4e-5 + 0.5 * (4e-4 - 4e-5) * (1 + math.cos(math.pi * progress))


def f1_minus_adamw(iteration: int) -> float:
    return 0.20 * math.exp(-iteration / 60) - 0.07 * (
        1 - math.exp(-iteration / 130)
    )


def f2_minus_f1(iteration: int) -> float:
    if iteration <= 177:
        return 0.12 * (1 - iteration / 177)
    advantage = -0.112 * (1 - math.exp(-(iteration - 177) / 45))
    if iteration > 300:
        advantage *= math.exp(-(iteration - 300) / 360)
    return advantage


def write_log(path: Path, losses: list[float]) -> None:
    start = datetime(2026, 8, 5, 1, 0, 0)
    lines: list[str] = []
    for iteration, loss in enumerate(losses, start=1):
        timestamp = start + timedelta(seconds=52 * iteration)
        mtp_loss = loss + 0.18 + 0.006 * math.sin(iteration / 19)
        aux_loss = 16.0 + 0.18 * math.sin(iteration / 37)
        grad_norm = 0.38 + 0.035 * math.sin(iteration / 23)
        lines.append(
            "[Rank 0 | Local Rank 0] "
            f"{timestamp:%Y-%m-%d %H:%M:%S},000 INFO "
            "[mindspeed_mm.fsdp.train.train_engine:47] =>  "
            f"[{timestamp:%Y-%m-%d %H:%M:%S}] iteration {iteration:8d}/    5000 | "
            f"consumed samples: {iteration * 256:12d} | "
            "elapsed time per iteration (ms): 52000.0 | "
            f"learning rate: {learning_rate(iteration):.6E} | "
            "global batch size:   256 | "
            f"loss: {loss:.6E} | mtp_1 loss: {mtp_loss:.6E} | "
            f"aux loss: {aux_loss:.6E} | grad norm: {grad_norm:.3f} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    shared_rng = random.Random(20260805)
    local_rngs = {
        "adamw": random.Random(11),
        "f1": random.Random(22),
        "f2": random.Random(33),
    }
    curves = {name: [] for name in local_rngs}
    for iteration in range(1, args.iterations + 1):
        backbone = 3.05 + 14 / (iteration + 30) ** 0.55
        shared_noise = (
            0.035 * math.sin(iteration / 8)
            + 0.018 * math.sin(iteration / 31)
            + shared_rng.gauss(0, 0.024)
        )
        offsets = {
            "adamw": 0.0,
            "f1": f1_minus_adamw(iteration),
            "f2": f1_minus_adamw(iteration) + f2_minus_f1(iteration),
        }
        for name, rng in local_rngs.items():
            curves[name].append(
                backbone + shared_noise + offsets[name] + rng.gauss(0, 0.008)
            )

    for name, losses in curves.items():
        write_log(args.output / f"{name}.log", losses)
    (args.output / "SYNTHETIC_DATA.txt").write_text(
        __doc__ + "\nseed=20260805; iterations=" + str(args.iterations) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

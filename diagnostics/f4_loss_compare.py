#!/usr/bin/env python3
"""Extract and compare MindSpeed-MM Rank-0 training logs.

The first input run is the reference.  Pairwise loss differences are reported
as ``candidate - reference``; negative values therefore mean that the
candidate has lower loss.

Only Python's standard library is required.  The default smoothing settings
are intended for noisy, per-step training loss:

* bias-aware EMA with a 20-iteration half-life;
* trailing mean over the latest 25 iterations;
* no centered mean unless explicitly requested (it uses future observations).

Usage example (PowerShell, run from the repository/worktree root):

    uv run --no-project diagnostics/f4_loss_compare.py `
      --run "AdamW=D:/logs/adamw.log" `
      --run "WrongMuon=D:/logs/wrong_muon.log" `
      --run "F1=D:/logs/f1.log" `
      --run "F2=D:/logs/f2.log" `
      --baseline F1 `
      --ema-halflife 20 --curve ema --confirm-points 3 `
      --at-step 177 --at-step 300 --at-step 700 `
      --csv f4_loss_h20.csv --html f4_loss_h20.html

Repeat with ``--ema-halflife 50`` and different output filenames for a more
heavily smoothed robustness check. To compare selected equal-loss levels,
repeat ``--threshold`` or use a comma-separated list; omit it to auto-select
shared levels. Relative paths and quoted globs resolve from the current working
directory. Without ``NAME=``, filenames become run names and collisions receive
``#2``, ``#3``, and so on.
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import math
import re
import statistics
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


NUMBER = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?|inf|nan)"


USAGE_EXAMPLES = r"""examples (PowerShell):
  Multi-run comparison with raw traces, EMA smoothing, CSV, and HTML output:

    uv run --no-project diagnostics/f4_loss_compare.py `
      --run "AdamW=D:\logs\adamw.log" `
      --run "WrongMuon=D:\logs\wrong_muon.log" `
      --run "F1=D:\logs\f1.log" `
      --run "F2=D:\logs\f2.log" `
      --baseline F1 `
      --ema-halflife 20 --curve ema --confirm-points 3 `
      --at-step 177 --at-step 300 --at-step 700 `
      --csv f4_loss_h20.csv --html f4_loss_h20.html

  Explicit equal-loss levels (omit --threshold to auto-select shared levels):

    uv run --no-project f4_loss_compare.py `
      --run "adamw=adam_lr4e-4_5k-its.log" `
      --run "r1=muon_lr4e-4_r1_1k-5k-its.log" --run "r2=muon_lr4e-4_r2_1k-5k-its.log" `
      --baseline adamw --ema-halflife 20 --curve ema --confirm-points 3 `
      --threshold 3.8 --threshold 3.7 --threshold 3.6 `
      --threshold 3.5 --threshold 3.4 `
      --html f1_f2_equal_loss.html

  Relative paths/globs with filename-based names. Quote the glob so uv receives
  one argument; duplicate filenames become name#2, name#3:

    uv run --no-project f4_loss_compare.py `
      -run "local_logs/*.log" --baseline "F1.log" `
      --threshold 3.8,3.7,3.6,3.5,3.4 `
      --html f4_loss_auto_named.html

  Robustness check: repeat the first command with --ema-halflife 50 and a
  different output filename. Positive step lead means earlier than baseline;
  negative candidate-minus-baseline loss gap means lower loss.
"""


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Preserve example layout while retaining default values in --help."""


def _field_pattern(label: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?:^|\|)\s*{label}\s*:\s*({NUMBER})",
        flags=re.IGNORECASE,
    )


ITERATION_RE = re.compile(
    r"\biteration\s+(\d+)\s*/\s*(\d+)", flags=re.IGNORECASE
)
CONSUMED_RE = _field_pattern(r"consumed\s+samples")
ELAPSED_RE = _field_pattern(r"elapsed\s+time\s+per\s+iteration\s*\(ms\)")
LR_RE = _field_pattern(r"learning\s+rate")
GLOBAL_BATCH_RE = _field_pattern(r"global\s+batch\s+size")
LOSS_RE = _field_pattern(r"loss")
AUX_LOSS_RE = _field_pattern(r"aux\s+loss")
GRAD_NORM_RE = _field_pattern(r"grad\s+norm")
MTP_LOSS_RE = re.compile(
    rf"(?:^|\|)\s*mtp_(\d+)\s+loss\s*:\s*({NUMBER})",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class TrainingRecord:
    iteration: int
    total_iterations: int
    consumed_samples: int | None
    elapsed_ms: float | None
    learning_rate: float | None
    global_batch_size: int | None
    loss: float | None
    mtp_losses: Mapping[int, float] = field(default_factory=dict)
    aux_loss: float | None = None
    grad_norm: float | None = None
    line_number: int = 0


@dataclass
class RunData:
    name: str
    path: Path
    records: list[TrainingRecord]
    warnings: list[str]


@dataclass(frozen=True)
class SmoothedPoint:
    iteration: int
    raw: float
    ema: float
    rolling: float
    centered: float | None


def _search_float(pattern: re.Pattern[str], line: str) -> float | None:
    match = pattern.search(line)
    return float(match.group(1)) if match else None


def _search_int(pattern: re.Pattern[str], line: str) -> int | None:
    value = _search_float(pattern, line)
    return int(value) if value is not None else None


def parse_line(line: str, line_number: int = 0) -> TrainingRecord | None:
    """Parse one training status line, returning None for unrelated lines."""

    iteration_match = ITERATION_RE.search(line)
    if not iteration_match:
        return None

    mtp_losses = {
        int(match.group(1)): float(match.group(2))
        for match in MTP_LOSS_RE.finditer(line)
    }
    return TrainingRecord(
        iteration=int(iteration_match.group(1)),
        total_iterations=int(iteration_match.group(2)),
        consumed_samples=_search_int(CONSUMED_RE, line),
        elapsed_ms=_search_float(ELAPSED_RE, line),
        learning_rate=_search_float(LR_RE, line),
        global_batch_size=_search_int(GLOBAL_BATCH_RE, line),
        loss=_search_float(LOSS_RE, line),
        mtp_losses=mtp_losses,
        aux_loss=_search_float(AUX_LOSS_RE, line),
        grad_norm=_search_float(GRAD_NORM_RE, line),
        line_number=line_number,
    )


def _compact_ranges(values: Sequence[int], limit: int = 8) -> str:
    if not values:
        return ""
    ranges: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append((start, previous))
        start = previous = value
    ranges.append((start, previous))
    rendered = [str(a) if a == b else f"{a}-{b}" for a, b in ranges[:limit]]
    if len(ranges) > limit:
        rendered.append(f"... ({len(ranges) - limit} more ranges)")
    return ", ".join(rendered)


def load_run(path: Path, name: str | None = None) -> RunData:
    """Load a log and canonicalize duplicate iterations by keeping the last."""

    parsed: list[TrainingRecord] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = parse_line(line, line_number)
            if record is not None:
                parsed.append(record)

    warnings: list[str] = []
    if not parsed:
        return RunData(name or path.stem, path, [], ["no training records found"])

    backwards = [
        (previous.iteration, current.iteration, current.line_number)
        for previous, current in zip(parsed, parsed[1:])
        if current.iteration < previous.iteration
    ]
    if backwards:
        preview = ", ".join(
            f"{old}->{new} at line {line}" for old, new, line in backwards[:5]
        )
        warnings.append(f"non-monotonic iteration order: {preview}")

    by_iteration: dict[int, TrainingRecord] = {}
    duplicate_iterations: list[int] = []
    for record in parsed:
        if record.iteration in by_iteration:
            duplicate_iterations.append(record.iteration)
        by_iteration[record.iteration] = record
    if duplicate_iterations:
        unique = sorted(set(duplicate_iterations))
        warnings.append(
            "duplicate iterations (last occurrence kept): " + _compact_ranges(unique)
        )

    records = sorted(by_iteration.values(), key=lambda item: item.iteration)
    if records[0].iteration > 1:
        warnings.append(
            f"parsed log starts at iteration {records[0].iteration}; "
            "threshold reaches at the first parsed point are left-censored "
            "(the true reach may be earlier)"
        )
    present = {record.iteration for record in records}
    missing = [
        iteration
        for iteration in range(records[0].iteration, records[-1].iteration + 1)
        if iteration not in present
    ]
    if missing:
        warnings.append("missing iterations: " + _compact_ranges(missing))

    total_iterations = sorted({record.total_iterations for record in records})
    if len(total_iterations) > 1:
        warnings.append(f"total-iteration target changes: {total_iterations}")

    batch_sizes = sorted(
        {
            record.global_batch_size
            for record in records
            if record.global_batch_size is not None
        }
    )
    if len(batch_sizes) > 1:
        warnings.append(f"global batch size changes: {batch_sizes}")

    for previous, current in zip(records, records[1:]):
        step_delta = current.iteration - previous.iteration
        if (
            previous.consumed_samples is not None
            and current.consumed_samples is not None
            and current.global_batch_size is not None
        ):
            actual = current.consumed_samples - previous.consumed_samples
            expected = current.global_batch_size * step_delta
            if actual != expected:
                warnings.append(
                    "consumed-sample increment mismatch at "
                    f"{previous.iteration}->{current.iteration}: "
                    f"actual {actual}, expected {expected}"
                )
        if current.learning_rate is not None:
            if not math.isfinite(current.learning_rate) or current.learning_rate < 0:
                warnings.append(
                    f"invalid learning rate at iteration {current.iteration}: "
                    f"{current.learning_rate}"
                )
        if current.elapsed_ms is not None and (
            not math.isfinite(current.elapsed_ms) or current.elapsed_ms <= 0
        ):
            warnings.append(
                f"invalid elapsed time at iteration {current.iteration}: "
                f"{current.elapsed_ms}"
            )

    missing_loss = [record.iteration for record in records if record.loss is None]
    if missing_loss:
        warnings.append("records without main loss: " + _compact_ranges(missing_loss))

    return RunData(name or path.stem, path, records, warnings)


def smooth_loss(
    records: Sequence[TrainingRecord],
    ema_halflife: float,
    rolling_window: int,
    centered_window: int = 0,
) -> list[SmoothedPoint]:
    """Compute bias-aware EMA and iteration-window means for finite losses."""

    values = [
        (record.iteration, record.loss)
        for record in records
        if record.loss is not None and math.isfinite(record.loss)
    ]
    if not values:
        return []
    if ema_halflife <= 0:
        raise ValueError("EMA half-life must be positive")
    if rolling_window <= 0:
        raise ValueError("rolling window must be positive")
    if centered_window < 0:
        raise ValueError("centered window cannot be negative")

    decay_per_iteration = math.exp(math.log(0.5) / ema_halflife)
    ema_numerator = 0.0
    ema_weight = 0.0
    previous_iteration: int | None = None
    trailing: deque[tuple[int, float]] = deque()
    trailing_sum = 0.0
    intermediate: list[tuple[int, float, float, float]] = []

    for iteration, value_or_none in values:
        value = float(value_or_none)
        delta = 1 if previous_iteration is None else iteration - previous_iteration
        decay = decay_per_iteration ** max(1, delta)
        ema_numerator = decay * ema_numerator + (1.0 - decay) * value
        ema_weight = decay * ema_weight + (1.0 - decay)
        ema = ema_numerator / ema_weight
        previous_iteration = iteration

        trailing.append((iteration, value))
        trailing_sum += value
        minimum_iteration = iteration - rolling_window + 1
        while trailing and trailing[0][0] < minimum_iteration:
            _, expired = trailing.popleft()
            trailing_sum -= expired
        rolling = trailing_sum / len(trailing)
        intermediate.append((iteration, value, ema, rolling))

    centered: dict[int, float] = {}
    if centered_window:
        left_width = (centered_window - 1) // 2
        right_width = centered_window // 2
        for iteration, _, _, _ in intermediate:
            candidates = [
                value
                for other_iteration, value in values
                if iteration - left_width <= other_iteration <= iteration + right_width
            ]
            centered[iteration] = statistics.fmean(candidates)

    return [
        SmoothedPoint(
            iteration=iteration,
            raw=raw,
            ema=ema,
            rolling=rolling,
            centered=centered.get(iteration),
        )
        for iteration, raw, ema, rolling in intermediate
    ]


def _points_by_iteration(points: Sequence[SmoothedPoint]) -> dict[int, SmoothedPoint]:
    return {point.iteration: point for point in points}


def _curve_value(point: SmoothedPoint, curve: str) -> float:
    value = getattr(point, curve)
    if value is None:
        raise ValueError(f"curve {curve!r} is not available")
    return float(value)


def find_crossings(
    reference: Sequence[SmoothedPoint],
    candidate: Sequence[SmoothedPoint],
    curve: str = "ema",
) -> list[tuple[float, str]]:
    """Return linear-interpolated common-iteration loss-difference crossings."""

    reference_by_iteration = _points_by_iteration(reference)
    candidate_by_iteration = _points_by_iteration(candidate)
    common = sorted(reference_by_iteration.keys() & candidate_by_iteration.keys())
    differences = [
        (
            iteration,
            _curve_value(candidate_by_iteration[iteration], curve)
            - _curve_value(reference_by_iteration[iteration], curve),
        )
        for iteration in common
    ]
    crossings: list[tuple[float, str]] = []
    for (x0, y0), (x1, y1) in zip(differences, differences[1:]):
        if y0 == 0.0:
            crossing = float(x0)
        elif y0 * y1 < 0.0:
            crossing = x0 - y0 * (x1 - x0) / (y1 - y0)
        else:
            continue
        direction = "candidate becomes lower" if y1 < 0 else "candidate becomes higher"
        if not crossings or abs(crossing - crossings[-1][0]) > 1e-12:
            crossings.append((crossing, direction))
    return crossings


def first_reach(
    points: Sequence[SmoothedPoint],
    threshold: float,
    curve: str = "ema",
    confirm_points: int = 1,
) -> float | None:
    """Return first confirmed descending threshold crossing.

    The crossing itself is linearly interpolated.  It is accepted only when
    the crossing point and the next ``confirm_points - 1`` observed curve
    points are all at or below the threshold.  This suppresses one-point dips
    through a threshold without imposing monotonicity on the whole curve.
    """

    if not points:
        return None
    if confirm_points <= 0:
        raise ValueError("confirm points must be positive")

    def confirmed(start_index: int) -> bool:
        stop = start_index + confirm_points
        return stop <= len(points) and all(
            _curve_value(point, curve) <= threshold
            for point in points[start_index:stop]
        )

    first_value = _curve_value(points[0], curve)
    if first_value <= threshold and confirmed(0):
        return float(points[0].iteration)
    for index, (previous, current) in enumerate(zip(points, points[1:]), start=1):
        y0 = _curve_value(previous, curve)
        y1 = _curve_value(current, curve)
        if y0 > threshold >= y1 and confirmed(index):
            if y1 == y0:
                return float(current.iteration)
            return previous.iteration + (threshold - y0) * (
                current.iteration - previous.iteration
            ) / (y1 - y0)
    return None


def automatic_thresholds(
    smoothed: Mapping[str, Sequence[SmoothedPoint]], curve: str, count: int
) -> list[float]:
    """Choose interior loss thresholds in the range covered by every run."""

    if count <= 0 or not smoothed:
        return []
    ranges: list[tuple[float, float]] = []
    for points in smoothed.values():
        values = [_curve_value(point, curve) for point in points]
        if not values:
            return []
        ranges.append((min(values), max(values)))
    shared_low = max(low for low, _ in ranges)
    shared_high = min(high for _, high in ranges)
    if not math.isfinite(shared_low + shared_high) or shared_high <= shared_low:
        return []
    return [
        shared_high - (shared_high - shared_low) * index / (count + 1)
        for index in range(1, count + 1)
    ]


def _interpolate_curve(points: Sequence[SmoothedPoint], step: float, curve: str) -> float | None:
    if not points or step < points[0].iteration or step > points[-1].iteration:
        return None
    for point in points:
        if point.iteration == step:
            return _curve_value(point, curve)
    for left, right in zip(points, points[1:]):
        if left.iteration < step < right.iteration:
            left_value = _curve_value(left, curve)
            right_value = _curve_value(right, curve)
            fraction = (step - left.iteration) / (right.iteration - left.iteration)
            return left_value + fraction * (right_value - left_value)
    return None


def interpolate_consumed_samples(
    records: Sequence[TrainingRecord], step: float | None
) -> float | None:
    """Linearly interpolate consumed samples at a fractional reached step."""

    if step is None:
        return None
    available = [
        (record.iteration, float(record.consumed_samples))
        for record in records
        if record.consumed_samples is not None
    ]
    if not available or step < available[0][0] or step > available[-1][0]:
        return None
    for iteration, consumed in available:
        if iteration == step:
            return consumed
    for (left_step, left_value), (right_step, right_value) in zip(
        available, available[1:]
    ):
        if left_step < step < right_step:
            fraction = (step - left_step) / (right_step - left_step)
            return left_value + fraction * (right_value - left_value)
    return None


def _local_slope(
    points: Sequence[SmoothedPoint], step: float, curve: str, window: int
) -> tuple[float | None, int]:
    """Fit y = intercept + slope*x around step using ordinary least squares."""

    if window <= 1:
        raise ValueError("slope window must exceed one iteration")
    left_width = (window - 1) / 2.0
    right_width = window / 2.0
    nearby = [
        (point.iteration, _curve_value(point, curve))
        for point in points
        if step - left_width <= point.iteration <= step + right_width
    ]
    if len(nearby) < 2:
        return None, len(nearby)
    x_mean = statistics.fmean(x for x, _ in nearby)
    y_mean = statistics.fmean(y for _, y in nearby)
    denominator = sum((x - x_mean) ** 2 for x, _ in nearby)
    if denominator == 0:
        return None, len(nearby)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in nearby)
    return numerator / denominator, len(nearby)


def horizontal_lead(
    reference: Sequence[SmoothedPoint],
    candidate: Sequence[SmoothedPoint],
    step: float,
    curve: str,
    slope_window: int,
) -> tuple[float, float, float, int] | None:
    """Convert vertical candidate-reference gap to reference-curve step lead.

    The linear approximation is ``lead = vertical_gap / reference_slope``.
    A positive result means the candidate is ahead when reference loss is
    decreasing and candidate loss is lower.
    """

    reference_value = _interpolate_curve(reference, step, curve)
    candidate_value = _interpolate_curve(candidate, step, curve)
    if reference_value is None or candidate_value is None:
        return None
    slope, count = _local_slope(reference, step, curve, slope_window)
    if slope is None or abs(slope) < 1e-15:
        return None
    gap = candidate_value - reference_value
    return gap / slope, gap, slope, count


def cross_run_warnings(runs: Sequence[RunData], tolerance: float = 1e-9) -> list[str]:
    if len(runs) < 2:
        return []
    warnings: list[str] = []
    reference = runs[0]
    reference_by_iteration = {record.iteration: record for record in reference.records}
    for candidate in runs[1:]:
        candidate_by_iteration = {record.iteration: record for record in candidate.records}
        common = sorted(reference_by_iteration.keys() & candidate_by_iteration.keys())
        lr_mismatch: list[int] = []
        batch_mismatch: list[int] = []
        consumed_mismatch: list[int] = []
        for iteration in common:
            left = reference_by_iteration[iteration]
            right = candidate_by_iteration[iteration]
            if left.learning_rate is not None and right.learning_rate is not None:
                scale = max(abs(left.learning_rate), abs(right.learning_rate), 1e-30)
                if abs(left.learning_rate - right.learning_rate) > tolerance * scale:
                    lr_mismatch.append(iteration)
            if (
                left.global_batch_size is not None
                and right.global_batch_size is not None
                and left.global_batch_size != right.global_batch_size
            ):
                batch_mismatch.append(iteration)
            if (
                left.consumed_samples is not None
                and right.consumed_samples is not None
                and left.consumed_samples != right.consumed_samples
            ):
                consumed_mismatch.append(iteration)
        prefix = f"{candidate.name} vs {reference.name}"
        if lr_mismatch:
            warnings.append(prefix + " learning-rate mismatch: " + _compact_ranges(lr_mismatch))
        if batch_mismatch:
            warnings.append(prefix + " global-batch mismatch: " + _compact_ranges(batch_mismatch))
        if consumed_mismatch:
            warnings.append(prefix + " consumed-sample mismatch: " + _compact_ranges(consumed_mismatch))
    return warnings


def write_csv(
    output_path: Path,
    runs: Sequence[RunData],
    smoothed: Mapping[str, Sequence[SmoothedPoint]],
) -> None:
    mtp_indices = sorted(
        {
            index
            for run in runs
            for record in run.records
            for index in record.mtp_losses
        }
    )
    fieldnames = [
        "run",
        "source",
        "iteration",
        "total_iterations",
        "consumed_samples",
        "elapsed_ms",
        "learning_rate",
        "global_batch_size",
        "loss",
        *(f"mtp_{index}_loss" for index in mtp_indices),
        "aux_loss",
        "grad_norm",
        "ema_loss",
        "rolling_loss",
        "centered_loss",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            smooth_by_iteration = _points_by_iteration(smoothed[run.name])
            for record in run.records:
                point = smooth_by_iteration.get(record.iteration)
                row: dict[str, object] = {
                    "run": run.name,
                    "source": str(run.path),
                    "iteration": record.iteration,
                    "total_iterations": record.total_iterations,
                    "consumed_samples": record.consumed_samples,
                    "elapsed_ms": record.elapsed_ms,
                    "learning_rate": record.learning_rate,
                    "global_batch_size": record.global_batch_size,
                    "loss": record.loss,
                    "aux_loss": record.aux_loss,
                    "grad_norm": record.grad_norm,
                    "ema_loss": point.ema if point else None,
                    "rolling_loss": point.rolling if point else None,
                    "centered_loss": point.centered if point else None,
                }
                for index in mtp_indices:
                    row[f"mtp_{index}_loss"] = record.mtp_losses.get(index)
                writer.writerow(row)


def _format_number(value: float | None, precision: int = 6) -> str:
    return "n/a" if value is None else f"{value:.{precision}g}"


def print_report(
    runs: Sequence[RunData],
    smoothed: Mapping[str, Sequence[SmoothedPoint]],
    thresholds: Sequence[float],
    curve: str,
    at_steps: Sequence[float],
    slope_window: int,
    confirm_points: int,
) -> None:
    reference = runs[0]
    print("Convention: pairwise gap = candidate - reference; negative means candidate is lower.")
    print(f"Comparison curve: {curve}")
    print()
    print("Runs")
    for run in runs:
        points = smoothed[run.name]
        if points:
            span = f"{points[0].iteration}-{points[-1].iteration}"
            final = _curve_value(points[-1], curve)
            print(
                f"  {run.name}: {len(run.records)} records, iterations {span}, "
                f"final {curve}={final:.6g}, source={run.path}"
            )
        else:
            print(f"  {run.name}: no finite loss records, source={run.path}")
        for warning in run.warnings:
            print(f"    WARNING: {warning}")

    for warning in cross_run_warnings(runs):
        print(f"  WARNING: {warning}")

    if thresholds:
        print()
        print(
            f"First descending threshold reach on {curve} (linear interpolation; "
            f"{confirm_points} consecutive points required)"
        )
        header = "  threshold" + "".join(f" | {run.name:>14}" for run in runs)
        print(header)
        for threshold in thresholds:
            reaches = [
                first_reach(
                    smoothed[run.name], threshold, curve, confirm_points=confirm_points
                )
                for run in runs
            ]
            row = f"  {threshold:9.6g}" + "".join(
                f" | {_format_number(reach, 8):>14}" for reach in reaches
            )
            print(row)
            consumed = [
                interpolate_consumed_samples(run.records, reach)
                for run, reach in zip(runs, reaches)
            ]
            consumed_row = "  samples  " + "".join(
                f" | {_format_number(value, 10):>14}" for value in consumed
            )
            print(consumed_row + "  (linearly interpolated)")
            if reaches[0] is not None:
                leads = [
                    None if reach is None else reaches[0] - reach for reach in reaches
                ]
                lead_row = "  step lead" + "".join(
                    f" | {_format_number(lead, 8):>14}" for lead in leads
                )
                print(lead_row + "  (positive = earlier than reference)")

    if len(runs) < 2:
        return

    print()
    print(f"Pairwise comparisons against reference {reference.name}")
    reference_points = smoothed[reference.name]
    reference_iterations = {point.iteration for point in reference_points}
    for candidate in runs[1:]:
        candidate_points = smoothed[candidate.name]
        common = sorted(
            reference_iterations
            & {point.iteration for point in candidate_points}
        )
        print(f"  {candidate.name} - {reference.name}")
        if not common:
            print("    no common finite-loss iterations")
            continue
        reference_by_iteration = _points_by_iteration(reference_points)
        candidate_by_iteration = _points_by_iteration(candidate_points)
        gaps = [
            _curve_value(candidate_by_iteration[iteration], curve)
            - _curve_value(reference_by_iteration[iteration], curve)
            for iteration in common
        ]
        print(
            f"    common exact iterations: {common[0]}-{common[-1]} "
            f"({len(common)} points); mean gap={statistics.fmean(gaps):.6g}; "
            f"final gap={gaps[-1]:.6g}"
        )
        crossings = find_crossings(reference_points, candidate_points, curve)
        if crossings:
            shown = crossings[:12]
            rendered = "; ".join(
                f"{step:.3f} ({direction})" for step, direction in shown
            )
            if len(crossings) > len(shown):
                rendered += f"; ... ({len(crossings) - len(shown)} more)"
            print(f"    crossings: {rendered}")
        else:
            print("    crossings: none")

        requested_steps = list(at_steps) if at_steps else [float(common[-1])]
        for step in requested_steps:
            result = horizontal_lead(
                reference_points,
                candidate_points,
                step,
                curve,
                slope_window,
            )
            if result is None:
                print(f"    at step {step:g}: horizontal lead unavailable")
                continue
            lead, gap, slope, count = result
            qualifier = "ahead" if lead >= 0 else "behind"
            print(
                f"    at step {step:g}: gap={gap:.6g}, reference local slope="
                f"{slope:.6g}/step over {count} points, candidate {qualifier} "
                f"by approximately {abs(lead):.3f} steps"
            )


COLORS = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#ea580c",
    "#0891b2",
    "#be185d",
    "#4d7c0f",
    "#4338ca",
    "#a16207",
)


def _expanded_bounds(values: Sequence[float], padding_fraction: float = 0.05) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if high == low:
        padding = max(abs(low) * padding_fraction, 1.0)
    else:
        padding = (high - low) * padding_fraction
    return low - padding, high + padding


def _svg_path(
    values: Iterable[tuple[float, float]],
    x_scale,
    y_scale,
) -> str:
    commands: list[str] = []
    for index, (x_value, y_value) in enumerate(values):
        command = "M" if index == 0 else "L"
        commands.append(f"{command}{x_scale(x_value):.2f},{y_scale(y_value):.2f}")
    return " ".join(commands)


def _axis_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if count <= 1 or high == low:
        return [low]
    return [low + (high - low) * index / (count - 1) for index in range(count)]


def _loss_chart_svg(
    runs: Sequence[RunData],
    smoothed: Mapping[str, Sequence[SmoothedPoint]],
    curve: str,
    show_raw: bool,
) -> str:
    width, height = 1200.0, 620.0
    left, right, top, bottom = 82.0, 24.0, 32.0, 66.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_points = [point for run in runs for point in smoothed[run.name]]
    if not all_points:
        return '<p class="empty">No finite loss data.</p>'
    x_low, x_high = _expanded_bounds([float(point.iteration) for point in all_points], 0.01)
    y_values = [_curve_value(point, curve) for point in all_points]
    if show_raw:
        y_values.extend(point.raw for point in all_points)
    y_low, y_high = _expanded_bounds(y_values)

    def x_scale(value: float) -> float:
        return left + (value - x_low) / (x_high - x_low) * plot_width

    def y_scale(value: float) -> float:
        return top + (y_high - value) / (y_high - y_low) * plot_height

    output = [
        f'<svg id="loss-chart" class="interactive-chart" '
        f'viewBox="0 0 {width:g} {height:g}" role="img" '
        f'data-x-min="{x_low:.17g}" data-x-max="{x_high:.17g}" '
        f'data-y-min="{y_low:.17g}" data-y-max="{y_high:.17g}" '
        f'data-plot-left="{left:g}" data-plot-top="{top:g}" '
        f'data-plot-width="{plot_width:g}" data-plot-height="{plot_height:g}" '
        'aria-label="training loss comparison">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<defs><clipPath id="loss-chart-clip"><rect x="{left:g}" y="{top:g}" '
        f'width="{plot_width:g}" height="{plot_height:g}"/></clipPath></defs>',
        '<g class="y-grid-ticks">',
    ]
    for tick in _axis_ticks(y_low, y_high, 6):
        y = y_scale(tick)
        output.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        output.append(
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" '
            f'class="tick">{tick:.4g}</text>'
        )
    output.append('</g><g class="x-grid-ticks">')
    for tick in _axis_ticks(x_low, x_high, 7):
        x = x_scale(tick)
        output.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" '
            'stroke="#f3f4f6" stroke-width="1"/>'
        )
        output.append(
            f'<text x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle" '
            f'class="tick">{tick:.0f}</text>'
        )
    output.append("</g>")
    output.extend(
        [
            f'<text x="{left+plot_width/2:.2f}" y="{height-14}" text-anchor="middle" '
            'class="axis-label">optimizer iteration</text>',
            f'<text transform="translate(20 {top+plot_height/2:.2f}) rotate(-90)" '
            f'text-anchor="middle" class="axis-label">loss ({html.escape(curve)})</text>',
            '<g class="plot-clip" clip-path="url(#loss-chart-clip)"><g class="plot-data">',
        ]
    )

    for index, run in enumerate(runs):
        color = COLORS[index % len(COLORS)]
        points = smoothed[run.name]
        if show_raw:
            raw_path = _svg_path(
                ((point.iteration, point.raw) for point in points), x_scale, y_scale
            )
            output.append(
                f'<path d="{raw_path}" fill="none" stroke="{color}" '
                'stroke-opacity="0.16" stroke-width="1" '
                'vector-effect="non-scaling-stroke"/>'
            )
        smooth_path = _svg_path(
            (
                (point.iteration, _curve_value(point, curve))
                for point in points
            ),
            x_scale,
            y_scale,
        )
        output.append(
            f'<path d="{smooth_path}" fill="none" stroke="{color}" '
            'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" '
            'vector-effect="non-scaling-stroke"/>'
        )
    output.extend(
        [
            "</g></g>",
            f'<rect class="axis-frame" x="{left:g}" y="{top:g}" '
            f'width="{plot_width:g}" height="{plot_height:g}" fill="none" '
            'stroke="#374151" pointer-events="none"/>',
            f'<rect class="zoom-selection" x="{left:g}" y="{top:g}" width="0" '
            'height="0" visibility="hidden"/>',
            f'<rect class="plot-interaction" x="{left:g}" y="{top:g}" '
            f'width="{plot_width:g}" height="{plot_height:g}" fill="transparent"/>',
            "</svg>",
        ]
    )
    return "".join(output)


def _threshold_chart_svg(
    runs: Sequence[RunData],
    smoothed: Mapping[str, Sequence[SmoothedPoint]],
    thresholds: Sequence[float],
    curve: str,
    confirm_points: int,
) -> str:
    reaches = {
        run.name: [
            first_reach(
                smoothed[run.name], threshold, curve, confirm_points=confirm_points
            )
            for threshold in thresholds
        ]
        for run in runs
    }
    finite_reaches = [
        value
        for run_values in reaches.values()
        for value in run_values
        if value is not None
    ]
    if not thresholds or not finite_reaches:
        return '<p class="empty">No confirmed threshold reaches to chart.</p>'

    width, height = 1200.0, 480.0
    left, right, top, bottom = 82.0, 24.0, 32.0, 76.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_low, x_high = _expanded_bounds(finite_reaches)
    y_low, y_high = _expanded_bounds(list(thresholds))

    def x_scale(value: float) -> float:
        return left + (value - x_low) / (x_high - x_low) * plot_width

    def y_scale(value: float) -> float:
        return top + (y_high - value) / (y_high - y_low) * plot_height

    output = [
        f'<svg id="threshold-chart" class="interactive-chart" '
        f'viewBox="0 0 {width:g} {height:g}" role="img" '
        f'data-x-min="{x_low:.17g}" data-x-max="{x_high:.17g}" '
        f'data-y-min="{y_low:.17g}" data-y-max="{y_high:.17g}" '
        f'data-plot-left="{left:g}" data-plot-top="{top:g}" '
        f'data-plot-width="{plot_width:g}" data-plot-height="{plot_height:g}" '
        'aria-label="iteration at equal smoothed loss">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<defs><clipPath id="threshold-chart-clip"><rect x="{left:g}" y="{top:g}" '
        f'width="{plot_width:g}" height="{plot_height:g}"/></clipPath></defs>',
        '<g class="y-grid-ticks">',
    ]
    for tick in _axis_ticks(y_low, y_high, 6):
        y = y_scale(tick)
        output.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" '
            'stroke="#e5e7eb"/>'
        )
        output.append(
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" '
            f'class="tick">{tick:.4g}</text>'
        )
    output.append('</g><g class="x-grid-ticks">')
    for tick in _axis_ticks(x_low, x_high, 7):
        x = x_scale(tick)
        output.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" '
            'stroke="#f3f4f6"/>'
        )
        output.append(
            f'<text x="{x:.2f}" y="{height-bottom+25}" text-anchor="middle" '
            f'class="tick">{tick:.0f}</text>'
        )
    output.append("</g>")
    output.extend(
        [
            f'<text x="{left+plot_width/2:.2f}" y="{height-15}" text-anchor="middle" '
            'class="axis-label">first confirmed iteration (lower is better)</text>',
            f'<text transform="translate(20 {top+plot_height/2:.2f}) rotate(-90)" '
            'text-anchor="middle" class="axis-label">smoothed loss threshold</text>',
            '<g class="plot-clip" clip-path="url(#threshold-chart-clip)"><g class="plot-data">',
        ]
    )
    for run_index, run in enumerate(runs):
        color = COLORS[run_index % len(COLORS)]
        coordinates = [
            (x_scale(reach), y_scale(threshold))
            for threshold, reach in zip(thresholds, reaches[run.name])
            if reach is not None
        ]
        if len(coordinates) >= 2:
            path = " ".join(
                ("M" if index == 0 else "L") + f"{x:.2f},{y:.2f}"
                for index, (x, y) in enumerate(coordinates)
            )
            output.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5" '
                'vector-effect="non-scaling-stroke"/>'
            )
        for x, y in coordinates:
            output.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
    output.extend(
        [
            "</g></g>",
            f'<rect class="axis-frame" x="{left:g}" y="{top:g}" '
            f'width="{plot_width:g}" height="{plot_height:g}" fill="none" '
            'stroke="#374151" pointer-events="none"/>',
            f'<rect class="zoom-selection" x="{left:g}" y="{top:g}" width="0" '
            'height="0" visibility="hidden"/>',
            f'<rect class="plot-interaction" x="{left:g}" y="{top:g}" '
            f'width="{plot_width:g}" height="{plot_height:g}" fill="transparent"/>',
            "</svg>",
        ]
    )
    return "".join(output)


def write_html(
    output_path: Path,
    runs: Sequence[RunData],
    smoothed: Mapping[str, Sequence[SmoothedPoint]],
    thresholds: Sequence[float],
    curve: str,
    confirm_points: int,
    show_raw: bool,
    ema_halflife: float,
    rolling_window: int,
    centered_window: int,
) -> None:
    reference = runs[0]
    loss_chart = _loss_chart_svg(runs, smoothed, curve, show_raw)
    threshold_chart = _threshold_chart_svg(
        runs, smoothed, thresholds, curve, confirm_points
    )
    legend = "".join(
        '<span class="legend-item">'
        f'<span class="swatch" style="background:{COLORS[index % len(COLORS)]}"></span>'
        f'{html.escape(run.name)}</span>'
        for index, run in enumerate(runs)
    )

    threshold_rows: list[str] = []
    for threshold in thresholds:
        reference_reach = first_reach(
            smoothed[reference.name], threshold, curve, confirm_points=confirm_points
        )
        cells = [f"<td>{threshold:.6g}</td>"]
        for run in runs:
            reach = first_reach(
                smoothed[run.name], threshold, curve, confirm_points=confirm_points
            )
            if reach is None:
                cells.append("<td>n/a</td><td>n/a</td><td>n/a</td>")
            else:
                lead = None if reference_reach is None else reference_reach - reach
                consumed = interpolate_consumed_samples(run.records, reach)
                lead_text = "n/a" if lead is None else f"{lead:+.2f}"
                consumed_text = "n/a" if consumed is None else f"{consumed:.0f}"
                lead_class = "" if lead is None else (" good" if lead > 0 else " bad" if lead < 0 else "")
                cells.append(
                    f"<td>{reach:.2f}</td><td>{consumed_text}</td>"
                    f"<td class=\"{lead_class.strip()}\">{lead_text}</td>"
                )
        threshold_rows.append("<tr>" + "".join(cells) + "</tr>")
    threshold_header = "".join(
        f'<th colspan="3">{html.escape(run.name)}</th>' for run in runs
    )
    threshold_subheader = "".join(
        "<th>iteration</th><th>consumed samples</th><th>step lead</th>"
        for _ in runs
    )

    warning_items = []
    for run in runs:
        warning_items.extend(
            f"<li><strong>{html.escape(run.name)}:</strong> {html.escape(warning)}</li>"
            for warning in run.warnings
        )
    warning_items.extend(
        f"<li>{html.escape(warning)}</li>" for warning in cross_run_warnings(runs)
    )
    warnings_html = (
        "<ul>" + "".join(warning_items) + "</ul>"
        if warning_items
        else '<p class="ok">No structural consistency warning detected.</p>'
    )

    interaction_script = r"""<script>
const SVG_NS = "http://www.w3.org/2000/svg";

document.querySelectorAll("svg.interactive-chart").forEach((svg) => {
  const initial = {
    xMin: Number(svg.dataset.xMin), xMax: Number(svg.dataset.xMax),
    yMin: Number(svg.dataset.yMin), yMax: Number(svg.dataset.yMax)
  };
  const plot = {
    left: Number(svg.dataset.plotLeft), top: Number(svg.dataset.plotTop),
    width: Number(svg.dataset.plotWidth), height: Number(svg.dataset.plotHeight)
  };
  let domain = {...initial};
  let drag = null;
  const dataLayer = svg.querySelector(".plot-data");
  const xTickLayer = svg.querySelector(".x-grid-ticks");
  const yTickLayer = svg.querySelector(".y-grid-ticks");
  const interaction = svg.querySelector(".plot-interaction");
  const selection = svg.querySelector(".zoom-selection");
  const controls = document.querySelector(`[data-chart-controls="${svg.id}"]`);

  const svgPoint = (event) => {
    const rect = svg.getBoundingClientRect();
    const viewBox = svg.viewBox.baseVal;
    return {
      x: viewBox.x + (event.clientX - rect.left) / rect.width * viewBox.width,
      y: viewBox.y + (event.clientY - rect.top) / rect.height * viewBox.height
    };
  };
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const makeSvg = (tag, attributes) => {
    const element = document.createElementNS(SVG_NS, tag);
    Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, value));
    return element;
  };
  const tickText = (value, span, count) => {
    const step = Math.abs(span) / Math.max(1, count - 1);
    if ((Math.abs(value) >= 1e6) || (Math.abs(value) > 0 && Math.abs(value) < 1e-4)) {
      return value.toExponential(3);
    }
    const decimals = step > 0 ? Math.max(0, Math.min(6, Math.ceil(-Math.log10(step)))) : 0;
    let text = value.toFixed(decimals);
    if (decimals > 0) text = text.replace(/\.?0+$/, "");
    return text;
  };
  const syncControls = () => {
    if (!controls) return;
    ["xMin", "xMax", "yMin", "yMax"].forEach((key) => {
      const input = controls.querySelector(`[data-domain-bound="${key}"]`);
      if (input) input.value = Number(domain[key].toPrecision(10));
    });
  };
  const renderTicks = () => {
    xTickLayer.replaceChildren();
    yTickLayer.replaceChildren();
    const xCount = 7;
    const yCount = 6;
    for (let index = 0; index < xCount; index += 1) {
      const fraction = index / (xCount - 1);
      const value = domain.xMin + (domain.xMax - domain.xMin) * fraction;
      const x = plot.left + plot.width * fraction;
      xTickLayer.appendChild(makeSvg("line", {
        x1: x, y1: plot.top, x2: x, y2: plot.top + plot.height,
        stroke: "#f3f4f6", "stroke-width": 1
      }));
      const text = makeSvg("text", {
        x, y: plot.top + plot.height + 24, "text-anchor": "middle", class: "tick"
      });
      text.textContent = tickText(value, domain.xMax - domain.xMin, xCount);
      xTickLayer.appendChild(text);
    }
    for (let index = 0; index < yCount; index += 1) {
      const fraction = index / (yCount - 1);
      const value = domain.yMin + (domain.yMax - domain.yMin) * fraction;
      const y = plot.top + plot.height * (1 - fraction);
      yTickLayer.appendChild(makeSvg("line", {
        x1: plot.left, y1: y, x2: plot.left + plot.width, y2: y,
        stroke: "#e5e7eb", "stroke-width": 1
      }));
      const text = makeSvg("text", {
        x: plot.left - 10, y: y + 4, "text-anchor": "end", class: "tick"
      });
      text.textContent = tickText(value, domain.yMax - domain.yMin, yCount);
      yTickLayer.appendChild(text);
    }
  };
  const renderDomain = () => {
    const initialXSpan = initial.xMax - initial.xMin;
    const initialYSpan = initial.yMax - initial.yMin;
    const xSpan = domain.xMax - domain.xMin;
    const ySpan = domain.yMax - domain.yMin;
    const scaleX = initialXSpan / xSpan;
    const scaleY = initialYSpan / ySpan;
    const translateX = plot.left + (initial.xMin - domain.xMin) * plot.width / xSpan - scaleX * plot.left;
    const translateY = plot.top + (domain.yMax - initial.yMax) * plot.height / ySpan - scaleY * plot.top;
    dataLayer.setAttribute(
      "transform",
      `matrix(${scaleX} 0 0 ${scaleY} ${translateX} ${translateY})`
    );
    dataLayer.querySelectorAll("circle").forEach((circle) => {
      const centerX = Number(circle.getAttribute("cx"));
      const centerY = Number(circle.getAttribute("cy"));
      circle.setAttribute(
        "transform",
        `translate(${centerX} ${centerY}) scale(${1 / scaleX} ${1 / scaleY}) translate(${-centerX} ${-centerY})`
      );
    });
    renderTicks();
    syncControls();
  };
  const resetDomain = () => {
    domain = {...initial};
    renderDomain();
  };

  interaction.addEventListener("wheel", (event) => {
    event.preventDefault();
    const point = svgPoint(event);
    const fractionX = clamp((point.x - plot.left) / plot.width, 0, 1);
    const fractionY = clamp((point.y - plot.top) / plot.height, 0, 1);
    const oldXSpan = domain.xMax - domain.xMin;
    const oldYSpan = domain.yMax - domain.yMin;
    const factor = event.deltaY < 0 ? 0.82 : 1.22;
    const newXSpan = clamp(oldXSpan * factor, (initial.xMax - initial.xMin) / 500, (initial.xMax - initial.xMin) * 20);
    const newYSpan = clamp(oldYSpan * factor, (initial.yMax - initial.yMin) / 500, (initial.yMax - initial.yMin) * 20);
    const anchorX = domain.xMin + fractionX * oldXSpan;
    const anchorY = domain.yMax - fractionY * oldYSpan;
    domain.xMin = anchorX - fractionX * newXSpan;
    domain.xMax = domain.xMin + newXSpan;
    domain.yMax = anchorY + fractionY * newYSpan;
    domain.yMin = domain.yMax - newYSpan;
    renderDomain();
  }, {passive: false});

  interaction.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    interaction.setPointerCapture(event.pointerId);
    const point = svgPoint(event);
    drag = {mode: event.shiftKey ? "zoom" : "pan", start: point, domain: {...domain}};
    interaction.classList.add("dragging");
    if (drag.mode === "zoom") {
      selection.setAttribute("x", point.x);
      selection.setAttribute("y", point.y);
      selection.setAttribute("width", 0);
      selection.setAttribute("height", 0);
      selection.setAttribute("visibility", "visible");
    }
  });
  interaction.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const point = svgPoint(event);
    if (drag.mode === "pan") {
      const xSpan = drag.domain.xMax - drag.domain.xMin;
      const ySpan = drag.domain.yMax - drag.domain.yMin;
      const deltaX = (point.x - drag.start.x) / plot.width * xSpan;
      const deltaY = (point.y - drag.start.y) / plot.height * ySpan;
      domain.xMin = drag.domain.xMin - deltaX;
      domain.xMax = drag.domain.xMax - deltaX;
      domain.yMin = drag.domain.yMin + deltaY;
      domain.yMax = drag.domain.yMax + deltaY;
      renderDomain();
      return;
    }
    const x0 = clamp(Math.min(drag.start.x, point.x), plot.left, plot.left + plot.width);
    const x1 = clamp(Math.max(drag.start.x, point.x), plot.left, plot.left + plot.width);
    const y0 = clamp(Math.min(drag.start.y, point.y), plot.top, plot.top + plot.height);
    const y1 = clamp(Math.max(drag.start.y, point.y), plot.top, plot.top + plot.height);
    selection.setAttribute("x", x0);
    selection.setAttribute("y", y0);
    selection.setAttribute("width", x1 - x0);
    selection.setAttribute("height", y1 - y0);
  });
  const stopDrag = (event) => {
    if (!drag) return;
    if (drag.mode === "zoom" && event) {
      const x = Number(selection.getAttribute("x"));
      const y = Number(selection.getAttribute("y"));
      const width = Number(selection.getAttribute("width"));
      const height = Number(selection.getAttribute("height"));
      if (width >= 8 && height >= 8) {
        const xSpan = drag.domain.xMax - drag.domain.xMin;
        const ySpan = drag.domain.yMax - drag.domain.yMin;
        domain.xMin = drag.domain.xMin + (x - plot.left) / plot.width * xSpan;
        domain.xMax = drag.domain.xMin + (x + width - plot.left) / plot.width * xSpan;
        domain.yMax = drag.domain.yMax - (y - plot.top) / plot.height * ySpan;
        domain.yMin = drag.domain.yMax - (y + height - plot.top) / plot.height * ySpan;
        renderDomain();
      }
    }
    selection.setAttribute("visibility", "hidden");
    interaction.classList.remove("dragging");
    drag = null;
  };
  interaction.addEventListener("pointerup", stopDrag);
  interaction.addEventListener("pointercancel", () => stopDrag(null));
  interaction.addEventListener("dblclick", resetDomain);

  if (controls) {
    const error = controls.querySelector(".range-error");
    const applyInputs = () => {
      const next = {};
      ["xMin", "xMax", "yMin", "yMax"].forEach((key) => {
        const text = controls.querySelector(`[data-domain-bound="${key}"]`).value.trim();
        next[key] = text === "" ? Number.NaN : Number(text);
      });
      if (!Object.values(next).every(Number.isFinite) || next.xMin >= next.xMax || next.yMin >= next.yMax) {
        error.textContent = "Require finite min < max for both axes.";
        return;
      }
      error.textContent = "";
      domain = next;
      renderDomain();
    };
    controls.querySelector("[data-apply-chart]").addEventListener("click", applyInputs);
    controls.querySelector("[data-reset-chart]").addEventListener("click", resetDomain);
    controls.querySelectorAll("input").forEach((input) => {
      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") applyInputs();
      });
    });
  }
  renderDomain();
});
</script>"""

    def chart_toolbar(chart_id: str) -> str:
        return (
            f'<div class="chart-toolbar" data-chart-controls="{chart_id}">'
            '<span class="interaction-hint">Wheel: zoom · drag: pan · '
            'Shift+drag: box zoom · double-click: reset</span>'
            '<div class="axis-range-controls">'
            '<label>X min <input type="number" step="any" data-domain-bound="xMin"></label>'
            '<label>X max <input type="number" step="any" data-domain-bound="xMax"></label>'
            '<label>Y min <input type="number" step="any" data-domain-bound="yMin"></label>'
            '<label>Y max <input type="number" step="any" data-domain-bound="yMax"></label>'
            '<button type="button" data-apply-chart>Apply range</button>'
            '<button type="button" data-reset-chart>Reset view</button>'
            '<span class="range-error" role="status"></span>'
            '</div></div>'
        )

    loss_toolbar = chart_toolbar("loss-chart")
    threshold_toolbar = chart_toolbar("threshold-chart")

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>F4 training-loss comparison</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 24px; color: #111827; background: #f8fafc; }}
main {{ max-width: 1280px; margin: auto; }}
.card {{ background: white; border: 1px solid #e5e7eb; border-radius: 12px; margin: 18px 0; padding: 18px; box-shadow: 0 1px 2px #0000000d; }}
h1, h2 {{ margin-top: 0; }}
.meta {{ color: #4b5563; line-height: 1.55; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin: 8px 0 12px; }}
.legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
.swatch {{ width: 18px; height: 4px; border-radius: 2px; display: inline-block; }}
svg {{ width: 100%; height: auto; }}
.interactive-chart {{ touch-action: none; user-select: none; }}
.plot-interaction {{ cursor: grab; }}
.plot-interaction.dragging {{ cursor: grabbing; }}
.zoom-selection {{ fill: #2563eb24; stroke: #2563eb; stroke-width: 1.5; stroke-dasharray: 6 4; pointer-events: none; }}
.chart-toolbar {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px 14px; margin: 4px 0 8px; color: #6b7280; font-size: 13px; }}
.axis-range-controls {{ display: flex; flex-wrap: wrap; align-items: center; gap: 7px; }}
.axis-range-controls label {{ display: inline-flex; align-items: center; gap: 4px; }}
.axis-range-controls input {{ width: 86px; border: 1px solid #d1d5db; border-radius: 5px; padding: 4px 5px; font-variant-numeric: tabular-nums; }}
.chart-toolbar button {{ border: 1px solid #9ca3af; border-radius: 6px; background: #fff; padding: 5px 10px; cursor: pointer; }}
.chart-toolbar button:hover {{ background: #f3f4f6; }}
.range-error {{ color: #b91c1c; min-width: 8px; }}
.tick {{ font-size: 12px; fill: #4b5563; }}
.axis-label {{ font-size: 14px; fill: #1f2937; font-weight: 600; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ border: 1px solid #d1d5db; padding: 7px 9px; text-align: right; }}
th {{ background: #f3f4f6; }}
td:first-child, th:first-child {{ text-align: left; }}
.good {{ color: #047857; font-weight: 600; }}
.bad {{ color: #b91c1c; }}
.ok {{ color: #047857; }}
.empty {{ color: #6b7280; font-style: italic; }}
code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
</style>
</head>
<body><main>
<h1>F4 training-loss comparison</h1>
<p class="meta">Reference run: <strong>{html.escape(reference.name)}</strong>. Pairwise gap is candidate minus reference, so negative loss gap is better. Threshold lead is reference reach minus candidate reach, so positive step lead is better.<br>
Comparison curve: <code>{html.escape(curve)}</code>; EMA half-life: {ema_halflife:g}; trailing window: {rolling_window}; centered window: {centered_window or 'disabled'}; threshold confirmation: {confirm_points} consecutive points; raw traces: {'shown at low opacity' if show_raw else 'hidden'}.</p>
<section class="card"><h2>Loss versus optimizer iteration</h2><div class="legend">{legend}</div>{loss_toolbar}{loss_chart}</section>
<section class="card"><h2>Iteration at equal smoothed loss</h2><div class="legend">{legend}</div>{threshold_toolbar}{threshold_chart}
<table><thead><tr><th rowspan="2">loss threshold</th>{threshold_header}</tr><tr>{threshold_subheader}</tr></thead>
<tbody>{''.join(threshold_rows) if threshold_rows else '<tr><td colspan="99">No shared threshold available.</td></tr>'}</tbody></table>
<p class="meta">A threshold is accepted only after {confirm_points} consecutive {html.escape(curve)} points remain at or below it. Reaching iteration and consumed samples are linearly interpolated across the accepted descending crossing. Step lead is not sample efficiency when global batch sizes differ; see warnings below.</p></section>
<section class="card"><h2>Consistency warnings</h2>{warnings_html}</section>
</main>{interaction_script}</body></html>
"""
    output_path.write_text(document, encoding="utf-8")


def _expand_run_specs(specifications: Sequence[str]) -> tuple[list[str], list[Path]]:
    """Expand named paths, unnamed paths, and quoted globs deterministically."""

    entries: list[tuple[str, Path, bool]] = []
    for specification in specifications:
        specification = specification.strip()
        if not specification:
            raise ValueError("empty -run/--run value")
        if "=" in specification:
            name, path_text = specification.split("=", 1)
            if not name.strip() or not path_text.strip():
                raise ValueError(
                    f"invalid --run {specification!r}; expected NAME=PATH, PATH, or GLOB"
                )
            entries.append((name.strip(), Path(path_text.strip()), True))
            continue

        if glob.has_magic(specification):
            matches = sorted(
                {
                    Path(match)
                    for match in glob.glob(specification, recursive=True)
                    if Path(match).is_file()
                },
                key=lambda path: str(path).casefold(),
            )
            if not matches:
                raise ValueError(f"--run glob matched no files: {specification}")
        else:
            matches = [Path(specification)]
        entries.extend((path.name, path, False) for path in matches)

    explicit_names = [name for name, _, explicit in entries if explicit]
    if len(set(explicit_names)) != len(explicit_names):
        raise ValueError("explicit NAME=PATH run names must be unique")

    used_names = set(explicit_names)
    names: list[str] = []
    paths: list[Path] = []
    for base_name, path, explicit in entries:
        if explicit:
            name = base_name
        else:
            name = base_name
            suffix = 2
            while name in used_names:
                name = f"{base_name}#{suffix}"
                suffix += 1
            used_names.add(name)
        names.append(name)
        paths.append(path)
    return names, paths


def _parse_threshold_specs(specifications: Sequence[str]) -> list[float]:
    """Parse repeated and/or comma-separated finite loss thresholds."""

    thresholds: list[float] = []
    for specification in specifications:
        for token in specification.split(","):
            token = token.strip()
            if not token:
                raise ValueError(
                    f"invalid --threshold list {specification!r}: empty value"
                )
            try:
                threshold = float(token)
            except ValueError as error:
                raise ValueError(f"invalid loss threshold {token!r}") from error
            if not math.isfinite(threshold):
                raise ValueError(f"loss threshold must be finite: {token!r}")
            thresholds.append(threshold)
    return thresholds


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and compare MindSpeed-MM Rank-0 raw training loss logs.",
        epilog=USAGE_EXAMPLES,
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "logs",
        nargs="*",
        type=Path,
        help="log files; alternatively use repeated -run/--run values",
    )
    parser.add_argument(
        "-run",
        "--run",
        action="append",
        default=[],
        metavar="[NAME=]PATH_OR_GLOB",
        help=(
            "named or auto-named run; relative paths are resolved from the current "
            "directory; quote globs; may be repeated"
        ),
    )
    parser.add_argument(
        "--name",
        action="append",
        default=[],
        help="run name, repeated in the same order as log files",
    )
    parser.add_argument(
        "--baseline",
        help="run used as the comparison reference; defaults to the first run",
    )
    parser.add_argument(
        "--ema-halflife",
        type=float,
        default=20.0,
        help="EMA half-life measured in optimizer iterations",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=25,
        help="trailing mean width measured in optimizer iterations",
    )
    parser.add_argument(
        "--centered-window",
        type=int,
        default=0,
        help="offline centered mean width; zero disables it",
    )
    parser.add_argument(
        "--curve",
        choices=("raw", "ema", "rolling", "centered"),
        default="ema",
        help="curve used for comparisons and threshold reaches",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="loss threshold; may be repeated and/or comma-separated",
    )
    parser.add_argument(
        "--auto-thresholds",
        type=int,
        default=5,
        help="choose this many shared interior thresholds when --threshold is omitted",
    )
    parser.add_argument(
        "--confirm-points",
        type=int,
        default=3,
        help="consecutive curve points at/below threshold required to confirm reach",
    )
    parser.add_argument(
        "--at-step",
        type=float,
        action="append",
        default=[],
        help="step for vertical-gap/horizontal-lead estimate; may be repeated",
    )
    parser.add_argument(
        "--slope-window",
        type=int,
        default=101,
        help="centered iteration width for reference local-slope regression",
    )
    parser.add_argument("--csv", type=Path, help="write extracted and smoothed long-form CSV")
    parser.add_argument(
        "--html",
        type=Path,
        help="write a self-contained HTML report with inline SVG charts",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="hide low-opacity raw-loss traces in the HTML chart",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.run and args.logs:
        print("error: use either positional logs or -run/--run, not both", file=sys.stderr)
        return 2
    if args.run and args.name:
        print("error: --name is only valid with positional logs", file=sys.stderr)
        return 2
    if not args.run and not args.logs:
        print("error: provide at least one log or -run/--run value", file=sys.stderr)
        return 2
    if args.name and len(args.name) != len(args.logs):
        print("error: repeat --name exactly once per log, or omit all names", file=sys.stderr)
        return 2
    if args.curve == "centered" and args.centered_window <= 0:
        print("error: --curve centered requires --centered-window > 0", file=sys.stderr)
        return 2
    if args.slope_window <= 1:
        print("error: --slope-window must exceed one iteration", file=sys.stderr)
        return 2
    if args.confirm_points <= 0:
        print("error: --confirm-points must be positive", file=sys.stderr)
        return 2
    if args.auto_thresholds < 0:
        print("error: --auto-thresholds cannot be negative", file=sys.stderr)
        return 2

    try:
        specified_thresholds = _parse_threshold_specs(args.threshold)
        if args.run:
            names, paths = _expand_run_specs(args.run)
        else:
            paths = list(args.logs)
            names = args.name or [path.stem for path in paths]
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if len(set(names)) != len(names):
        print("error: positional run names must be unique", file=sys.stderr)
        return 2
    missing_paths = [path for path in paths if not path.is_file()]
    if missing_paths:
        print(
            "error: missing log file(s): " + ", ".join(str(path) for path in missing_paths),
            file=sys.stderr,
        )
        return 2

    try:
        runs = [load_run(path, name) for path, name in zip(paths, names)]
        if args.baseline:
            matching = [run for run in runs if run.name == args.baseline]
            if not matching:
                print(
                    f"error: baseline {args.baseline!r} is not one of: "
                    + ", ".join(run.name for run in runs),
                    file=sys.stderr,
                )
                return 2
            baseline = matching[0]
            runs = [baseline] + [run for run in runs if run is not baseline]
        smoothed = {
            run.name: smooth_loss(
                run.records,
                ema_halflife=args.ema_halflife,
                rolling_window=args.rolling_window,
                centered_window=args.centered_window,
            )
            for run in runs
        }
        thresholds = specified_thresholds
        if not thresholds:
            thresholds = automatic_thresholds(smoothed, args.curve, args.auto_thresholds)
        print_report(
            runs,
            smoothed,
            thresholds=thresholds,
            curve=args.curve,
            at_steps=args.at_step,
            slope_window=args.slope_window,
            confirm_points=args.confirm_points,
        )
        if args.csv:
            write_csv(args.csv, runs, smoothed)
            print()
            print(f"Wrote CSV: {args.csv}")
        if args.html:
            write_html(
                args.html,
                runs,
                smoothed,
                thresholds=thresholds,
                curve=args.curve,
                confirm_points=args.confirm_points,
                show_raw=not args.no_raw,
                ema_halflife=args.ema_halflife,
                rolling_window=args.rolling_window,
                centered_window=args.centered_window,
            )
            print()
            print(f"Wrote HTML: {args.html}")
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

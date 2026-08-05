from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from diagnostics.f4_loss_compare import (
    TrainingRecord,
    _expand_run_specs,
    _parse_threshold_specs,
    automatic_thresholds,
    build_argument_parser,
    find_crossings,
    first_reach,
    interpolate_consumed_samples,
    load_run,
    main,
    parse_line,
    smooth_loss,
    write_html,
)


SAMPLE = """\
[Rank 0 | Local Rank 0] 2026-08-05 01:43:46,057 INFO [mindspeed_mm.fsdp.train.train_engine:47] =>  [2026-08-05 01:43:46] iteration      827/    5000 | consumed samples:       211712 | elapsed time per iteration (ms): 52504.8 | learning rate: 3.786076E-04 | global batch size:   256 | loss: 3.439196E+00 | mtp_1 loss: 3.615747E+00 | aux loss: 1.611671E+01 | grad norm: 0.402 |
[Rank 0 | Local Rank 0] 2026-08-05 01:44:38,417 INFO [mindspeed_mm.fsdp.train.train_engine:47] =>  [2026-08-05 01:44:38] iteration      828/    5000 | consumed samples:       211968 | elapsed time per iteration (ms): 52357.4 | learning rate: 3.785536E-04 | global batch size:   256 | loss: 3.441777E+00 | mtp_1 loss: 3.622087E+00 | aux loss: 1.608987E+01 | grad norm: 0.371 |
[Rank 0 | Local Rank 0] 2026-08-05 01:45:31,362 INFO [mindspeed_mm.fsdp.train.train_engine:47] =>  [2026-08-05 01:45:31] iteration      829/    5000 | consumed samples:       212224 | elapsed time per iteration (ms): 52939.4 | learning rate: 3.784995E-04 | global batch size:   256 | loss: 3.393171E+00 | mtp_1 loss: 3.575280E+00 | aux loss: 1.608639E+01 | grad norm: 0.373 |
[Rank 0 | Local Rank 0] 2026-08-05 01:46:24,732 INFO [mindspeed_mm.fsdp.train.train_engine:47] =>  [2026-08-05 01:46:24] iteration      830/    5000 | consumed samples:       212480 | elapsed time per iteration (ms): 53362.9 | learning rate: 3.784453E-04 | global batch size:   256 | loss: 3.410245E+00 | mtp_1 loss: 3.590187E+00 | aux loss: 1.610828E+01 | grad norm: 0.406 |
"""


def make_records(losses: list[float]) -> list[TrainingRecord]:
    return [
        TrainingRecord(
            iteration=index,
            total_iterations=len(losses),
            consumed_samples=index,
            elapsed_ms=1.0,
            learning_rate=1e-3,
            global_batch_size=1,
            loss=loss,
        )
        for index, loss in enumerate(losses, start=1)
    ]


class ParseTests(unittest.TestCase):
    def test_help_contains_uv_multi_run_examples(self) -> None:
        help_text = build_argument_parser().format_help()
        self.assertIn("uv run --no-project diagnostics/f4_loss_compare.py", help_text)
        self.assertIn('--run "AdamW=D:\\logs\\adamw.log"', help_text)
        self.assertIn("--ema-halflife 50", help_text)
        self.assertIn("Explicit equal-loss levels", help_text)
        self.assertIn("-run", help_text)
        self.assertIn("comma-separated", help_text)

    def test_relative_glob_auto_names_and_numbers_collisions(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            first_directory = root / "first"
            second_directory = root / "second"
            first_directory.mkdir()
            second_directory.mkdir()
            first = first_directory / "train.log"
            second = second_directory / "train.log"
            first.write_text(SAMPLE, encoding="utf-8")
            second.write_text(SAMPLE, encoding="utf-8")
            relative_glob = str((root / "*" / "train.log").relative_to(Path.cwd()))
            names, paths = _expand_run_specs([relative_glob])
        self.assertEqual(names, ["train.log", "train.log#2"])
        self.assertEqual([path.name for path in paths], ["train.log", "train.log"])

    def test_explicit_name_has_priority_over_auto_filename(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            explicit = root / "explicit.log"
            automatic = root / "F1"
            explicit.write_text(SAMPLE, encoding="utf-8")
            automatic.write_text(SAMPLE, encoding="utf-8")
            relative_automatic = str(automatic.relative_to(Path.cwd()))
            names, _ = _expand_run_specs(
                [f"F1={explicit}", relative_automatic]
            )
        self.assertEqual(names, ["F1", "F1#2"])

    def test_comma_separated_and_repeated_thresholds(self) -> None:
        self.assertEqual(
            _parse_threshold_specs(["4.5, 4.4,4.3", "4.2"]),
            [4.5, 4.4, 4.3, 4.2],
        )

    def test_user_sample(self) -> None:
        records = [
            record
            for index, line in enumerate(SAMPLE.splitlines(), start=1)
            if (record := parse_line(line, index)) is not None
        ]
        self.assertEqual([record.iteration for record in records], [827, 828, 829, 830])
        self.assertEqual(records[0].consumed_samples, 211712)
        self.assertEqual(records[0].global_batch_size, 256)
        self.assertAlmostEqual(records[0].learning_rate or 0.0, 3.786076e-4)
        self.assertAlmostEqual(records[2].loss or 0.0, 3.393171)
        self.assertAlmostEqual(records[3].mtp_losses[1], 3.590187)
        self.assertAlmostEqual(records[1].aux_loss or 0.0, 16.08987)
        self.assertAlmostEqual(records[3].grad_norm or 0.0, 0.406)

    def test_load_ignores_noise_and_checks_sample_increments(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "sample.log"
            path.write_text("unrelated\n" + SAMPLE + "shutdown\n", encoding="utf-8")
            run = load_run(path, "sample")
        self.assertEqual(len(run.records), 4)
        self.assertEqual(len(run.warnings), 1)
        self.assertIn("starts at iteration 827", run.warnings[0])
        self.assertIn("left-censored", run.warnings[0])

    def test_named_multi_run_cli_writes_csv_and_html(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            first = root / "first.log"
            second = root / "second.log"
            csv_path = root / "comparison.csv"
            html_path = root / "comparison.html"
            first.write_text(SAMPLE, encoding="utf-8")
            second.write_text(SAMPLE, encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = main(
                    [
                        "--run",
                        f"F1={first}",
                        "--run",
                        f"F2={second}",
                        "--baseline",
                        "F1",
                        "--threshold",
                        "3.42,3.41",
                        "--confirm-points",
                        "1",
                        "--csv",
                        str(csv_path),
                        "--html",
                        str(html_path),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertTrue(csv_path.is_file())
            self.assertTrue(html_path.is_file())
            self.assertIn("F2 - F1", stdout.getvalue())
            self.assertIn("consumed samples", html_path.read_text(encoding="utf-8"))


class SmoothingTests(unittest.TestCase):
    def test_bias_aware_ema_preserves_constant(self) -> None:
        points = smooth_loss(make_records([4.0] * 10), 20.0, 5)
        self.assertTrue(all(abs(point.ema - 4.0) < 1e-12 for point in points))
        self.assertTrue(all(abs(point.rolling - 4.0) < 1e-12 for point in points))

    def test_threshold_and_crossing_interpolation(self) -> None:
        reference = smooth_loss(make_records([5.0, 4.0, 3.0]), 0.01, 1)
        candidate = smooth_loss(make_records([4.0, 4.5, 2.0]), 0.01, 1)
        self.assertAlmostEqual(first_reach(reference, 4.5, "raw") or 0.0, 1.5)
        crossings = find_crossings(reference, candidate, "raw")
        self.assertAlmostEqual(crossings[0][0], 1.6666666667)
        self.assertAlmostEqual(crossings[1][0], 2.3333333333)

    def test_confirm_points_rejects_one_point_false_crossing(self) -> None:
        points = smooth_loss(make_records([5.0, 3.9, 4.2, 3.8, 3.7, 3.6]), 0.01, 1)
        reached = first_reach(points, 4.0, "raw", confirm_points=3)
        self.assertAlmostEqual(reached or 0.0, 3.5)

    def test_consumed_samples_interpolation(self) -> None:
        records = make_records([5.0, 4.0, 3.0])
        self.assertAlmostEqual(interpolate_consumed_samples(records, 1.5) or 0.0, 1.5)

    def test_multi_run_auto_thresholds_and_html(self) -> None:
        run_a_records = make_records([5.0, 4.5, 4.0, 3.5])
        run_b_records = make_records([5.1, 4.4, 3.9, 3.4])
        run_a = load_run_from_records("A", run_a_records)
        run_b = load_run_from_records("B", run_b_records)
        smoothed = {
            "A": smooth_loss(run_a_records, 1.0, 2),
            "B": smooth_loss(run_b_records, 1.0, 2),
        }
        thresholds = automatic_thresholds(smoothed, "ema", 3)
        self.assertEqual(len(thresholds), 3)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            output = Path(directory) / "report.html"
            write_html(
                output,
                [run_a, run_b],
                smoothed,
                thresholds,
                "ema",
                confirm_points=1,
                show_raw=True,
                ema_halflife=1.0,
                rolling_window=2,
                centered_window=0,
            )
            rendered = output.read_text(encoding="utf-8")
        self.assertIn("<svg", rendered)
        self.assertIn("consumed samples", rendered)
        self.assertIn("step lead", rendered)
        self.assertIn('id="loss-chart"', rendered)
        self.assertIn('id="threshold-chart"', rendered)
        self.assertIn(
            'class="axis-label">first confirmed iteration (lower is better)</text>',
            rendered,
        )
        self.assertIn(
            'text-anchor="middle" class="axis-label">smoothed loss threshold</text>',
            rendered,
        )
        self.assertIn('data-chart-controls="loss-chart"', rendered)
        self.assertIn('data-domain-bound="xMin"', rendered)
        self.assertIn('data-domain-bound="yMax"', rendered)
        self.assertIn('class="plot-data"', rendered)
        self.assertIn(
            'class="plot-clip" clip-path="url(#loss-chart-clip)"><g class="plot-data">',
            rendered,
        )
        self.assertIn('stroke-opacity="0.16"', rendered)
        self.assertIn('stroke-width="2.5"', rendered)
        self.assertIn("Shift+drag: box zoom", rendered)
        self.assertIn('data-apply-chart', rendered)
        self.assertIn('addEventListener("wheel"', rendered)
        self.assertIn('addEventListener("pointermove"', rendered)
        self.assertIn('dataLayer.setAttribute(', rendered)
        self.assertIn('`matrix(${scaleX} 0 0 ${scaleY}', rendered)
        self.assertNotIn('svg.setAttribute("viewBox"', rendered)


def load_run_from_records(name: str, records: list[TrainingRecord]):
    from diagnostics.f4_loss_compare import RunData

    return RunData(name=name, path=Path(name + ".log"), records=records, warnings=[])


if __name__ == "__main__":
    unittest.main()

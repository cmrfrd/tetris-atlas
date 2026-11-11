"""Checks for sweep result classification, units and process deadlines."""
import json
from pathlib import Path
import sys
import signal
import subprocess
from unittest.mock import MagicMock, call, patch
import tempfile
import unittest

import codec_sweep as sweep


class SweepTests(unittest.TestCase):
    def test_timeout_is_unknown_even_if_partial_log_mentions_a_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "build.log"
            log.write_text("codec count exceeds u64")
            build = {"log": str(log), "returncode": -15, "timed_out": True}
            self.assertEqual(sweep.compile_status(build), "compile_timeout")
            build["timed_out"] = False
            self.assertEqual(sweep.compile_status(build), "count_overflow")
            log.write_text("an unrelated compiler error")
            self.assertEqual(sweep.compile_status(build), "compile_error")

    def test_batch_estimates_are_converted_once_to_ns_per_board(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            raw = {
                key: {"point_estimate": 2560, "confidence_interval": {
                    "lower_bound": 2304, "upper_bound": 2816,
                }} for key in ("mean", "median", "std_dev", "median_abs_dev")
            }
            for operation in sweep.OPERATIONS:
                folder = home / "board_codec_test" / f"{operation}_256/new"
                folder.mkdir(parents=True)
                (folder / "estimates.json").write_text(json.dumps(raw))
                (folder / "sample.json").write_text(json.dumps({"iters": [1] * 20}))
            measurements = sweep.estimates(home, {"name": "test", "samples": 256})
            for value in measurements.values():
                self.assertEqual(value["n"], 20)
                self.assertEqual(value["mean_ns"], 10)
                self.assertEqual(value["mean_ci_ns"], [9, 11])
                self.assertEqual(value["boards_per_second"], 100_000_000)

    def test_failed_configs_have_no_fabricated_timing_or_size(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sweep.report(output, [{"rows": 6, "holes": 5, "status": "compile_timeout"}])
            report = (output / "report.md").read_text()
            self.assertIn("compile_timeout | — | — | — | — | —", report)
            self.assertEqual(len((output / "results.csv").read_text().splitlines()), 5)

    def test_reaped_process_group_is_not_signalled_again(self):
        process = MagicMock(pid=12345, returncode=-15)
        process.wait.side_effect = [subprocess.TimeoutExpired(["probe"], 1), None]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sweep.subprocess, "Popen", return_value=process):
                with patch.object(sweep.os, "killpg") as kill:
                    sweep.run(["probe"], Path(directory) / "log", 1)
        self.assertEqual(kill.call_args_list, [call(12345, signal.SIGTERM)])

    def test_unresponsive_process_group_is_killed(self):
        process = MagicMock(pid=12345, returncode=-9)
        process.wait.side_effect = [subprocess.TimeoutExpired(["probe"], 1),
                                    subprocess.TimeoutExpired(["probe"], 5), None]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sweep.subprocess, "Popen", return_value=process):
                with patch.object(sweep.os, "killpg") as kill:
                    sweep.run(["probe"], Path(directory) / "log", 1)
        self.assertEqual(kill.call_args_list, [call(12345, signal.SIGTERM),
                                              call(12345, signal.SIGKILL)])

    def test_subprocess_deadline_returns_and_reaps_the_process(self):
        with tempfile.TemporaryDirectory() as directory:
            result = sweep.run(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                Path(directory) / "timeout.log", 0.05,
            )
            self.assertTrue(result["timed_out"])
            self.assertIsNotNone(result["returncode"])
            self.assertLess(result["seconds"], 5)


if __name__ == "__main__":
    unittest.main()

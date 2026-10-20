"""Run with: python3 -m unittest discover -s tests -p 'test_*.py'."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_parallel_results import compare_timings, extract_timings
from plot_benchmarks import load_benchmark_data
from plot_index_structure import predict_segment


class ReportingTests(unittest.TestCase):
    def test_models_use_exported_coordinates_and_keep_legacy_support(self):
        segment = dict(start_idx=10, end_idx=100, key_origin=100, inv_scale=.1)
        values = np.array([100., 101., 102.])
        for model, params, expected in [
            ("LINEAR", dict(slope=2, intercept=10), [10, 12, 14]),
            ("QUADRATIC", dict(a=1, b=2, c=10), [10, 13, 18]),
            ("CUBIC", dict(a=1, b=2, c=3, d=10), [10, 10.321, 10.688]),
        ]:
            segment.update(model_type=model, params=params)
            np.testing.assert_allclose(predict_segment(segment, values), expected)
        legacy = dict(start_idx=10, end_idx=100, model_type="LINEAR",
                      params=dict(slope=2, intercept=-190))
        np.testing.assert_allclose(predict_segment(legacy, values), [10, 12, 14])

    def test_aggregates_units_and_range_functions_stay_separate(self):
        name = "JazzyIndex/Uniform/S64/N100/RandomMixed"
        runs = [
            dict(name=name, real_time=99, time_unit="us"),
            dict(name=name + "_mean", run_name=name, run_type="aggregate",
                 aggregate_name="mean", real_time=2, time_unit="us"),
            dict(name=name + "_stddev", run_name=name, run_type="aggregate",
                 aggregate_name="stddev", real_time=.5, time_unit="us"),
        ]
        for function in ["EqualRange", "LowerBound", "UpperBound"]:
            runs.append(dict(name=f"BM_JazzyIndex_{function}_64_Uniform/FoundMiddle/100",
                             real_time=3, time_unit="ns"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(json.dumps(dict(benchmarks=runs)))
            result = load_benchmark_data(path)
        self.assertEqual(result["Uniform"]["RandomMixed"][("JazzyIndex", 64)],
                         [(100, 2000)])
        for function in ["EqualRange", "LowerBound", "UpperBound"]:
            self.assertIn(f"Uniform ({function})", result)

    def test_comparison_rejects_missing_evidence_and_normalizes_units(self):
        with self.assertRaises(ValueError):
            compare_timings({"a": 1}, {"b": 1}, "unmatched")
        self.assertEqual(extract_timings(dict(benchmarks=[
            dict(name="a_mean", cpu_time=2, time_unit="us", run_type="aggregate"),
            dict(name="a", cpu_time=7, time_unit="us"),
        ])), {"a": 2000})


if __name__ == "__main__":
    unittest.main()

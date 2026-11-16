"""第一层的数学、时间边界与第二层事件接口回归检查。"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from costep_diagnosis.data_convert.data import FEATURE_COLUMNS, NormalScaler
from costep_diagnosis.layer1.cca import (
    CCAModel, ObjectSpec, Trajectory, alarm_events, alarm_mask,
    evaluate, graph_objects, make_pairs, model_predictions, read_trajectories,
)


def example_spec(targets=1):
    return ObjectSpec("对象", "设备", ["M1"], ["SM_001"],
                      [f"SM_{i + 1:03d}" for i in range(targets)],
                      [f"变量{i + 1}" for i in range(targets)], ["kPa"] * targets)


class CCATests(unittest.TestCase):
    def test_classic_cca_recovers_known_linear_prediction_with_singular_input(self):
        rng = np.random.default_rng(19)
        independent = rng.normal(size=(600, 4))
        history = np.column_stack([independent, independent[:, 0], np.ones(600)])
        future = independent @ rng.normal(size=(4, 2)) + [10, -3]
        model = CCAModel.fit(example_spec(2), 1, history, future)
        self.assertEqual(model.p_rank, 4)
        self.assertEqual(model.f_rank, 2)
        fresh = rng.normal(size=(30, 4))
        augmented = np.column_stack([fresh, fresh[:, 0], np.ones(30)])
        coefficients = np.linalg.lstsq(np.c_[independent, np.ones(600)], future, rcond=None)[0]
        np.testing.assert_allclose(model.predict(augmented), np.c_[fresh, np.ones(30)] @ coefficients, atol=1e-10)

    def test_constant_targets_remain_monitorable(self):
        model = CCAModel.fit(example_spec(), 1, np.arange(20.).reshape(10, 2), np.ones((10, 1)) * 4)
        self.assertEqual(model.f_rank, 0)
        np.testing.assert_array_equal(model.predict(np.zeros((3, 2))), np.full((3, 1), 4))
        model.calibrate(np.zeros((10, 1)), .99, np.array([4.]))
        scores, direction = model.score(np.array([[0.], [1.]]))
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(scores[0, 0], 0)
        self.assertGreater(scores[1, 0], 1)
        self.assertEqual(direction[1, 0], 1)

    def test_calibration_uses_median_and_absolute_residual_percentile(self):
        model = CCAModel.fit(example_spec(), 1, np.arange(20.).reshape(10, 2), np.arange(10.).reshape(10, 1))
        old_A, old_B, old_c = model.A.copy(), model.B.copy(), model.c.copy()
        residual = np.array([1., 2., 3., 4., 20.]).reshape(-1, 1)
        model.calibrate(residual, .99, np.array([1.]))
        self.assertEqual(model.center[0], 3)
        self.assertAlmostEqual(model.quantile[0], np.quantile(np.abs(residual - 3), .99))
        score, direction = model.score(np.array([[2.], [5.]]))
        self.assertEqual(direction[:, 0].tolist(), [-1, 1])
        np.testing.assert_allclose(score[:, 0], [1 / model.threshold[0], 2 / model.threshold[0]])
        for actual, expected in [(model.A, old_A), (model.B, old_B), (model.c, old_c)]:
            np.testing.assert_array_equal(actual, expected)

    def test_saved_model_preserves_predictions_and_calibration(self):
        rng = np.random.default_rng(2)
        p = rng.normal(size=(50, 3))
        model = CCAModel.fit(example_spec(), 3, p, p[:, :1] * 2 + 7)
        model.calibrate(np.arange(20.).reshape(20, 1), .99, np.array([1.]))
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "model"
            model.save(folder)
            loaded = CCAModel.load(folder)
        np.testing.assert_array_equal(model.predict(p), loaded.predict(p))
        np.testing.assert_array_equal(model.center, loaded.center)
        self.assertEqual(model.spec, loaded.spec)

    def test_windows_are_past_only_and_reset_at_missing_samples(self):
        raw = np.arange(24.).reshape(12, 2)
        times = np.r_[np.arange(6), np.arange(7, 13)] / 3600
        trajectory = Trajectory("file", "normal_train", raw, raw, times, 0, 1, np.zeros(12))
        history, future, indices = make_pairs(trajectory, [0, 1], [1], 3)
        self.assertEqual(indices.tolist(), [3, 4, 5, 9, 10, 11])
        np.testing.assert_array_equal(history[0], raw[:3].ravel())
        self.assertEqual(future[0, 0], raw[3, 1])
        trajectory.scaled = raw.copy()
        trajectory.scaled[3, 0] = np.nan
        _, _, indices = make_pairs(trajectory, [0, 1], [1], 3)
        self.assertEqual(indices.tolist(), [9, 10, 11])

    def test_consecutive_counts_reset_at_prediction_gaps_and_end_at_recovery(self):
        scores = np.array([2., 2., 2., 2., .5, 2., 2., 2.]).reshape(-1, 1)
        indices = np.array([3, 4, 5, 6, 7, 8, 10, 11])
        active = alarm_mask(scores, indices, 3)
        self.assertEqual(active[:, 0].tolist(), [False, False, True, True, False, False, False, False])

    def test_event_has_original_units_identity_onset_and_confirmation(self):
        raw = np.arange(20.).reshape(10, 2)
        trajectory = Trajectory("normal/file.csv", "normal_calibration", raw, raw,
                                np.arange(10) / 3600, 100, 1, np.zeros(10))
        model = CCAModel.fit(example_spec(), 1, raw, raw[:, :1])
        model.calibrate(np.array([[-1.], [0.], [1.]]), .99, np.array([1.]))
        values = {"indices": np.array([3, 4, 5, 6]), "actual": np.array([[11.], [12.], [13.], [14.]]),
                  "prediction": np.array([[9.], [10.], [11.], [12.]]), "residual": np.full((4, 1), 2.)}
        scores, direction = model.score(values["residual"])
        events = list(alarm_events(model, trajectory, values, scores, direction, 3))
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertAlmostEqual(event["onset_time_seconds"], 3)
        self.assertAlmostEqual(event["confirmation_time_seconds"], 5)
        self.assertEqual(event["onset_source_row"], 103)
        self.assertEqual(event["confirmation_source_row"], 105)
        self.assertEqual(event["actual"], 13)
        self.assertEqual(event["raw_residual"], 2)
        self.assertEqual(event["unit"], "kPa")
        self.assertEqual(event["module_names"], ["M1"])
        self.assertEqual(event["direction"], "偏高")
        self.assertNotIn("last_exceedance_time_h", event)
        self.assertNotIn("peak_score", event)

    def test_graph_uses_explicit_locations_and_module_membership(self):
        graph = {"nodes": [
            {"id": "m", "labels": ["模块"], "properties": {"模块名称": "M1"}},
            {"id": "e", "labels": ["设备"], "properties": {"设备名称": "设备1"}},
            {"id": "s", "labels": ["检测变量"], "properties": {"变量ID": "SM_001", "变量名称": "压力", "单位": "kPa"}},
            {"id": "u", "labels": ["操作参数"], "properties": {"参数ID": "XMV_01"}},
            {"id": "d", "labels": ["检测变量"], "properties": {"变量ID": "XMEAS_01"}},
        ], "edges": [
            {"source": "m", "target": "e", "type": "包含"},
            {"source": "s", "target": "e", "type": "检测位置"},
            {"source": "u", "target": "e", "type": "操作位置"},
            {"source": "d", "target": "e", "type": "检测位置"},
        ]}
        specs, audit = graph_objects(graph, ["XMV_01", "SM_001"], ["XMEAS_01"])
        self.assertEqual(specs[0].history_ids, ["XMV_01", "SM_001"])
        self.assertEqual(specs[0].target_ids, ["SM_001"])
        self.assertEqual(specs[0].modules, ["M1"])
        self.assertEqual(audit["features_without_location"], [])

    def test_storage_chunks_join_but_train_calibration_windows_stay_separate(self):
        width = len(FEATURE_COLUMNS)
        raw = np.arange(16 * width, dtype=float).reshape(16, width)
        source = "train_faultfree/file.csv"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            groups = []
            for split, start, stop, cuts in [("normal_train", 0, 8, [0, 3, 8]),
                                              ("normal_calibration", 8, 16, [8, 12, 16])]:
                paths = []
                for part, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
                    relative = f"{split}/{part}.npz"
                    (root / split).mkdir(exist_ok=True)
                    np.savez(root / relative, X=raw[a:b].astype(np.float32), X_raw=raw[a:b],
                             feature_names=np.array(FEATURE_COLUMNS), time_h=np.arange(a, b) / 3600,
                             y_class=np.zeros(b - a, int))
                    paths.append(relative)
                groups.append({"split": split, "row_start": start, "row_stop": stop, "parts": paths})
            report = {"complete": True, "prepared": True, "feature_names": list(FEATURE_COLUMNS),
                      "scaler_fit_split": "normal_train", "files": [{"source": source,
                      "normal_groups": groups, "sample_interval_seconds": [1., 1.]}]}
            (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
            scaler = NormalScaler(np.zeros(width), np.ones(width), [], [], 8)
            training = list(read_trajectories(root, "normal_train", scaler))
            calibration = list(read_trajectories(root, "normal_calibration", scaler))
        self.assertEqual(len(training), 1)
        self.assertEqual(len(calibration), 1)
        p_train, f_train, idx_train = make_pairs(training[0], [0], [0], 3)
        p_cal, f_cal, idx_cal = make_pairs(calibration[0], [0], [0], 3)
        self.assertEqual(len(idx_train), 5)
        self.assertEqual(len(idx_cal), 5)
        self.assertEqual(p_train[1, -1], raw[3, 0])
        self.assertEqual(p_cal[0, 0], raw[8, 0])
        self.assertEqual(f_cal[0, 0], raw[11, 0])


class TrajectoryExportTests(unittest.TestCase):
    def setUp(self):
        width = len(FEATURE_COLUMNS)
        self.scaler = NormalScaler(np.zeros(width), np.ones(width), [], [], 5)
        self.model = CCAModel.fit(example_spec(), 1, np.zeros((5, 1)), np.zeros((5, 1)))
        self.model.calibrate(np.array([[-1.], [0.], [1.]]), .99, np.array([1.]))

    def trajectory(self, source, value=2., split="train_fault"):
        raw = np.zeros((8, len(FEATURE_COLUMNS)))
        raw[:, FEATURE_COLUMNS.index("SM_001")] = value
        return Trajectory(source, split, raw, raw.copy(), np.arange(8) / 3600, 0, 1,
                          np.r_[np.zeros(4, int), np.ones(4, int)])

    def test_each_fault_is_published_before_the_next_trajectory_is_requested(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "predictions"

            def trajectories():
                yield self.trajectory("train_fault/fault_01.csv")
                first = output / "fault_01"
                self.assertTrue((first / "COMPLETE.json").exists())
                self.assertFalse((first / "INCOMPLETE.txt").exists())
                self.assertTrue((output / "INCOMPLETE.txt").exists())
                self.assertFalse((output / "events.jsonl").exists())
                self.assertFalse((output / "report.json").exists())
                # 在第二条轨迹开始之前，可以完整读取第一条轨迹的事件与逐点文件。
                with np.load(first / "object_000.npz", allow_pickle=False) as points:
                    self.assertEqual(points["raw_residual"].shape, (7, 1))
                events = (first / "events.jsonl").read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(events), 1)
                self.assertEqual(json.loads(events[0])["source"], "train_fault/fault_01.csv")
                yield self.trajectory("train_fault/fault_02.csv", value=-2.)

            result = evaluate([self.model], trajectories(), self.scaler, [1, 3], 3, output,
                              per_trajectory_output=True)
            self.assertEqual(result["output_layout"], "per_trajectory")
            self.assertEqual(len(result["trajectories"]), 2)
            self.assertFalse((output / "INCOMPLETE.txt").exists())
            self.assertTrue((output / "report.json").exists())
            normal_points = false_alarm_points = 0
            for number, direction in [(1, "偏高"), (2, "偏低")]:
                folder = output / f"fault_{number:02d}"
                completion = json.loads((folder / "COMPLETE.json").read_text(encoding="utf-8"))
                self.assertEqual(completion["alarm_event_count"], 1)
                self.assertEqual(completion["point_files"], [{"object_name": "对象", "file": "object_000.npz"}])
                events = [json.loads(line) for line in (folder / completion["events_file"]).read_text(encoding="utf-8").splitlines()]
                self.assertEqual({event["source"] for event in events}, {f"train_fault/fault_{number:02d}.csv"})
                self.assertEqual(events[0]["direction"], direction)
                for field in ["variable_id", "object_name", "module_names", "onset_time_h", "confirmation_time_h",
                              "score", "actual", "prediction", "raw_residual", "unit"]:
                    self.assertIn(field, events[0])
                report = json.loads((folder / completion["report_file"]).read_text(encoding="utf-8"))
                self.assertEqual(len(report["objects"]), 1)
                normal_points += report["rules"]["3"]["normal_points"]
                false_alarm_points += report["rules"]["3"]["false_alarm_points"]
            self.assertEqual(result["rules"]["3"]["normal_points"], normal_points)
            self.assertEqual(result["rules"]["3"]["false_alarm_points"], false_alarm_points)

    def test_later_failure_keeps_the_first_fault_ready_and_failed_fault_incomplete(self):
        def fail_second(model, trajectory, scaler):
            if "fault_02" in trajectory.source:
                raise ValueError("模拟第二条轨迹处理失败")
            return model_predictions(model, trajectory, scaler)

        with TemporaryDirectory() as directory:
            output = Path(directory) / "predictions"
            trajectories = [self.trajectory("fault_01.csv"), self.trajectory("fault_02.csv")]
            with patch("costep_diagnosis.layer1.cca.model_predictions", side_effect=fail_second):
                with self.assertRaisesRegex(ValueError, "第二条轨迹"):
                    evaluate([self.model], trajectories, self.scaler, [3], 3, output,
                             per_trajectory_output=True)
            self.assertTrue((output / "fault_01" / "COMPLETE.json").exists())
            self.assertFalse((output / "fault_01" / "INCOMPLETE.txt").exists())
            self.assertEqual(len((output / "fault_01" / "events.jsonl").read_text(encoding="utf-8").splitlines()), 1)
            self.assertTrue((output / "fault_02" / "INCOMPLETE.txt").exists())
            self.assertFalse((output / "fault_02" / "COMPLETE.json").exists())
            self.assertTrue((output / "INCOMPLETE.txt").exists())
            self.assertFalse((output / "report.json").exists())

    def test_fault_without_alarm_still_publishes_an_empty_event_file(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "predictions"
            evaluate([self.model], [self.trajectory("fault_01.csv", value=0.)], self.scaler,
                     [3], 3, output, per_trajectory_output=True)
            folder = output / "fault_01"
            self.assertEqual((folder / "events.jsonl").read_text(encoding="utf-8"), "")
            completion = json.loads((folder / "COMPLETE.json").read_text(encoding="utf-8"))
            self.assertEqual(completion["alarm_event_count"], 0)
            self.assertTrue((folder / completion["report_file"]).exists())
            self.assertTrue((folder / "object_000.npz").exists())

    def test_normal_evaluation_preserves_the_shared_event_output(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "predictions"
            result = evaluate([self.model], [self.trajectory("normal.csv", split="normal_calibration")],
                              self.scaler, [1, 3], 3, output)
            self.assertEqual(len(result["objects"]), 1)
            self.assertEqual(len((output / "events.jsonl").read_text(encoding="utf-8").splitlines()), 1)
            self.assertTrue((output / "normal" / "object_000.npz").exists())
            self.assertNotIn("output_layout", result)
            self.assertFalse((output / "INCOMPLETE.txt").exists())


if __name__ == "__main__":
    unittest.main()

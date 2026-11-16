"""离线验证预处理语义，不修改原始数据。"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from costep_diagnosis.data_convert.data import (
    DataError, EXCLUDED_FEATURE_COLUMNS, FEATURE_COLUMNS, METADATA_COLUMNS,
    RAW_FEATURE_COLUMNS, NormalScaler, SPLITS, iter_csv_chunks, normal_groups,
    iter_prepared_chunks, process_dataset,
)


class DataTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "COSTE"
        for split in SPLITS:
            folder = self.source / split
            folder.mkdir(parents=True)
            fault_id = 0 if split.endswith("faultfree") else 3
            base = np.arange(6, dtype=float)
            if split != "train_faultfree":
                base += 1000
            frame = pd.DataFrame({name: base + i for i, name in enumerate(RAW_FEATURE_COLUMNS)})
            frame["SM_003"] = frame["XMEAS_02"]
            frame["SM_004"] = frame["SM_003"]
            frame["XMV_12"] = 50.0 if split == "train_faultfree" else 80.0
            frame.insert(0, "random_seed", 100 if split.startswith("train_") else 200)
            frame.insert(0, "fault_label", [0, 0, fault_id, fault_id, fault_id, fault_id])
            frame.insert(0, "fault_id", fault_id)
            frame.insert(0, "scenario_index", 1 if fault_id == 0 else 2)
            frame.insert(0, "time_h", np.arange(6) / 3600)
            frame.to_csv(folder / "trajectory.csv", index=False)

    def run_pipeline(self, name="out", **kwargs):
        chunksize = kwargs.pop("chunksize", 2)
        with redirect_stdout(io.StringIO()):
            report = process_dataset(self.source, self.root / name, chunksize=chunksize, **kwargs)
        return report

    def test_scaler_uses_only_normal_training_and_preserves_fault_changes(self):
        report = self.run_pipeline()
        scaler = NormalScaler.load(self.root / "out" / "scaler.json")
        self.assertEqual(report["scaler_fit_rows"], 4)
        self.assertEqual(report["scaler_fit_split"], "normal_train")
        self.assertAlmostEqual(scaler.mean[0], 1.5 + RAW_FEATURE_COLUMNS.index(FEATURE_COLUMNS[0]))
        self.assertAlmostEqual(scaler.scale[0], np.std(np.arange(4)))
        constant_index = FEATURE_COLUMNS.index("XMV_12")
        self.assertEqual(scaler.mean[constant_index], 50.0)
        self.assertEqual(scaler.scale[constant_index], 1.0)
        chunk = next(iter_csv_chunks(self.source / "test_fault" / "trajectory.csv"))
        self.assertTrue((scaler.transform(chunk)[:, constant_index] == 30).all())
        self.assertEqual(report["class_values"], [0, 3])
        self.assertEqual(report["class_to_index"], {"0": 0, "3": 1})
        self.assertIn(["SM_003", "SM_004"], report["duplicate_feature_groups"])
        self.assertEqual(report["excluded_feature_names"], list(EXCLUDED_FEATURE_COLUMNS))
        self.assertEqual(len(scaler.mean), len(FEATURE_COLUMNS))

    def test_excluded_columns_removed_by_name_and_counterparts_preserved(self):
        path = self.source / "train_faultfree" / "trajectory.csv"
        original = pd.read_csv(path)
        # 打乱原始列顺序后，仍须准确排除指定变量，并按固定顺序返回保留变量。
        original.loc[:, list(reversed(original.columns))].to_csv(path, index=False)
        chunks = list(iter_csv_chunks(path, chunksize=2))
        retained = pd.concat(chunks, ignore_index=True)
        self.assertEqual(tuple(retained.columns), METADATA_COLUMNS + FEATURE_COLUMNS)
        self.assertEqual(len(FEATURE_COLUMNS), 161)
        self.assertEqual(len(EXCLUDED_FEATURE_COLUMNS), 30)
        self.assertEqual(
            [name for name in FEATURE_COLUMNS if name.startswith("XMEAS_")],
            ["XMEAS_07", "XMEAS_08", "XMEAS_09", "XMEAS_11", "XMEAS_12",
             "XMEAS_13", "XMEAS_15", "XMEAS_16", "XMEAS_18", "XMEAS_19", "XMEAS_20"],
        )
        self.assertTrue(set(EXCLUDED_FEATURE_COLUMNS).isdisjoint(retained.columns))
        np.testing.assert_array_equal(retained["SM_003"], original["SM_003"])
        np.testing.assert_array_equal(retained["SM_003"], original["XMEAS_02"])

    def test_export_labels_boundaries_and_raw_units(self):
        report = self.run_pipeline(prepare=True, keep_raw=True)
        chunks = list(iter_prepared_chunks(self.root / "out", "train_fault"))
        self.assertEqual(len(chunks), 3)
        np.testing.assert_array_equal(np.concatenate([c["y_class"] for c in chunks]), [0, 0, 3, 3, 3, 3])
        np.testing.assert_array_equal(np.concatenate([c["y_detect"] for c in chunks]), [0, 0, 1, 1, 1, 1])
        self.assertEqual(chunks[0]["X"].dtype, np.float32)
        self.assertEqual(chunks[0]["X_raw"].dtype, np.float64)
        self.assertEqual(chunks[0]["X_raw"][0, 0], 1000 + RAW_FEATURE_COLUMNS.index(FEATURE_COLUMNS[0]))
        self.assertEqual(chunks[0]["X"].shape[1], len(FEATURE_COLUMNS))
        self.assertEqual(chunks[0]["X_raw"].shape[1], len(FEATURE_COLUMNS))
        np.testing.assert_array_equal(chunks[0]["feature_names"], FEATURE_COLUMNS)
        self.assertEqual(str(chunks[0]["source"]), "train_fault/trajectory.csv")
        self.assertEqual(chunks[0]["random_seed"].dtype, np.int64)
        self.assertTrue(report["prepared"])
        self.assertFalse((self.root / "out" / "INCOMPLETE.txt").exists())

    def test_reject_artifacts_with_old_feature_set(self):
        self.run_pipeline(prepare=True)
        for filename in ("report.json", "scaler.json"):
            path = self.root / "out" / filename
            params = json.loads(path.read_text(encoding="utf-8"))
            params["feature_names"] = list(RAW_FEATURE_COLUMNS)
            path.write_text(json.dumps(params), encoding="utf-8")
        with self.assertRaises(DataError):
            NormalScaler.load(self.root / "out" / "scaler.json")
        with self.assertRaises(DataError):
            list(iter_prepared_chunks(self.root / "out", "train_fault"))

    def test_calibration_data_does_not_change_learned_parameters(self):
        path = self.source / "train_faultfree" / "trajectory.csv"
        frame = pd.read_csv(path)
        frame.loc[4:, "XMEAS_07"] += 10_000
        frame.loc[4:, "XMV_12"] = 80.0
        frame.loc[4:, "SM_004"] += 500
        frame.to_csv(path, index=False)
        report = self.run_pipeline(prepare=True)
        scaler = NormalScaler.load(self.root / "out" / "scaler.json")
        self.assertAlmostEqual(scaler.mean[0], frame.XMEAS_07.iloc[:4].mean())
        constant_index = FEATURE_COLUMNS.index("XMV_12")
        self.assertEqual(scaler.mean[constant_index], 50)
        self.assertEqual(scaler.scale[constant_index], 1)
        self.assertIn(["SM_003", "SM_004"], scaler.duplicate_feature_groups)
        calibration = list(iter_prepared_chunks(self.root / "out", "normal_calibration"))
        self.assertTrue((np.concatenate([c["X"] for c in calibration])[:, constant_index] == 30).all())
        self.assertEqual(report["normal_split"]["train_rows"], 4)
        self.assertEqual(report["normal_split"]["calibration_rows"], 2)

    def test_normal_groups_are_disjoint_even_inside_storage_chunk(self):
        report = self.run_pipeline(prepare=True, keep_raw=True, chunksize=3)
        train = list(iter_prepared_chunks(self.root / "out", "normal_train"))
        calibration = list(iter_prepared_chunks(self.root / "out", "normal_calibration"))
        train_times = np.concatenate([c["time_h"] for c in train])
        calibration_times = np.concatenate([c["time_h"] for c in calibration])
        np.testing.assert_allclose(train_times, np.arange(4) / 3600)
        np.testing.assert_allclose(calibration_times, np.arange(4, 6) / 3600)
        self.assertLess(train_times[-1], calibration_times[0])
        self.assertEqual(str(train[0]["split"]), "normal_train")
        self.assertEqual(str(calibration[0]["split"]), "normal_calibration")
        self.assertEqual(report["files"][0]["normal_groups"][0]["row_stop"], 4)
        with self.assertRaises(DataError):
            list(iter_prepared_chunks(self.root / "out", "train_faultfree"))

    def test_isolation_rows_never_enter_scaler_or_exported_groups(self):
        path = self.source / "train_faultfree" / "trajectory.csv"
        frame = pd.read_csv(path)
        frame.loc[2:3, "XMEAS_07"] += 10_000
        frame.to_csv(path, index=False)
        report = self.run_pipeline(prepare=True, normal_gap_rows=2)
        scaler = NormalScaler.load(self.root / "out" / "scaler.json")
        self.assertAlmostEqual(scaler.mean[0], frame.XMEAS_07.iloc[:2].mean())
        train = list(iter_prepared_chunks(self.root / "out", "normal_train"))
        calibration = list(iter_prepared_chunks(self.root / "out", "normal_calibration"))
        np.testing.assert_allclose(np.concatenate([c["time_h"] for c in train]), np.arange(2) / 3600)
        np.testing.assert_allclose(np.concatenate([c["time_h"] for c in calibration]), np.arange(4, 6) / 3600)
        self.assertEqual(report["normal_split"]["excluded_gap_rows"], 2)
        self.assertEqual(scaler.n_samples, 2)

    def test_reject_old_parameters_even_when_features_match(self):
        self.run_pipeline(prepare=True)
        for filename, field_name in (("scaler.json", "fit_split"), ("report.json", "scaler_fit_split")):
            path = self.root / "out" / filename
            params = json.loads(path.read_text(encoding="utf-8"))
            params[field_name] = "train_faultfree"
            path.write_text(json.dumps(params), encoding="utf-8")
        with self.assertRaises(DataError):
            NormalScaler.load(self.root / "out" / "scaler.json")
        with self.assertRaises(DataError):
            list(iter_prepared_chunks(self.root / "out", "normal_calibration"))

    def test_invalid_normal_split_configuration(self):
        for ratio, gap in [(0, 0), (1, 0), (float("nan"), 0), (0.7, -1), (0.7, 3)]:
            with self.subTest(ratio=ratio, gap=gap):
                with self.assertRaises(DataError):
                    normal_groups(6, ratio, gap)

    def test_reject_nonfinite_and_invalid_labels(self):
        path = self.source / "train_fault" / "trajectory.csv"
        original = pd.read_csv(path)
        for column, value in [("SM_001", np.nan), ("XMEAS_07", np.inf), ("fault_label", 2)]:
            with self.subTest(column=column, value=value):
                frame = original.copy()
                frame.loc[3, column] = value
                frame.to_csv(path, index=False)
                with self.assertRaises(DataError):
                    list(iter_csv_chunks(path, chunksize=2))

    def test_reject_duplicate_time_and_changed_identity_across_chunks(self):
        path = self.source / "train_fault" / "trajectory.csv"
        original = pd.read_csv(path)
        for column, value in [("time_h", original.time_h.iloc[1]), ("random_seed", 999)]:
            with self.subTest(column=column):
                frame = original.copy()
                frame.loc[2, column] = value
                frame.to_csv(path, index=False)
                with self.assertRaises(DataError):
                    list(iter_csv_chunks(path, chunksize=2))

    def test_protect_source_and_existing_output(self):
        with self.assertRaises(DataError):
            process_dataset(self.source, self.source / "processed")
        self.assertFalse((self.source / "processed").exists())
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaises(DataError):
            process_dataset(self.source, existing)

    def test_failed_export_is_marked_incomplete(self):
        path = self.source / "test_fault" / "trajectory.csv"
        frame = pd.read_csv(path)
        frame.loc[3, "SM_001"] = np.nan
        frame.to_csv(path, index=False)
        with self.assertRaises(DataError):
            self.run_pipeline(prepare=True)
        self.assertTrue((self.root / "out" / "INCOMPLETE.txt").exists())
        self.assertFalse((self.root / "out" / "report.json").exists())
        with self.assertRaises(DataError):
            list(iter_prepared_chunks(self.root / "out", "train_fault"))


if __name__ == "__main__":
    unittest.main()

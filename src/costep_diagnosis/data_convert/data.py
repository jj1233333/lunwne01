r"""COSTE 数据检查与预处理（在项目根目录运行）。

    .venv\Scripts\python.exe -m costep_diagnosis.data_convert.data
    .venv\Scripts\python.exe -m costep_diagnosis.data_convert.data --prepare

默认仅生成质量报告和标准化参数；--prepare 另外导出分块 NPZ。
将 train_faultfree 按时间分成正常训练组和正常校准组，只用正常训练组拟合均值/标准差。
默认按 7:3 划分，可配置两组间的隔离行数。按变量名排除 30 个重复 XMEAS，保留 161 个过程变量，
其余常量和重复变量仅记录。原始 CSV 的物理量用于守恒计算；
--keep-raw 可以在 NPZ 中同时保存保留变量的原始物理量。
正常校准组复用训练参数，供后续 CCA 残差和报警规则校准；原始测试集用于最终评估。
不打乱时序、不填补缺失、不平滑故障、不重采样。
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = PROJECT_ROOT / "data" / "COSTE"
SPLITS = ("train_faultfree", "train_fault", "test_faultfree", "test_fault")
NORMAL_SPLITS = ("normal_train", "normal_calibration")
PREPARED_SPLITS = NORMAL_SPLITS + SPLITS[1:]
DEFAULT_NORMAL_TRAIN_RATIO = 0.7
METADATA_COLUMNS = ("time_h", "scenario_index", "fault_id", "fault_label", "random_seed")
RAW_FEATURE_COLUMNS = (
    tuple(f"XMEAS_{i:02d}" for i in range(1, 42))
    + tuple(f"XMV_{i:02d}" for i in range(1, 13))
    + tuple(f"SM_{i:03d}" for i in range(1, 139))
)
# 原始文件仍按完整表头校验，模型输入和导出结果按变量名筛选，避免列移动误删。
EXCLUDED_XMEAS_NUMBERS = tuple(range(1, 7)) + (10, 14, 17) + tuple(range(21, 42))
EXCLUDED_FEATURE_COLUMNS = tuple(f"XMEAS_{i:02d}" for i in EXCLUDED_XMEAS_NUMBERS)
FEATURE_COLUMNS = tuple(name for name in RAW_FEATURE_COLUMNS if name not in EXCLUDED_FEATURE_COLUMNS)
ALL_COLUMNS = METADATA_COLUMNS + RAW_FEATURE_COLUMNS
PROCESSED_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS
EXCLUSION_REASON = "按用户指定排除与 COSTEP SM（smeas）检测变量重复的经典 TE XMEAS 变量"


class DataError(ValueError):
    """源数据不满足时序、标签或数值要求。"""


def validate_normal_split(train_ratio: float, gap_rows: int) -> None:
    if not np.isfinite(train_ratio) or not 0 < train_ratio < 1:
        raise DataError("正常训练组比例必须在 0 与 1 之间")
    if not isinstance(gap_rows, (int, np.integer)) or gap_rows < 0:
        raise DataError("正常训练/校准组之间的隔离行数必须为非负整数")


def normal_groups(rows: int, train_ratio: float, gap_rows: int) -> list[dict]:
    """先扣除隔离段，再按比例分配；行号从 0 开始，row_stop 为开区间。"""
    validate_normal_split(train_ratio, gap_rows)
    train_stop = int(np.floor((rows - gap_rows) * train_ratio))
    calibration_start = train_stop + gap_rows
    if train_stop < 2 or rows - calibration_start < 2:
        raise DataError("正常训练组和正常校准组各至少需要 2 行，请调整比例或隔离行数")
    return [
        {"split": "normal_train", "row_start": 0, "row_stop": train_stop,
         "rows": train_stop, "start_h": None, "end_h": None, "parts": []},
        {"split": "normal_calibration", "row_start": calibration_start, "row_stop": rows,
         "rows": rows - calibration_start, "start_h": None, "end_h": None, "parts": []},
    ]


def normal_group_slices(chunk: pd.DataFrame, row_offset: int,
                        groups: list[dict]) -> Iterator[tuple[dict, pd.DataFrame]]:
    """在 CSV 分块内部切出样本组；存储分块边界不改变训练/校准边界。"""
    for group in groups:
        start = max(group["row_start"], row_offset) - row_offset
        stop = min(group["row_stop"], row_offset + len(chunk)) - row_offset
        if start < stop:
            yield group, chunk.iloc[start:stop]


def discover_files(root: Path) -> dict[str, list[Path]]:
    files = {split: sorted((root / split).glob("*.csv")) for split in SPLITS}
    for split, paths in files.items():
        if not paths:
            raise DataError(f"{root / split}: 没有 CSV 文件")
    return files


def iter_csv_chunks(path: Path, chunksize: int = 20_000) -> Iterator[pd.DataFrame]:
    """校验后按原顺序读取一条轨迹；每个 CSV 须有固定场景、种子、故障编号。

    校验完整原始表后，返回元数据和筛选后的过程变量；指定的重复 XMEAS 不返回。
    元数据不作为模型特征。发现坏值或重复/倒序时间直接报错，避免误删故障。
    多条仿真轨迹应分别存放，后续滑窗也必须以原 CSV 为边界。
    """
    path = Path(path)
    if chunksize <= 0:
        raise ValueError("chunksize 必须大于 0")
    columns = tuple(pd.read_csv(path, nrows=0).columns)
    if len(columns) != len(ALL_COLUMNS) or set(columns) != set(ALL_COLUMNS):
        raise DataError(
            f"{path}: 列名不符合 COSTE 格式；缺少 {sorted(set(ALL_COLUMNS) - set(columns))}，"
            f"多出 {sorted(set(columns) - set(ALL_COLUMNS))}"
        )
    dtypes = {name: np.float64 for name in ALL_COLUMNS}
    dtypes.update({name: np.int64 for name in METADATA_COLUMNS if name != "time_h"})
    previous_time = None
    identity = None
    rows = 0
    try:
        with pd.read_csv(path, dtype=dtypes, chunksize=chunksize) as reader:
            for chunk in reader:
                chunk = chunk.loc[:, list(ALL_COLUMNS)]
                if chunk.empty:
                    continue
                finite = np.isfinite(chunk.to_numpy())
                if not finite.all():
                    bad = [name for name, ok in zip(ALL_COLUMNS, finite.all(axis=0)) if not ok]
                    raise DataError(f"{path}: 第 {rows + 2} 行起的分块有缺失/无穷值，列 {bad}")
                keys = chunk[["scenario_index", "fault_id", "random_seed"]].to_numpy()
                if identity is None:
                    identity = keys[0].copy()
                if not (keys == identity).all():
                    raise DataError(f"{path}: 一个 CSV 内有多条轨迹，请先按场景、种子和故障编号拆分")
                times = chunk.time_h.to_numpy()
                if (times < 0).any() or (np.diff(times) <= 0).any() or (
                    previous_time is not None and times[0] <= previous_time
                ):
                    raise DataError(f"{path}: 时间包含负值、重复或倒序；请核查原始轨迹")
                previous_time = float(times[-1])
                fault_id = int(identity[1])
                labels = chunk.fault_label.to_numpy()
                if fault_id < 0 or not ((labels == 0) | (labels == fault_id)).all():
                    raise DataError(f"{path}: fault_label 应为 0 或本轨迹 fault_id={fault_id}")
                if path.parent.name.endswith("faultfree") and fault_id != 0:
                    raise DataError(f"{path}: 正常工况文件的 fault_id 必须为 0")
                if path.parent.name in ("train_fault", "test_fault") and fault_id == 0:
                    raise DataError(f"{path}: 故障工况文件的 fault_id 必须大于 0")
                rows += len(chunk)
                yield chunk.loc[:, list(PROCESSED_COLUMNS)]
    except (ValueError, pd.errors.ParserError) as exc:
        if isinstance(exc, DataError):
            raise
        raise DataError(f"{path}: 无法解析数值或标签：{exc}") from exc
    if not rows:
        raise DataError(f"{path}: 数据为空")


@dataclass
class NormalScaler:
    mean: np.ndarray
    scale: np.ndarray
    constant_features: list[str]
    duplicate_feature_groups: list[list[str]]
    n_samples: int
    fit_partitions: list[dict] = field(default_factory=list)

    def transform(self, chunk: pd.DataFrame) -> np.ndarray:
        """返回 float32 模型输入；所有分区使用同一组正常训练参数。"""
        values = chunk.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            result = ((values - self.mean) / self.scale).astype(np.float32)
        if not np.isfinite(result).all():
            raise DataError("标准化后出现非有限值，请核查输入和变量尺度")
        return result

    def save(self, path: Path) -> None:
        write_json(path, {
            "feature_names": list(FEATURE_COLUMNS), "mean": self.mean.tolist(),
            "scale": self.scale.tolist(), "constant_features": self.constant_features,
            "duplicate_feature_groups": self.duplicate_feature_groups,
            "n_samples": self.n_samples, "fit_split": "normal_train", "ddof": 0,
            "fit_partitions": self.fit_partitions,
            "excluded_feature_names": list(EXCLUDED_FEATURE_COLUMNS),
            "exclusion_reason": EXCLUSION_REASON,
            "constant_policy": "保留该列；scale=1，数值为相对正常均值的原单位偏差",
        })

    @classmethod
    def load(cls, path: Path) -> NormalScaler:
        params = json.loads(Path(path).read_text(encoding="utf-8"))
        if params["feature_names"] != list(FEATURE_COLUMNS):
            raise DataError("标准化参数的特征名称或顺序不一致")
        if params.get("fit_split") != "normal_train":
            raise DataError("该参数未按正常训练/校准分组拟合，请重新预处理")
        mean, scale = np.array(params["mean"]), np.array(params["scale"])
        if mean.shape != (len(FEATURE_COLUMNS),) or scale.shape != mean.shape or (
            not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any()
        ):
            raise DataError("标准化参数无效")
        return cls(mean, scale, params["constant_features"],
                   params["duplicate_feature_groups"], params["n_samples"], params["fit_partitions"])


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def scan_file(path: Path, chunksize: int,
              visit: Callable[[pd.DataFrame, int], None] | None = None) -> dict:
    rows = 0
    counts: Counter = Counter()
    start = end = first_fault = None
    min_dt, max_dt = float("inf"), 0.0
    identity = None
    for part, chunk in enumerate(iter_csv_chunks(path, chunksize)):
        times = chunk.time_h.to_numpy()
        dt = np.diff(times if end is None else np.r_[end, times]) * 3600
        if len(dt):
            min_dt, max_dt = min(min_dt, float(dt.min())), max(max_dt, float(dt.max()))
        if start is None:
            start = float(times[0])
            identity = {name: int(chunk[name].iloc[0]) for name in (
                "scenario_index", "fault_id", "random_seed"
            )}
        end = float(times[-1])
        counts.update({int(k): int(v) for k, v in chunk.fault_label.value_counts().items()})
        fault = chunk.loc[chunk.fault_label.ne(0), "time_h"]
        if first_fault is None and not fault.empty:
            first_fault = float(fault.iloc[0])
        if visit is not None:
            visit(chunk, part)
        rows += len(chunk)
    return {
        "source": f"{path.parent.name}/{path.name}", "split": path.parent.name,
        "rows": rows, "start_h": start, "end_h": end, "first_fault_h": first_fault,
        "sample_interval_seconds": [min_dt, max_dt] if rows > 1 else None,
        "label_counts": dict(sorted(counts.items())), "normal_rows": counts[0],
        "fault_rows": rows - counts[0], "missing_or_infinite_values": 0, **identity,
    }


def fit_normal_scaler(paths: list[Path], chunksize: int = 20_000, *,
                      train_ratio: float = DEFAULT_NORMAL_TRAIN_RATIO,
                      gap_rows: int = 0) -> tuple[NormalScaler, list[dict]]:
    """每条正常轨迹按时间划分；仅用 normal_train 分块合并均值/方差。

    校准段和隔离段均不参与常量、重复列及标准化参数的学习。
    """
    validate_normal_split(train_ratio, gap_rows)
    size = len(FEATURE_COLUMNS)
    count, mean, m2 = 0, np.zeros(size), np.zeros(size)
    low, high = np.full(size, np.inf), np.full(size, -np.inf)
    hashes = [hashlib.sha256() for _ in FEATURE_COLUMNS]

    def update(values: np.ndarray) -> None:
        nonlocal count, mean, m2, low, high
        n = len(values)
        batch_mean = values.mean(axis=0)
        delta = batch_mean - mean
        m2 += ((values - batch_mean) ** 2).sum(axis=0) + delta ** 2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
        low, high = np.minimum(low, values.min(axis=0)), np.maximum(high, values.max(axis=0))
        for column, digest in enumerate(hashes):
            digest.update(values[:, column].tobytes())

    summaries = []
    for path in paths:
        # 先只计行；完整表的数值、时间、标签等随后在 scan_file 中校验。
        with pd.read_csv(path, usecols=["time_h"], chunksize=chunksize) as reader:
            rows = sum(len(chunk) for chunk in reader)
        groups = normal_groups(rows, train_ratio, gap_rows)
        row_offset = 0

        def visit(chunk: pd.DataFrame, part: int) -> None:
            nonlocal row_offset
            if chunk.fault_id.ne(0).any() or chunk.fault_label.ne(0).any():
                raise DataError("正常训练/校准源文件中不能含故障样本")
            for group, section in normal_group_slices(chunk, row_offset, groups):
                if group["start_h"] is None:
                    group["start_h"] = float(section.time_h.iloc[0])
                group["end_h"] = float(section.time_h.iloc[-1])
                if group["split"] == "normal_train":
                    update(section.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64))
            row_offset += len(chunk)

        summary = scan_file(path, chunksize, visit)
        if summary["rows"] != rows:
            raise DataError(f"{path}: 两次读取行数不同，请确认源文件未被修改")
        summary["normal_groups"] = groups
        summary["normal_gap"] = {
            "row_start": groups[0]["row_stop"], "row_stop": groups[1]["row_start"], "rows": gap_rows,
        }
        summaries.append(summary)
    if count < 2:
        raise DataError("正常训练数据至少需要两个样本")
    constant = low == high
    scale = np.sqrt(np.maximum(m2 / count, 0))
    scale[constant] = 1.0
    mean[constant] = low[constant]
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise DataError("正常训练数据的统计量无效，请核查变量数值范围")
    groups: dict[bytes, list[str]] = {}
    for name, digest in zip(FEATURE_COLUMNS, hashes):
        groups.setdefault(digest.digest(), []).append(name)
    scaler = NormalScaler(mean, scale, [name for name, flag in zip(FEATURE_COLUMNS, constant) if flag],
                          [group for group in groups.values() if len(group) > 1], count,
                          [{"source": s["source"], **{k: v for k, v in s["normal_groups"][0].items()
                                                     if k not in ("split", "parts")}}
                           for s in summaries])
    return scaler, summaries


def process_dataset(root: Path, output: Path, *, prepare: bool = False,
                    keep_raw: bool = False, chunksize: int = 20_000,
                    normal_train_ratio: float = DEFAULT_NORMAL_TRAIN_RATIO,
                    normal_gap_rows: int = 0) -> dict:
    """全量检查；可选导出。输出目录须不存在，防止混合两次处理的结果。"""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise DataError("输出目录必须放在原始 COSTE 目录之外")
    if chunksize <= 0:
        raise ValueError("chunksize 必须大于 0")
    validate_normal_split(normal_train_ratio, normal_gap_rows)
    files = discover_files(root)
    if output.exists():
        raise DataError(f"输出目录已存在：{output}；请用 --output 指定新目录")
    output.mkdir(parents=True)
    marker = output / "INCOMPLETE.txt"
    marker.write_text("处理尚未完成；仅当 report.json 已生成且本文件消失时使用结果。", encoding="utf-8")
    print("划分正常训练/校准组，仅拟合正常训练组参数……", flush=True)
    scaler, normal_summaries = fit_normal_scaler(
        files["train_faultfree"], chunksize, train_ratio=normal_train_ratio, gap_rows=normal_gap_rows,
    )
    scaler.save(output / "scaler.json")
    summaries = []
    for split in SPLITS:
        for index, path in enumerate(files[split]):
            parts: list[str] = []
            row_offset = 0
            normal_summary = normal_summaries[index] if split == "train_faultfree" else None

            def write_part(chunk: pd.DataFrame, part: int, target_split: str) -> str:
                folder = output / target_split / path.stem
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / f"part_{part:05d}.npz"
                labels = chunk.fault_label.to_numpy(dtype=np.int64)
                arrays = {"X": scaler.transform(chunk), "y_class": labels,
                          "y_detect": (labels != 0).astype(np.int8),
                          "feature_names": np.array(FEATURE_COLUMNS)}
                arrays.update({name: chunk[name].to_numpy() for name in METADATA_COLUMNS})
                if keep_raw:
                    arrays["X_raw"] = chunk.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
                np.savez_compressed(target, **arrays)
                return target.relative_to(output).as_posix()

            def export(chunk: pd.DataFrame, part: int) -> None:
                nonlocal row_offset
                if normal_summary is None:
                    parts.append(write_part(chunk, part, split))
                else:
                    for group, section in normal_group_slices(
                        chunk, row_offset, normal_summary["normal_groups"]
                    ):
                        relative_path = write_part(section, part, group["split"])
                        parts.append(relative_path)
                        group["parts"].append(relative_path)
                row_offset += len(chunk)

            if normal_summary is not None:
                summary = normal_summary
                if prepare:
                    scan_file(path, chunksize, export)
            else:
                summary = scan_file(path, chunksize, export if prepare else None)
            summary["parts"] = parts
            summaries.append(summary)
            print(f"{summary['source']}: {summary['rows']:,} 行，正常 {summary['normal_rows']:,}，"
                  f"故障 {summary['fault_rows']:,}", flush=True)
    train = [s for s in summaries if s["split"].startswith("train_")]
    test = [s for s in summaries if s["split"].startswith("test_")]
    train_seeds, test_seeds = {s["random_seed"] for s in train}, {s["random_seed"] for s in test}
    class_values = sorted({k for s in train for k in s["label_counts"]})
    test_classes = {k for s in test for k in s["label_counts"]}
    warnings = [
        "正常训练组拟合标准化参数及 CCA；正常校准组仅用于固定模型下的残差、阈值与持续报警规则校准。",
        "正常校准组参与了报警规则选择，其上虚警结果属于校准结果；使用独立测试集报告最终性能。",
        "滑窗不可跨原 CSV 或训练/隔离/校准边界；同组同 CSV 的相邻 NPZ 块需保留窗口上下文。",
        "已按用户指定排除重复 XMEAS；其余常量列 scale=1，重复/常量列需结合物理映射与 CCA 分组处理。",
        "守恒关系使用原始物理量；标准化数据用于统计模型。",
    ]
    if normal_gap_rows == 0:
        warnings.append("当前隔离行数为 0；连续轨迹的两段仍可能相关，建模时结合采样、滞后和相关性设置隔离长度。")
    if train_seeds & test_seeds:
        warnings.append(f"训练与测试存在相同随机种子，请核查独立性：{sorted(train_seeds & test_seeds)}")
    if test_classes - set(class_values):
        warnings.append(f"测试集包含训练集未见类别：{sorted(test_classes - set(class_values))}")
    report = {
        "complete": True, "prepared": prepare, "raw_in_npz": prepare and keep_raw,
        "data_root": str(root), "feature_names": list(FEATURE_COLUMNS),
        "raw_feature_count": len(RAW_FEATURE_COLUMNS), "feature_count": len(FEATURE_COLUMNS),
        "excluded_feature_names": list(EXCLUDED_FEATURE_COLUMNS), "exclusion_reason": EXCLUSION_REASON,
        "metadata_columns": list(METADATA_COLUMNS), "class_values": class_values,
        "class_to_index": {str(value): index for index, value in enumerate(class_values)},
        "detection_label": "fault_label != 0", "classification_label": "fault_label（保留原故障编号）",
        "scaler_fit_split": "normal_train", "scaler_fit_rows": scaler.n_samples,
        "normal_split": {
            "method": "chronological_per_trajectory", "train_ratio": normal_train_ratio,
            "gap_rows_per_trajectory": normal_gap_rows,
            "train_rows": sum(s["normal_groups"][0]["rows"] for s in normal_summaries),
            "calibration_rows": sum(s["normal_groups"][1]["rows"] for s in normal_summaries),
            "excluded_gap_rows": len(normal_summaries) * normal_gap_rows,
        },
        "constant_features": scaler.constant_features,
        "duplicate_feature_groups": scaler.duplicate_feature_groups,
        "total_rows": sum(s["rows"] for s in summaries), "files": summaries, "warnings": warnings,
    }
    write_json(output / "report.json", report)
    marker.unlink()
    return report


def iter_prepared_chunks(output: Path, split: str) -> Iterator[dict[str, np.ndarray]]:
    """逐块读取已导出的数据；不同 CSV 的边界以 source 字段区分。"""
    output = Path(output)
    if split == "train_faultfree":
        raise DataError("请明确选择 normal_train 或 normal_calibration，避免合并两组用于模型拟合")
    if split not in PREPARED_SPLITS:
        raise ValueError(f"未知数据分区：{split}")
    if (output / "INCOMPLETE.txt").exists():
        raise DataError("预处理未完成")
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    if not report["complete"] or not report["prepared"]:
        raise DataError("该目录仅有检查报告，请先使用 --prepare 导出数据")
    if report["feature_names"] != list(FEATURE_COLUMNS):
        raise DataError("该目录的特征名称或顺序与当前筛选规则不一致，请重新预处理")
    if report.get("scaler_fit_split") != "normal_train":
        raise DataError("该目录未按正常训练/校准分组生成，请重新预处理")
    for source in report["files"]:
        selections = source.get("normal_groups", []) if split in NORMAL_SPLITS else [source]
        for selection in selections:
            if selection["split"] != split:
                continue
            for part in selection["parts"]:
                with np.load(output / part, allow_pickle=False) as archive:
                    result = {name: archive[name] for name in archive.files}
                result["source"] = np.array(source["source"])
                result["split"] = np.array(split)
                yield result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, help="新的输出目录")
    parser.add_argument("--prepare", action="store_true", help="另外导出标准化后的分块 NPZ")
    parser.add_argument("--keep-raw", action="store_true", help="NPZ 同时保存 float64 原始物理量")
    parser.add_argument("--chunksize", type=int, default=20_000)
    parser.add_argument("--normal-train-ratio", type=float, default=DEFAULT_NORMAL_TRAIN_RATIO,
                        help="每条正常轨迹扣除隔离段后，用于训练的比例（默认 0.7）")
    parser.add_argument("--normal-gap-rows", type=int, default=0,
                        help="正常训练和校准之间排除的样本数（默认 0）")
    args = parser.parse_args()
    if args.keep_raw and not args.prepare:
        parser.error("--keep-raw 需要同时指定 --prepare")
    output = args.output or PROJECT_ROOT / "data" / (
        "COSTE_preprocessed" if args.prepare else "COSTE_audit"
    )
    try:
        report = process_dataset(args.data, output, prepare=args.prepare,
                                 keep_raw=args.keep_raw, chunksize=args.chunksize,
                                 normal_train_ratio=args.normal_train_ratio,
                                 normal_gap_rows=args.normal_gap_rows)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"处理失败：{exc}\n")
    print(f"完成：{len(report['files'])} 个文件，{report['total_rows']:,} 行；输出 {output}")


if __name__ == "__main__":
    main()

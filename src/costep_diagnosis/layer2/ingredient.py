"""拟合并验证 CoSTEP S1–S4 的进料组成比例。

从项目根目录运行：
    .venv/Scripts/python.exe -X utf8 src/costep_diagnosis/layer2/ingredient.py

以原始测量计算同一管线各组分 / 该管线全部绑定组分之和，显示为 %。
负摩尔百分含量视为无效。占比要求该管线的全部绑定组分有效，
S4 的 A/C 只检查 A、C，并要求 C 大于零。无效时刻保留为空值。
S4 另计算 A/C（无量纲）。正常训练组拟合中位数及双侧经验分位阈值；
正常校准组用冻结参数验证。NPZ 只读取 time_h、feature_names、X_raw。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data/COSTE_cca_preprocessed"
DEFAULT_TRAIN = DATA / "normal_train"
DEFAULT_CALIBRATION = DATA / "normal_calibration/faultfree_train_20261107_161555_203"
DEFAULT_GRAPH_TEMPLATE = PROJECT / "data/kg/CoSTEP_模块守恒模板.json"
DEFAULT_OUTPUT = PROJECT / "outputs/layer2_ingredient"

STREAM_COMPONENTS = {"S1": ("A", "B"), "S2": ("B", "D"),
                     "S3": ("E", "F"), "S4": ("A", "B", "C")}
DENOMINATOR_FLOOR = 1e-12


@dataclass(frozen=True)
class NormalRun:
    run_id: str
    time_h: np.ndarray
    signals: dict[str, np.ndarray]


@dataclass(frozen=True)
class Limits:
    baseline: float
    lower: float
    upper: float
    training_points: int


def load_bindings(template_path: Path) -> dict[str, dict[str, str]]:
    """Use the curated graph export for the nine physical SMEAS columns."""
    template = json.loads(template_path.read_text(encoding="utf-8"))
    bindings = {}
    for stream, components in STREAM_COMPONENTS.items():
        entries = template["streams"][stream]["signals"]["摩尔百分含量"]
        bindings[stream] = {}
        for component in components:
            columns = {
                item["id"] for item in entries
                if item["component"] == component and item["unit"] == "mol%"
                and re.fullmatch(r"SM_\d{3}", item["id"])
            }
            if len(columns) != 1:
                raise ValueError(f"{stream}.{component} 需要唯一的 SMEAS 摩尔百分含量绑定")
            bindings[stream][component] = next(iter(columns))
    return bindings


def read_normal_runs(directory: Path, columns: tuple[str, ...]) -> list[NormalRun]:
    """Read sorted prepared chunks by run; never access label arrays."""
    directory = directory.resolve()
    if not directory.is_dir():
        raise ValueError(f"数据目录不存在：{directory}")
    if not all(re.fullmatch(r"SM_\d{3}", name) for name in columns):
        raise ValueError("组成输入只接受 SMEAS 测点")
    grouped = {}
    for path in sorted(directory.rglob("part_*.npz")):
        grouped.setdefault(path.parent, []).append(path)
    if not grouped:
        raise ValueError(f"{directory} 没有 part_*.npz")

    runs = []
    for folder, files in grouped.items():
        time_parts, value_parts = [], []
        for path in files:
            with np.load(path, allow_pickle=False) as chunk:
                names = chunk["feature_names"].tolist()
                time_h = np.asarray(chunk["time_h"], dtype=float)
                raw = chunk["X_raw"]
                if (not isinstance(names, list) or len(set(names)) != len(names)
                        or time_h.ndim != 1 or raw.shape != (len(time_h), len(names))):
                    raise ValueError(f"无效的原始矩阵或时间轴：{path}")
                missing = sorted(set(columns) - set(names))
                if missing:
                    raise ValueError(f"{path} 缺少物理列：{', '.join(missing)}")
                time_parts.append(time_h)
                value_parts.append(raw[:, [names.index(name) for name in columns]].astype(float, copy=False))
        times = np.concatenate(time_parts)
        values = np.concatenate(value_parts, axis=0)
        if len(times) == 0 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"{folder} 的 time_h 应严格递增且有限")
        runs.append(NormalRun(folder.name, times, dict(zip(columns, values.T))))
    return runs


def validate_split(training: list[NormalRun], calibration: list[NormalRun]) -> None:
    """Reject overlap for segments with the same source-run name."""
    for reference in training:
        for held_out in calibration:
            if (reference.run_id == held_out.run_id
                    and max(reference.time_h[0], held_out.time_h[0])
                    <= min(reference.time_h[-1], held_out.time_h[-1])):
                raise ValueError(f"训练与校准窗口发生重叠：{reference.run_id}")


def composition_ratios(run: NormalRun, bindings: dict[str, dict[str, str]]):
    """Clean invalid input and calculate ratios without changing the time grid.

    A missing component invalidates a complete-stream fraction. The separate
    S4 A/C ratio requires only valid A and C measurements.
    """
    metrics, quality = {}, {}
    for stream, by_component in bindings.items():
        values = np.column_stack([run.signals[column] for column in by_component.values()])
        negative = (values < 0).any(axis=1)
        cleaned = np.where(np.isfinite(values) & (values >= 0), values, np.nan)
        total = cleaned.sum(axis=1)
        usable = np.isfinite(total) & (total > DENOMINATOR_FLOOR)
        quality[stream] = {
            "points": len(values),
            "invalid_component_fraction_points": int(np.count_nonzero(~usable)),
            "negative_channel_points": int(np.count_nonzero(negative)),
            "negative_channel_counts": {
                name: int(np.count_nonzero(values[:, i] < 0)) for i, name in enumerate(by_component)
            },
            "measured_sum_min_mol_pct": float(total[usable].min()) if usable.any() else None,
            "measured_sum_max_mol_pct": float(total[usable].max()) if usable.any() else None,
        }
        for i, component in enumerate(by_component):
            ratio = np.full(len(values), np.nan)
            np.divide(cleaned[:, i], total, out=ratio, where=usable)
            metrics[f"{stream}.{component}"] = 100 * ratio
        if stream == "S4":
            a, c = run.signals[by_component["A"]], run.signals[by_component["C"]]
            ratio = np.full(len(values), np.nan)
            valid = np.isfinite(a) & (a >= 0) & np.isfinite(c) & (c > DENOMINATOR_FLOOR)
            np.divide(a, c, out=ratio, where=valid)
            metrics["S4.A/C"] = ratio
            quality[stream]["invalid_ac_ratio_points"] = int(np.count_nonzero(~valid))
    return metrics, quality


def fit_limits(training_metrics: list[dict[str, np.ndarray]], coverage: float) -> dict[str, Limits]:
    """Fit one normal median and one two-sided empirical interval per metric."""
    if not math.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError("coverage 必须在 0 和 1 之间")
    tail = (1 - coverage) / 2
    limits = {}
    for key in training_metrics[0]:
        values = np.concatenate([data[key][np.isfinite(data[key])] for data in training_metrics])
        if len(values) < 2:
            raise ValueError(f"{key} 的有效训练数据不足")
        lower, baseline, upper = np.quantile(values, (tail, 0.5, 1 - tail))
        limits[key] = Limits(float(baseline), float(lower), float(upper), len(values))
    return limits


def validate_limits(calibration_metrics: list[dict[str, np.ndarray]], limits: dict[str, Limits]) -> dict:
    summary = {}
    for key, limit in limits.items():
        all_values = np.concatenate([data[key] for data in calibration_metrics])
        values = all_values[np.isfinite(all_values)]
        below = int(np.count_nonzero(values < limit.lower))
        above = int(np.count_nonzero(values > limit.upper))
        summary[key] = {
            "total_points": len(all_values), "valid_points": len(values),
            "invalid_points": len(all_values) - len(values),
            "below_lower": below, "above_upper": above,
            "coverage": 1 - (below + above) / len(values) if len(values) else None,
            "calibration_median": float(np.median(values)) if len(values) else None,
        }
    return summary


def _unit(key):
    return "ratio" if key == "S4.A/C" else "%"


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_calibration_csv(path: Path, runs, data, keys) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["run_id", "time_h", *(f"{key} ({_unit(key)})" for key in keys)])
        for run, metrics in zip(runs, data):
            for row in zip(run.time_h, *(metrics[key] for key in keys)):
                writer.writerow([run.run_id, *(float(v) if np.isfinite(v) else "" for v in row)])


def plot_calibration(output: Path, runs, data, limits, *, start_h=None, end_h=None) -> dict[str, str]:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "costep-ingredient-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lower_time = min(run.time_h[0] for run in runs) if start_h is None else start_h
    upper_time = max(run.time_h[-1] for run in runs) if end_h is None else end_h
    if not math.isfinite(lower_time) or not math.isfinite(upper_time) or lower_time >= upper_time:
        raise ValueError("绘图起点必须早于终点，两者均为小时")
    masks = [(run.time_h >= lower_time) & (run.time_h <= upper_time) for run in runs]
    if not any(mask.any() for mask in masks):
        raise ValueError("绘图时段内没有校准样本")

    paths = {}
    for stream in STREAM_COMPONENTS:
        keys = [key for key in limits if key.startswith(f"{stream}.")]
        fig, axes = plt.subplots(len(keys), 1, figsize=(13.2, 2.75 * len(keys) + 0.8),
                                  sharex=True, squeeze=False, layout="constrained")
        fig.suptitle(f"Stream {stream[1:]} | Feed composition calibration", x=0.02, ha="left",
                     fontsize=17, fontweight="bold")
        for axis, key in zip(axes[:, 0], keys):
            params = limits[key]
            axis.set_title(f"{key.split('.', 1)[1]}  |  median {params.baseline:.6g}"
                           f"   [{params.lower:.6g}, {params.upper:.6g}]", loc="left", fontsize=11)
            axis.axhspan(params.lower, params.upper, color="#4b9177", alpha=0.13)
            axis.axhline(params.baseline, color="#16654b", lw=1.15, label="Training median", zorder=3)
            axis.axhline(params.lower, color="#805c2c", ls="--", lw=0.9, label="Training bounds", zorder=3)
            axis.axhline(params.upper, color="#805c2c", ls="--", lw=0.9, zorder=3)
            for i, (run, metrics, mask) in enumerate(zip(runs, data, masks)):
                times, values = run.time_h[mask], metrics[key][mask]
                axis.plot(times, values, color="#245c92", alpha=0.7, lw=0.6,
                          label="Calibration measurement" if i == 0 else None)
                outside = (values < params.lower) | (values > params.upper)
                axis.scatter(times[outside], values[outside], color="#ac392f", s=5, alpha=0.7,
                             label="Outside bounds" if i == 0 else None, zorder=4)
            axis.set_ylabel(_unit(key))
            axis.grid(axis="y", lw=0.5, color="#dce3e8")
            axis.spines[["top", "right"]].set_visible(False)
        axes[0, 0].legend(loc="upper right", ncols=4, frameon=False, fontsize=8)
        axes[-1, 0].set_xlabel("Process time (h)")
        axes[-1, 0].set_xlim(lower_time, upper_time)
        filename = f"{stream}_calibration.png"
        fig.savefig(output / filename, dpi=155)
        plt.close(fig)
        paths[stream] = filename
    return paths


def run_experiment(train_dir: Path, calibration_dir: Path, template_path: Path,
                   output: Path, coverage: float, *, start_h=None, end_h=None) -> dict:
    bindings = load_bindings(template_path)
    columns = tuple(dict.fromkeys(column for stream in bindings.values() for column in stream.values()))
    training = read_normal_runs(train_dir, columns)
    training_outputs = [composition_ratios(run, bindings) for run in training]
    limits = fit_limits([metrics for metrics, _ in training_outputs], coverage)
    calibration = read_normal_runs(calibration_dir, columns)
    validate_split(training, calibration)
    calibration_outputs = [composition_ratios(run, bindings) for run in calibration]
    actual = [metrics for metrics, _ in calibration_outputs]

    output.mkdir(parents=True, exist_ok=True)
    plots = plot_calibration(output, calibration, actual, limits, start_h=start_h, end_h=end_h)
    write_calibration_csv(output / "calibration_timeseries.csv", calibration, actual, limits)
    _write_json(output / "parameters.json", {
        "fit_partition": "normal_train", "train_directory": str(train_dir.resolve()),
        "method": "median_and_two_sided_empirical_quantiles",
        "negative_input_policy": "invalidate_affected_metrics_without_dropping_timestamps",
        "nominal_coverage": coverage, "bindings": bindings,
        "metrics": {key: {**asdict(limit), "unit": _unit(key)} for key, limit in limits.items()},
    })
    report = {
        "train_directory": str(train_dir.resolve()), "calibration_directory": str(calibration_dir.resolve()),
        "calibration_refit": False, "nominal_coverage": coverage,
        "negative_input_policy": "invalidate_affected_metrics_without_dropping_timestamps",
        "train_runs": len(training), "calibration_runs": len(calibration),
        "calibration_windows_h": [{"run_id": run.run_id, "start": float(run.time_h[0]),
                                   "end": float(run.time_h[-1])} for run in calibration],
        "metrics": validate_limits(actual, limits),
        "train_data_quality": [{"run_id": run.run_id, "streams": quality}
                               for run, (_, quality) in zip(training, training_outputs)],
        "calibration_data_quality": [{"run_id": run.run_id, "streams": quality}
                                     for run, (_, quality) in zip(calibration, calibration_outputs)],
        "plots": plots, "timeseries": "calibration_timeseries.csv",
    }
    _write_json(output / "calibration_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--template", type=Path, default=DEFAULT_GRAPH_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--coverage", type=float, default=0.99, help="训练阈值的名义双侧覆盖率")
    parser.add_argument("--plot-start-h", type=float, help="只缩放图形显示，计算仍使用完整校准集")
    parser.add_argument("--plot-end-h", type=float, help="只缩放图形显示，计算仍使用完整校准集")
    args = parser.parse_args()
    try:
        report = run_experiment(args.train_dir, args.calibration_dir, args.template, args.output,
                                args.coverage, start_h=args.plot_start_h, end_h=args.plot_end_h)
    except (ValueError, OSError, KeyError, ImportError) as exc:
        parser.exit(2, f"{exc}\n")
    print(f"校准图、参数和报告已保存到 {args.output.resolve()}")
    for key, item in report["metrics"].items():
        observed = "无有效数据" if item["coverage"] is None else f"{item['coverage']:.2%}"
        print(f"{key:8} 覆盖率 {observed} | 下超限 {item['below_lower']:5} | 上超限 {item['above_upper']:5}")


if __name__ == "__main__":
    main()

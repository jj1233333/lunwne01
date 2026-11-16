"""Run CCA → graph binding → Layer 2 diagnosis on one CoSTEP trajectory.

Default mode reads physical NPZ data and uses the frozen CCA model. With
--events it replays an existing Layer 1 events.jsonl. The output folder has
events.jsonl, plans.jsonl, diagnoses.jsonl and summary.json. Only completed
folders without INCOMPLETE.txt are suitable for subsequent evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT / "src"))

from costep_diagnosis.layer1 import cca
from costep_diagnosis.layer2 import baseline, difference, ingredient, llm


DEFAULT_CCA_MODEL = PROJECT / "outputs/cca_run2"
MATCH_FEATURES = {
    "reactor": ("XMEAS_08", "SM_035"),
    "condenser": ("SM_046",),
    "separator": ("XMEAS_12", "SM_057"),
    "stripper": ("XMEAS_15", "SM_018", "SM_112"),
    "feed_mixer": ("SM_013", "SM_003", "SM_008", "SM_090", "SM_024"),
}


def _frozen_cca(model_path: Path, graph_sha256: str):
    if (model_path / "INCOMPLETE.txt").exists():
        raise ValueError("CCA 模型训练尚未完成")
    manifest = llm._read_json(model_path / "manifest.json")
    if (manifest.get("format_version") != cca.FORMAT_VERSION
            or manifest.get("feature_names") != list(cca.FEATURE_COLUMNS)):
        raise ValueError("CCA 模型版本或特征顺序不兼容")
    if manifest.get("graph", {}).get("sha256", "").lower() != graph_sha256.lower():
        raise ValueError("CCA 模型和第二层图谱的 dump 指纹不一致")
    consecutive = manifest.get("consecutive")
    if type(consecutive) is not int or consecutive < 1:
        raise ValueError("CCA 持续报警参数无效")
    folders = manifest["model_folders"]
    if not folders or not all(isinstance(folder, str) and Path(folder).name == folder for folder in folders):
        raise ValueError("CCA 模型子目录无效")
    scaler = cca.NormalScaler.load(model_path / "scaler.json")
    return scaler, [cca.CCAModel.load(model_path / folder) for folder in folders], consecutive


def generate_alarms(run: baseline.NormalRun, model_path: Path, graph_sha256: str,
                    variables: Iterable[str] = ()):
    """Execute the existing CCA prediction/alarm functions without an evaluation reader."""
    scaler, models, consecutive = _frozen_cca(model_path, graph_sha256)
    time = baseline._times(run.data)
    raw = np.column_stack([run.data[name] for name in cca.FEATURE_COLUMNS])
    sample_s = float(np.median(np.diff(time))) * 3600.0
    # The CCA scoring API does not use labels. Supply an empty placeholder to
    # avoid the evaluation reader, which otherwise loads label arrays.
    trajectory = cca.Trajectory(run.run_id, "", raw, (raw - scaler.mean) / scaler.scale,
                                time, 0, sample_s, np.empty(0, dtype=np.int8))
    targets = set(variables)
    alarms = []
    for model in models:
        if targets and not targets.intersection(model.spec.target_ids):
            continue
        prediction = cca.model_predictions(model, trajectory, scaler)
        scores, direction = model.score(prediction["residual"])
        for alarm in cca.alarm_events(model, trajectory, prediction, scores, direction, consecutive):
            if not targets or alarm["variable_id"] in targets:
                alarms.append(llm._alarm_fields(alarm))
    alarms.sort(key=lambda row: (row["confirmation_time_h"], row["onset_time_h"], row["variable_id"]))
    yield from enumerate(alarms, start=1)


def replay_alarms(path: Path, run: baseline.NormalRun, variables=()):
    """Check event times and, where present, the measured value against this run."""
    times = baseline._times(run.data)
    for row, stored in llm._events(path):
        alarm = llm._alarm_fields(stored)
        variable = alarm["variable_id"]
        if variables and variable not in variables:
            continue
        start, end = difference._span(alarm, times)
        if start > end:
            raise ValueError(f"报警第 {row} 行时间无效")
        if "actual" in alarm and variable in run.data:
            observed = float(run.data[variable][end])
            if not math.isclose(observed, alarm["actual"], rel_tol=1e-7, abs_tol=1e-8):
                raise ValueError(f"报警第 {row} 行的实测值与待诊断轨迹不一致")
        yield row, alarm


class _Matchers:
    """Share the loaded normal data and reuse the matcher for each feature set."""

    def __init__(self, planner, normal_root, match_features, history_s, lookback_s, step):
        self.devices = planner.devices
        self.normal_root = Path(normal_root)
        self.override = match_features
        self.options = dict(history_s=history_s, lookback_s=lookback_s, step=step)
        self.runs = None
        self.matchers = {}

    def for_key(self, key):
        features = self.override or MATCH_FEATURES.get(key, ())
        if not features:
            return None
        if features not in self.matchers:
            if self.runs is None:
                names = baseline._columns(
                    [item.column for device in self.devices.values() for item in device.inputs]
                    + list(self.override or tuple(name for cols in MATCH_FEATURES.values() for name in cols))
                )
                self.runs = [
                    run for split in baseline.NORMAL_SPLITS
                    for run in baseline.read_physical_runs(self.normal_root / split, names, split=split)
                ]
            self.matchers[features] = baseline.BaselineMatcher(self.runs, features, **self.options)
        return self.matchers[features]

    def get(self, plan):
        return self.for_key(plan["model_key"])


def _record(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def run_pipeline(*, actual_dir: Path, output: Path, events_path: Path | None = None,
                 cca_model: Path = DEFAULT_CCA_MODEL, dump: Path = llm.DEFAULT_DUMP,
                 graph_snapshot: Path = llm.DEFAULT_SNAPSHOT, template: Path = ingredient.DEFAULT_GRAPH_TEMPLATE,
                 temperature_model: Path = llm.DEFAULT_TEMPERATURE_MODEL,
                 ingredient_model: Path = llm.DEFAULT_INGREDIENT_MODEL,
                 normal_root: Path = baseline.DEFAULT_DATA_ROOT, selector_mode: str = "auto",
                 match_features=(), exclude_features=(), event_variables=(), max_events: int | None = None,
                 history_s: float = 50, lookback_s: float = 30, step: int = 1,
                 max_distance: float | None = None, include_series: bool = False,
                 upstream_stages: int = 3, rank_rule: str = "alarm-first",
                 composition: bool = True) -> dict:
    """Load one physical run, diagnose its alarms, and persist linked results."""
    actual_dir, output = actual_dir.resolve(), output.resolve()
    if output == actual_dir or actual_dir in output.parents:
        raise ValueError("输出目录须位于待诊断数据目录之外")
    if output.exists():
        raise ValueError(f"输出目录已存在：{output}")
    if max_events is not None and (type(max_events) is not int or max_events < 1):
        raise ValueError("max_events 必须为正整数")
    if (not math.isfinite(history_s) or history_s <= 0 or not math.isfinite(lookback_s)
            or lookback_s < 0 or type(step) is not int or step < 1):
        raise ValueError("history_s、lookback_s 或 step 无效")
    if max_distance is not None and (not math.isfinite(max_distance) or max_distance < 0):
        raise ValueError("max_distance 必须为非负有限数")
    match_features = baseline._columns(match_features)
    exclude_features = baseline._columns(exclude_features)
    event_variables = tuple(dict.fromkeys(event_variables))
    if any(not llm.MEASUREMENT_ID.fullmatch(name) for name in event_variables):
        raise ValueError("event_variables 必须为检测变量 ID")

    # All CCA features are raw physical channels. The loader never opens label arrays.
    features = tuple(cca.FEATURE_COLUMNS)
    runs = baseline.read_physical_runs(actual_dir, features)
    if len(runs) != 1:
        raise ValueError("actual-dir 必须指向一条待诊断轨迹的 NPZ 块目录")
    run = runs[0]
    run_key = hashlib.sha256(str(actual_dir).encode("utf-8")).hexdigest()[:16]
    graph, origin = llm.load_graph(dump, graph_snapshot)
    digest = graph["provenance"]["sha256"]
    planner = llm.DiagnosticPlanner(
        graph, graph_origin=origin, template_path=template, temperature_model=temperature_model,
        ingredient_model=ingredient_model, available_columns=features,
        selector=llm.make_model_selector(selector_mode), upstream_stages=upstream_stages,
    )
    matchers = _Matchers(planner, normal_root, match_features, history_s, lookback_s, step)
    alarms = list(generate_alarms(run, cca_model, digest, event_variables) if events_path is None
                  else replay_alarms(events_path, run, event_variables))
    truncated = max_events is not None and len(alarms) > max_events
    if truncated:
        alarms = alarms[:max_events]
    # A borrowed upstream model only counts as evidence when its own target alarmed.
    alarm_variables = {row["variable_id"] for _, row in alarms}

    output.mkdir(parents=True, exist_ok=False)
    marker = output / "INCOMPLETE.txt"
    marker.write_text("流程完成后将移除此文件。\n", encoding="utf-8")
    counts, templates, reasons = Counter(), Counter(), Counter()
    upstream_used, upstream_skipped = Counter(), Counter()
    total = 0
    composition_pools, composition_exceeded = Counter(), Counter()
    with (output / "events.jsonl").open("x", encoding="utf-8") as event_file, \
         (output / "plans.jsonl").open("x", encoding="utf-8") as plan_file, \
         (output / "diagnoses.jsonl").open("x", encoding="utf-8") as diagnosis_file:
        for upstream_row, alarm in alarms:
            total += 1
            identity = {"run_id": run_key, "event_row": total, "upstream_event_row": upstream_row}
            _record(event_file, {**identity, **alarm})
            plan = {**planner.plan_alarm(alarm), **identity}
            _record(plan_file, plan)
            matcher = (matchers.get(plan) if plan["status"] == "ready" and plan["template_id"] == "TEMPERATURE"
                       else None)
            diagnosis = difference.diagnose(
                plan, run, matcher=matcher, graph=planner.graph, temperature_system=planner.temperature_system,
                max_distance=max_distance, exclude_features=exclude_features, include_series=include_series,
                matcher_for=matchers.for_key, alarm_variables=alarm_variables, rank_rule=rank_rule,
                use_composition=composition,
            )
            _record(diagnosis_file, {**diagnosis, **identity, "alarm": alarm})
            counts[diagnosis["status"]] += 1
            templates[plan["template_id"] or "UNSUPPORTED"] += 1
            if diagnosis.get("reason"):
                reasons[diagnosis["reason"]] += 1
            pool = diagnosis.get("candidate_pool") or {}
            for item in pool.get("models") or []:
                if item.get("upstream"):
                    upstream_used[item["model_key"]] += 1
            for item in pool.get("skipped_models") or []:
                upstream_skipped[item.get("reason")] += 1
            for item in pool.get("composition_models") or []:
                composition_pools[item.get("scope_id")] += 1
                for name in item.get("exceeded") or []:
                    composition_exceeded[name] += 1
    summary = {
        "complete": True, "run_id": run_key, "actual_dir": str(actual_dir),
        "event_source": "frozen_cca" if events_path is None else "existing_cca_events",
        "cca_model": str(cca_model.resolve()) if events_path is None else None,
        "events_input": str(events_path.resolve()) if events_path is not None else None,
        "alarm_source_row_basis": "zero_based_npz_segment" if events_path is None else "source_event_convention",
        "graph_sha256": digest, "event_count": total, "truncated": truncated, "event_variables": event_variables,
        "by_status": dict(counts), "by_template": dict(templates), "reasons": dict(reasons),
        "upstream_candidates": {"stages": upstream_stages, "alarm_pool": len(alarm_variables),
                                "models_used": dict(upstream_used),
                                "skipped": dict(upstream_skipped), "rank_rule": rank_rule,
                                "composition": composition,
                                "composition_pools": dict(composition_pools),
                                "composition_exceeded": dict(composition_exceeded)},
        "normal_matching": {"history_s": history_s, "lookback_s": lookback_s, "step": step,
                            "max_distance": max_distance, "match_features": match_features or MATCH_FEATURES,
                            "exclude_features": exclude_features},
        "files": {"events": "events.jsonl", "plans": "plans.jsonl", "diagnoses": "diagnoses.jsonl"},
    }
    cca.write_json(output / "summary.json", summary)
    marker.unlink()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-dir", type=Path, required=True, help="单条待诊断轨迹的 NPZ 块目录")
    parser.add_argument("--output", type=Path, required=True, help="新建本次诊断输出目录")
    parser.add_argument("--events", type=Path, help="可选，从已生成的 CCA events.jsonl 开始")
    parser.add_argument("--cca-model", type=Path, default=DEFAULT_CCA_MODEL)
    parser.add_argument("--dump", type=Path, default=llm.DEFAULT_DUMP)
    parser.add_argument("--graph-snapshot", type=Path, default=llm.DEFAULT_SNAPSHOT)
    parser.add_argument("--template", type=Path, default=ingredient.DEFAULT_GRAPH_TEMPLATE)
    parser.add_argument("--temperature-model", type=Path, default=llm.DEFAULT_TEMPERATURE_MODEL)
    parser.add_argument("--ingredient-model", type=Path, default=llm.DEFAULT_INGREDIENT_MODEL)
    parser.add_argument("--normal-root", type=Path, default=baseline.DEFAULT_DATA_ROOT)
    parser.add_argument("--selector", choices=("auto", "rules", "llm"), default="auto")
    parser.add_argument("--match-features", nargs="+", default=())
    parser.add_argument("--exclude-features", nargs="+", default=())
    parser.add_argument("--event-variables", nargs="+", default=(), help="仅处理指定的报警测点")
    parser.add_argument("--max-events", type=int, help="调试时最多处理多少条报警")
    parser.add_argument("--history-s", type=float, default=50)
    parser.add_argument("--lookback-s", type=float, default=30)
    parser.add_argument("--step", type=int, default=1, help="正常参考滑窗步幅，单位为采样点")
    parser.add_argument("--max-distance", type=float)
    parser.add_argument("--include-series", action="store_true")
    parser.add_argument("--upstream-stages", type=int, default=3,
                        help="第二层候选池沿工艺流反向回溯的设备级跳数；0 表示只用报警点所属设备")
    parser.add_argument("--rank-rule", choices=("alarm-first", "dj"), default="alarm-first",
                        help="温度候选排序主键：alarm-first 先看候选自身是否也报警，dj 只按替换敏感度")
    args = vars(parser.parse_args())
    args["events_path"] = args.pop("events")
    args["selector_mode"] = args.pop("selector")
    try:
        summary = run_pipeline(**args)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"诊断流程失败：{exc}\n")
    print(f"诊断完成：{summary['event_count']} 条报警，结果保存到 {args['output'].resolve()}")
    print(json.dumps(summary["by_status"], ensure_ascii=False))


if __name__ == "__main__":
    main()

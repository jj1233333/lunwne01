"""Execute a graph-bound diagnosis on a single CoSTEP physical trajectory.

INGREDIENT uses the frozen lower/upper thresholds on the actual ratios.
TEMPERATURE uses a fixed calibrated recursive window ending at CCA confirmation.
The observed initial temperature and the actual prefix preceding each replacement
are retained. Per-point scores follow the D_j and S_j equations. The aggregate
is the arithmetic mean on the part of the prediction window from CCA onset
through confirmation. Confirmation-point scores are also returned.

A temperature plan may carry `upstream_plans`; each one is scored with its own
ARX model when that device's boundary actually moved (its target or one of its
watched inputs alarmed). Candidates from every pool are merged, ranked by alarm
membership first and by D_j second, and annotated with model_key / scope_depth.

It may also carry `composition_plans`, the trained feed streams reaching the
device. Their frozen ratio metrics are scored in place and merged into the same
candidate list, so a root cause named S4.A or S4.A/C is reachable from a
temperature row. A metric beyond its frozen band joins the first rank tier.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Mapping
import json
import math
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT / "src"))

from costep_diagnosis.layer2 import baseline, ingredient, llm, temperature


def _columns(plan, candidate_columns=None):
    slots = plan["slot_bindings"]
    if plan["template_id"] == "INGREDIENT":
        scope = plan["scope_id"]
        expected = ingredient.STREAM_COMPONENTS.get(scope)
        group = slots["components"]
        if not expected or set(group) != set(expected):
            raise ValueError(f"{scope} 的组成测点绑定不完整")
        return baseline._columns(group[name]["column"] for name in expected)
    inputs = slots["inputs"]
    selected = None if candidate_columns is None else set(baseline._columns(candidate_columns))
    columns = [slots["target"]["column"], *(item["column"] for item in inputs)]
    for item in inputs:
        reference = item.get("composition_reference")
        if reference and (selected is None or item["column"] in selected):
            columns.append(reference["column"])
    return baseline._columns(columns)


def _window_data(actual, columns, times, start, end):
    section = slice(start, end + 1)
    matrix = baseline._slice_values(actual, columns, section, len(times))
    return {"time_h": times[section], **dict(zip(columns, matrix.T))}


def _span(alarm, times):
    onset = float(alarm["onset_time_h"])
    confirm = float(alarm["confirmation_time_h"])
    first, last = baseline._time_index(times, onset), baseline._time_index(times, confirm)
    if first > last:
        raise ValueError("报警确认时刻早于起点")
    return first, last


def _json_series(values):
    return [float(x) if math.isfinite(x) else None for x in values]


def _metric_limits(metric, unit):
    params = metric["parameters"]
    values = [params.get(key) for key in ("lower", "baseline", "upper")]
    if (metric["unit"] != unit or params.get("unit") != unit
            or not all(type(value) in (int, float) and math.isfinite(value) for value in values)
            or values != sorted(values)):
        raise ValueError(f"{metric['metric_id']} 的冻结阈值或单位无效")
    return values


def _ingredient(plan, actual, times, first, last, *, include_series):
    scope = plan["scope_id"]
    group = plan["slot_bindings"]["components"]
    names = _columns(plan)
    parts = ingredient.STREAM_COMPONENTS[scope]
    data = _window_data(actual, names, times, first, last)
    if len(names) != len(parts) or any(group[part]["unit"] != "mol%" for part in parts):
        raise ValueError(f"{scope} 的物理组成测点重复或单位错误")
    mapping = dict(zip(parts, names))
    run = ingredient.NormalRun("diagnosis_window", data["time_h"], {col: data[col] for col in names})
    ratios, quality = ingredient.composition_ratios(run, {scope: mapping})
    metrics = {}
    for metric in plan["metrics"]:
        key = metric["metric_id"]
        if key not in ratios or key in metrics:
            raise ValueError(f"不支持或重复的组成指标：{key}")
        is_ac = key == "S4.A/C"
        part = key.split(".", 1)[1]
        expected_num = mapping["A"] if is_ac else mapping[part]
        expected_den = [mapping["C"]] if is_ac else list(mapping.values())
        operation = "ratio" if is_ac else "percent_of_stream_total"
        if (metric["numerator"] != expected_num or metric["denominator_columns"] != expected_den
                or metric["operation"] != operation):
            raise ValueError(f"{key} 的公式槽位与成分模板不一致")
        lower, center, upper = _metric_limits(metric, "ratio" if is_ac else "%")
        values = ratios[key]
        valid = np.isfinite(values)
        below, above = valid & (values < lower), valid & (values > upper)
        outside = below | above
        n_valid = int(np.count_nonzero(valid))
        row = {
            "unit": metric["unit"], "baseline": center, "lower": lower, "upper": upper,
            "numerator": expected_num, "denominator_columns": expected_den,
            "valid_points": n_valid, "invalid_points": len(values) - n_valid,
            "below_lower": int(np.count_nonzero(below)), "above_upper": int(np.count_nonzero(above)),
            "status": ("exceeded" if outside.any() else "within_threshold" if n_valid else "no_valid_values"),
            "actual_at_confirmation": float(values[-1]) if valid[-1] else None,
            "min_actual": float(values[valid].min()) if n_valid else None,
            "max_actual": float(values[valid].max()) if n_valid else None,
            "first_exceedance_time_h": float(data["time_h"][np.flatnonzero(outside)[0]]) if outside.any() else None,
            "max_exceedance": (float(np.maximum(lower - values[valid], values[valid] - upper).clip(0).max())
                               if n_valid else None),
            "graph_evidence": [edge for p in parts for edge in group[p].get("evidence", [])],
        }
        if include_series:
            row["actual"] = _json_series(values)
            row["residual_from_baseline"] = _json_series(values - center)
        metrics[key] = row
    if set(metrics) != set(ratios):
        raise ValueError("诊断方案缺少已训练的组成指标")
    valid = any(item["valid_points"] for item in metrics.values())
    result = dict(status="ready" if valid else "blocked", reason=None if valid else "no_valid_ratios",
                  evaluation="direct_threshold_comparison", window_h=(float(times[first]), float(times[last])),
                  metrics=metrics, exceeded_metrics=[key for key, row in metrics.items() if row["status"] == "exceeded"],
                  data_quality=quality[scope])
    if include_series:
        result["time_h"] = data["time_h"].tolist()
    return result


def _temperature_model(plan, system):
    if system is None:
        system = temperature.TemperatureSystem.load(Path(plan["model_path"]))
    key = plan["model_key"]
    device, params = system.devices[key], system.parameters[key]
    slots = plan["slot_bindings"]
    if device.target != slots["target"]["column"]:
        raise ValueError("方案中的目标温度与已训练模型不一致")
    declared = [(item["column"], (item.get("composition_reference") or {}).get("column"))
                for item in slots["inputs"]]
    if declared != [(spec.column, spec.reference_column) for spec in device.inputs]:
        raise ValueError("方案中的输入顺序或组分参考列与已训练模型不一致")
    calibration = params.calibration
    if not calibration:
        raise ValueError("温度模型缺少正常残差校准参数")
    center, scale, limit = (float(calibration[name]) for name in ("center_c", "scale_c", "limit_c"))
    if not all(map(math.isfinite, (center, scale, limit))) or scale <= 0 or limit < 0:
        raise ValueError("正常残差中心、尺度或阈值无效")
    return system, key, device, params, center, scale, limit


def _series(prediction, key):
    return np.ravel(prediction[key])


def _candidate_path(slot, target, graph):
    """Search only along graph input/output relationships in their declared direction."""
    starts, goals = slot.get("locations", []), set(target.get("locations", []))
    evidence = slot.get("evidence", [])
    for site in starts:
        if site in goals:
            return {"status": "same_location", "edges": evidence + target.get("evidence", [])}
    if graph is not None:
        queue = deque((site, []) for site in starts)
        visited = set(starts)
        while queue:
            site, path = queue.popleft()
            for relation in ("输入", "输出"):
                for neighbor in sorted(graph.outgoing[site, relation]):
                    if neighbor in visited:
                        continue
                    step = [*path, [site, relation, neighbor]]
                    if neighbor in goals:
                        return {"status": "directed_path", "edges": evidence + step + target.get("evidence", [])}
                    visited.add(neighbor)
                    queue.append((neighbor, step))
    return {"status": "not_verified", "edges": evidence}


def _replace_input(data, aligned, spec):
    """Replace the aligned suffix while leaving model warmup and actual initial T intact."""
    times, ref_times = data["time_h"], aligned["time_h"]
    start = max(times[0], ref_times[0])
    data_idx = int(np.searchsorted(times, start))
    ref_idx = int(np.searchsorted(ref_times, start))
    if not np.array_equal(times[data_idx:], ref_times[ref_idx:]):
        raise ValueError("匹配正常轨迹与递推窗口未对齐")
    normal = aligned[spec.column][ref_idx:]
    observed = data[spec.column][data_idx:]
    replaced = {**data, spec.column: np.r_[data[spec.column][:data_idx], normal]}
    columns = [spec.column]
    if spec.reference_column:
        ref = spec.reference_column
        correction = data[ref][data_idx:] + observed - normal
        if (not np.isfinite(correction).all() or np.any((normal < 0) | (normal > 100))
                or np.any((correction < 0) | (correction > 100))):
            return None, None
        replaced[ref] = np.r_[data[ref][:data_idx], correction]
        columns.append(ref)
    deviation = observed - normal
    evidence = dict(columns=columns, window_h=(float(start), float(times[-1])),
                    actual_at_confirmation=float(observed[-1]), normal_at_confirmation=float(normal[-1]),
                    mean_signed_deviation=float(deviation.mean()), mean_absolute_deviation=float(abs(deviation).mean()),
                    maximum_absolute_deviation=float(abs(deviation).max()))
    return replaced, evidence


def _temperature_one(plan, actual, times, first, last, *, matcher, system, graph,
                     candidate_columns, max_distance, cutoff_time_h, exclude_features, include_series,
                     scope_depth=0, upstream=False):
    """Score every watched input of one ARX device by the leave-one-input-out rule."""
    system, key, device, params, center, scale, limit = _temperature_model(plan, system)
    horizon = temperature.steps(system.horizon_s, params.sample_s, "horizon_s")
    prediction_start = last - horizon
    context_start = prediction_start - max(params.delays_steps, default=0)
    if context_start < 0:
        return {"status": "blocked", "reason": "insufficient_history_for_calibrated_horizon"}
    physical = actual.data if isinstance(actual, baseline.NormalRun) else actual
    data = _window_data(physical, _columns(plan, candidate_columns), times, context_start, last)
    original = system.predict(key, data)
    grid = _series(original, "time_h")
    r0 = _series(original, "residual_c")
    keep = grid >= times[first]
    observed = _series(original, "observed_c")
    reference = {"residual_center_c": center, "residual_scale_c": scale, "residual_limit_c": limit,
                 "initial_temperature_c": float(original["initial_temperature_c"][0]),
                 "observed_at_confirmation_c": float(observed[-1]),
                 "predicted_at_confirmation_c": float(_series(original, "predicted_c")[-1]),
                 "residual_at_confirmation_c": float(r0[-1]),
                 "threshold_exceedance_points": int(np.count_nonzero(abs(r0[keep] - center) > limit))}
    if include_series:
        reference.update(time_h=grid[keep].tolist(), observed_c=observed[keep].tolist(),
                         predicted_c=_series(original, "predicted_c")[keep].tolist(), r0_c=r0[keep].tolist())
    result = dict(status="ready", reason=None, model_key=key, scope_depth=scope_depth, upstream=upstream,
                  prediction_window_h=(float(times[prediction_start]), float(times[last])),
                  score_window_h=(float(grid[keep][0]), float(grid[keep][-1])),
                  score_aggregation="mean_of_pointwise_scores", actual_temperature=reference,
                  independent_temperature=device.independent_temperature, candidates=[])
    if matcher is None:
        return {**result, "status": "partial", "reason": "missing_baseline_matcher"}
    chosen = set(spec.column for spec in device.inputs) if candidate_columns is None else set(baseline._columns(candidate_columns))
    if not chosen or chosen - {spec.column for spec in device.inputs}:
        raise ValueError("指定的候选变量不属于当前温度模型输入")
    aliases = [signal for source, _, target in plan.get("alias_evidence", ()) for signal in (source, target)]
    common_exclusions = baseline._columns((*baseline._columns(exclude_features), *aliases))
    for slot, spec in zip(plan["slot_bindings"]["inputs"], device.inputs):
        if spec.column not in chosen:
            continue
        excluded = common_exclusions + ((spec.reference_column,) if spec.reference_column else ())
        match = matcher.match(actual, plan, spec.column, cutoff_time_h=cutoff_time_h,
                              exclude_features=excluded, max_distance=max_distance,
                              nonnegative_candidates=(spec.column,) if spec.reference_column else ())
        row = dict(candidate=spec.column, slot_name=spec.label, input_unit=spec.unit, status="blocked", reason=None,
                   template_id="TEMPERATURE", exceeded=False, model_key=key, scope_depth=scope_depth,
                   upstream=upstream,
                   baseline={k: v for k, v in match.items() if k != "trajectory"},
                   graph_path=_candidate_path(slot, plan["slot_bindings"]["target"], graph))
        result["candidates"].append(row)
        if match["status"] != "ready":
            row["reason"] = match["reason"]
            continue
        replacement, displacement = _replace_input(data, match["trajectory"], spec)
        if replacement is None:
            row["reason"] = "invalid_composition_after_replacement"
            continue
        changed = system.predict(key, replacement, observed_data=data)
        if not np.array_equal(grid, _series(changed, "time_h")):
            raise ValueError("替换前后的预测时间网格不一致")
        rj = _series(changed, "residual_c")
        d = abs(rj - r0) / scale
        s = abs(rj - center) - abs(r0 - center)
        row.update(status="ready", candidate_deviation=displacement, D_j=float(d[keep].mean()),
                   S_j_c=float(s[keep].mean()), D_at_confirmation=float(d[-1]), S_at_confirmation_c=float(s[-1]),
                   replaced_prediction_at_confirmation_c=float(_series(changed, "predicted_c")[-1]),
                   replaced_residual_at_confirmation_c=float(rj[-1]))
        if include_series:
            row.update(rj_c=rj[keep].tolist(), D_t=d[keep].tolist(), S_t_c=s[keep].tolist())
    return result


def _composition_rows(plan, actual, times, first, last, *, include_series):
    """Score one frozen composition plan into candidate rows.

    The ingredient template judges a ratio against empirical normal quantiles, so
    there is no counterfactual to replace. The analogue of D_j is the largest
    window offset from the training median measured in half-bandwidths: a value
    of one means the metric touched a frozen limit during the window.
    """
    physical = actual.data if isinstance(actual, baseline.NormalRun) else actual
    detail = _ingredient(plan, physical, times, first, last, include_series=include_series)
    scope, model_key = plan["scope_id"], f"ingredient:{plan['scope_id']}"
    rows = []
    for key, item in detail.get("metrics", {}).items():
        base, lower, upper = item["baseline"], item["lower"], item["upper"]
        half = (upper - lower) / 2 if upper > lower else None
        peaks = [abs(value - base) for value in (item["min_actual"], item["max_actual"]) if value is not None]
        score = max(peaks) / half if half and peaks else None
        row = {
            "candidate": key, "template_id": "INGREDIENT", "model_key": model_key,
            "scope_id": scope, "scope_depth": plan.get("scope_depth", 0), "upstream": True,
            "status": "ready" if item["valid_points"] else "blocked",
            "reason": None if item["valid_points"] else "no_valid_values",
            "exceeded": item["status"] == "exceeded", "unit": item["unit"],
            "baseline": base, "lower": lower, "upper": upper,
            "actual_at_confirmation": item["actual_at_confirmation"],
            "min_actual": item["min_actual"], "max_actual": item["max_actual"],
            "max_exceedance": item["max_exceedance"],
            "D_j": score, "S_j_c": None, "graph_path": plan.get("graph_path"),
        }
        rows.append(row)
    return rows, detail


def _rank_key(item, alarming, rank_rule="alarm-first"):
    """Order one candidate: hard evidence first, then replacement sensitivity.

    A root cause that is itself measured and disturbed should already alarm, so
    alarm membership outranks D_j. A composition metric beyond its frozen band is
    the same kind of evidence — a threshold verdict rather than a counterfactual —
    so it shares the first tier instead of sitting behind every temperature
    candidate.
    """
    member = rank_rule == "alarm-first" and bool(alarming) and item["candidate"] in alarming
    charged = 0 if (member or item.get("exceeded")) else 1
    return (item["status"] != "ready", charged, -(item.get("D_j") or 0.0),
            item.get("scope_depth", 0), item["candidate"])


def _merge_candidates(rows):
    """Keep one row per column, preferring the model that explains it best."""
    merged, position = [], {}
    for row in rows:
        name = row.get("candidate")
        if name not in position:
            position[name] = len(merged)
            merged.append(row)
            continue
        current = merged[position[name]]
        better = (row.get("status") == "ready" and current.get("status") != "ready") or (
            row.get("status") == current.get("status")
            and (row.get("D_j") if row.get("D_j") is not None else -1.0)
            > (current.get("D_j") if current.get("D_j") is not None else -1.0))
        kept, dropped = (row, current) if better else (current, row)
        kept["also_scored_by"] = sorted({*(kept.get("also_scored_by") or []), dropped.get("model_key")} - {None})
        merged[position[name]] = kept
    return merged


def _temperature(plan, actual, times, first, last, *, matcher, system, graph,
                 candidate_columns, max_distance, cutoff_time_h, exclude_features, include_series,
                 matcher_for=None, alarm_variables=(), rank_rule="alarm-first", use_composition=True):
    local = _temperature_one(plan, actual, times, first, last, matcher=matcher, system=system, graph=graph,
                             candidate_columns=candidate_columns, max_distance=max_distance,
                             cutoff_time_h=cutoff_time_h, exclude_features=exclude_features,
                             include_series=include_series)
    if local["status"] == "blocked":
        return local
    rows = list(local["candidates"])
    used = [{"model_key": local["model_key"], "scope_depth": 0, "upstream": False,
             "candidates": len(local["candidates"])}]
    skipped = []
    for sub in plan.get("upstream_plans") or []:
        target = (sub.get("slot_bindings", {}).get("target") or {}).get("column")
        if sub.get("status") != "ready":
            skipped.append({"model_key": sub.get("model_key"), "target": target,
                            "reason": sub.get("reason") or "plan_not_ready"})
            continue
        # A borrowed model is only evidence when that device's own boundary moved:
        # its target alarmed, or at least one of its watched inputs alarmed.
        inputs = {item["column"] for item in sub.get("slot_bindings", {}).get("inputs", [])}
        disturbed = not alarm_variables or target in alarm_variables or bool(inputs & alarm_variables)
        if not disturbed:
            skipped.append({"model_key": sub.get("model_key"), "target": target,
                            "reason": "upstream_device_not_disturbed"})
            continue
        sub_matcher = matcher_for(sub["model_key"]) if matcher_for is not None else matcher
        if sub_matcher is None:
            skipped.append({"model_key": sub.get("model_key"), "target": target,
                            "reason": "missing_baseline_matcher"})
            continue
        extra = _temperature_one({**sub, "alarm": plan["alarm"]}, actual, times, first, last, matcher=sub_matcher,
                                 system=system, graph=graph, candidate_columns=None, max_distance=max_distance,
                                 cutoff_time_h=cutoff_time_h, exclude_features=exclude_features,
                                 include_series=include_series, scope_depth=sub.get("scope_depth", 1), upstream=True)
        used.append({"model_key": sub["model_key"], "target": target, "scope_depth": sub.get("scope_depth", 1),
                     "upstream": True, "site": sub.get("upstream_site"), "candidates": len(extra["candidates"])})
        rows.extend(extra["candidates"])
    composition_models = []
    if use_composition:
        for sub in plan.get("composition_plans") or []:
            entry = {"scope_id": sub.get("scope_id"), "scope_depth": sub.get("scope_depth", 0),
                     "site": sub.get("upstream_site"), "status": "ready", "metrics": 0, "exceeded": []}
            try:
                extra, detail = _composition_rows(sub, actual, times, first, last,
                                                  include_series=include_series)
            except (KeyError, ValueError) as error:
                composition_models.append({**entry, "status": "blocked", "reason": str(error)})
                continue
            entry.update(status=detail.get("status"), metrics=len(extra),
                         exceeded=[row["candidate"] for row in extra if row["exceeded"]])
            composition_models.append(entry)
            rows.extend(extra)
    result = {**local, "candidates": _merge_candidates(rows)}
    alarming = set(alarm_variables)

    result["candidates"].sort(key=lambda item: _rank_key(item, alarming, rank_rule))
    ranked = [item for item in result["candidates"] if item["status"] == "ready"]
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    result["candidate_pool"] = {"local_model": local["model_key"], "models": used,
                                "skipped_models": skipped, "alarm_variable_count": len(alarm_variables),
                                "rank_rule": rank_rule, "composition_models": composition_models}
    if not ranked:
        result.update(status="partial", reason=local.get("reason") or "no_scorable_replacements")
    else:
        result.update(status="ready", reason=None)
    return result


def diagnose(plan: Mapping, actual, *, matcher: baseline.BaselineMatcher | None = None,
             temperature_system=None, graph: llm.GraphIndex | None = None, candidate_columns=None,
             max_distance: float | None = None, cutoff_time_h: float | None = None,
             exclude_features=(), include_series: bool = False, matcher_for=None,
             alarm_variables=(), rank_rule: str = "alarm-first", use_composition: bool = True) -> dict:
    """Execute one llm.py plan, using physical columns and no evaluation labels."""
    result = {"plan_id": plan.get("plan_id"), "template_id": plan.get("template_id"),
              "scope_id": plan.get("scope_id"), "event_row": plan.get("event_row"),
              "graph_evidence": plan.get("graph_evidence", [])}
    if plan.get("status") != "ready":
        return {**result, "status": "blocked", "reason": plan.get("reason") or "plan_not_ready",
                "missing_slots": plan.get("missing_slots", [])}
    alarm = llm._alarm_fields(plan["alarm"])
    physical = actual.data if isinstance(actual, baseline.NormalRun) else actual
    try:
        times = baseline._times(physical)
        first, last = _span(alarm, times)
        result["alarm"] = alarm
        if plan["template_id"] == "INGREDIENT":
            details = _ingredient(plan, physical, times, first, last, include_series=include_series)
        elif plan["template_id"] == "TEMPERATURE":
            details = _temperature(plan, actual, times, first, last, matcher=matcher, system=temperature_system,
                                   graph=graph, candidate_columns=candidate_columns, cutoff_time_h=cutoff_time_h,
                                   max_distance=max_distance, exclude_features=exclude_features,
                                   include_series=include_series, matcher_for=matcher_for,
                                   alarm_variables=set(alarm_variables), rank_rule=rank_rule,
                                   use_composition=use_composition)
        else:
            return {**result, "status": "blocked", "reason": "unsupported_template"}
    except KeyError as error:
        return {**result, "status": "blocked", "reason": "missing_diagnostic_input",
                "missing_slots": [str(error.args[0])]}
    except temperature.TemperatureError as error:
        return {**result, "status": "blocked", "reason": "temperature_window_or_model_invalid",
                "detail": str(error)}
    return {**result, **details}


def _read_plan(path, event_row):
    if path.suffix.lower() == ".json":
        plan = llm._read_json(path)
        if event_row is not None and plan.get("event_row") != event_row:
            raise ValueError(f"诊断方案没有事件行 {event_row}")
        return plan
    for line, plan in llm._events(path):
        if event_row is None or plan.get("event_row", line) == event_row:
            return plan
    raise ValueError("方案文件中没有指定事件")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="llm.py 生成的 JSON 或 JSONL 方案")
    parser.add_argument("--event-row", type=int, help="对应 llm.py 输出的 event_row，默认选择第一条方案")
    parser.add_argument("--actual-dir", type=Path, required=True, help="单条待诊断轨迹的 NPZ 块目录")
    parser.add_argument("--normal-root", type=Path, default=baseline.DEFAULT_DATA_ROOT)
    parser.add_argument("--match-features", nargs="+", help="温度替换用的异常前工况测点")
    parser.add_argument("--candidates", nargs="+", help="可选，只替换指定温度输入")
    parser.add_argument("--exclude-features", nargs="+", default=())
    parser.add_argument("--history-s", type=float, default=50.0)
    parser.add_argument("--lookback-s", type=float, default=30.0)
    parser.add_argument("--max-distance", type=float)
    parser.add_argument("--dump", type=Path, default=llm.DEFAULT_DUMP)
    parser.add_argument("--graph-snapshot", type=Path, default=llm.DEFAULT_SNAPSHOT)
    parser.add_argument("--include-series", action="store_true")
    parser.add_argument("--ignore-upstream", action="store_true", help="只诊断报警点所属设备，忽略上游候选池")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        plan = _read_plan(args.plan, args.event_row)
        if plan["status"] != "ready":
            result = diagnose(plan, None)
        else:
            if args.ignore_upstream:
                plan = {key: value for key, value in plan.items() if key != "upstream_plans"}
            is_temperature = plan["template_id"] == "TEMPERATURE"
            columns = _columns(plan, args.candidates)
            for sub in plan.get("upstream_plans") or []:
                columns = tuple(dict.fromkeys((*columns, *_columns(sub))))
            features = baseline._columns(args.match_features or ()) if is_temperature else ()
            runs = baseline.read_physical_runs(args.actual_dir, (*columns, *features))
            if len(runs) != 1:
                raise ValueError("actual-dir 必须只包含一条待诊断轨迹")
            matcher, graph = None, None
            if is_temperature and features:
                candidates = args.candidates or [item["column"] for item in plan["slot_bindings"]["inputs"]]
                matcher = baseline.BaselineMatcher.from_preprocessed(
                    match_features=features, candidate_columns=columns, data_root=args.normal_root,
                    history_s=args.history_s, lookback_s=args.lookback_s,
                )
                indexed, _ = llm.load_graph(args.dump, args.graph_snapshot)
                expected = plan.get("graph", {}).get("sha256")
                if expected and indexed.get("provenance", {}).get("sha256") != expected:
                    raise ValueError("传播路径图谱与诊断方案的图谱版本不一致")
                graph = llm.GraphIndex(indexed)
            result = diagnose(plan, runs[0], matcher=matcher, graph=graph, candidate_columns=args.candidates,
                              max_distance=args.max_distance, exclude_features=args.exclude_features,
                              include_series=args.include_series)
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as file:
                file.write(payload)
        else:
            sys.stdout.write(payload)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()

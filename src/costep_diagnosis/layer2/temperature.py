"""一阶多输入 ARX 温度模型，训练/校准及候选替换共用同一递推方程。

项目根目录运行：
    python -X utf8 src/costep_diagnosis/layer2/__pycache__/temperature.py train
    python -X utf8 src/costep_diagnosis/layer2/__pycache__/temperature.py validate

T_hat[k+1] = T0 + c + a*(T_hat[k]-T0) + sum(bj*(uj[k-dj]-uj0))
0<a<1；tau_s=-sample_s/log(a)，Kj=bj/(1-a)，theta_j_s=dj*sample_s。
T/T0/c/残差单位为 ℃，b/K 的单位为 ℃/输入单位，组成使用原始 mol%（百分点）。
训练使用实测历史温度做有界岭回归；模型选择、校准、绘图使用窗口递推，
仅窗口初值使用实测温度。指标排除初值，报告同时包含单步误差和保持初温基线。

正常训练内部按时间先后 70/30 留出选择 ridge 和延迟，随后用全训练集重拟合。
默认共享延迟是少参数假设，各输入仍保存独立的 d/theta 字段；物理已知的独立
延迟可通过 train --delays-json 指定 {设备key: {输入列: 延迟秒数}}，不使用校准集选参。
校准仅估计递推残差的中位中心与 |r-center| 的分位阈值，不改变模型系数。
实际记录输入用于事后窗口分析；在线未来预测需另行提供未来输入假设。
闭环正常数据得到的有效增益不自动表示因果效应。tau/K 是确定性响应的换算值，
共享时间常数、充分激励和局部工况假设需要验证。报告保留边界命中及弱激励警告。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Mapping

import numpy as np

PROJECT = next(p for p in Path(__file__).resolve().parents
               if (p / "src/costep_diagnosis/data_convert/data.py").is_file())
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))
from costep_diagnosis.data_convert.data import iter_prepared_chunks

DATA = PROJECT / "data/COSTE_cca_preprocessed"
OUTPUT = PROJECT / "outputs/layer2_temperature"
TEMPLATE = PROJECT / "data/kg/CoSTEP_模块守恒模板.json"
VERSION = 1


class TemperatureError(ValueError):
    """测点、时间轴、分区或参数不满足温度模型要求。"""


def number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TemperatureError(f"{name} 应为有限数")
    value = float(value)
    if not math.isfinite(value) or positive and value <= 0:
        raise TemperatureError(f"{name} 数值范围无效")
    return value


def column(data, name):
    if not re.fullmatch(r"(?:XMEAS_\d{2}|XMV_\d{2}|SM_\d{3}|time_h)", name):
        raise TemperatureError(f"只允许物理测点和 time_h，禁止标签作为输入: {name}")
    values = np.asarray(data[name], dtype=float)
    if values.ndim != 1:
        raise TemperatureError(f"{name} 应为一维数组")
    return values


def times(data):
    t = column(data, "time_h")
    if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise TemperatureError("time_h 应为至少两个严格递增的有限时刻")
    return t


@dataclass
class NormalRun:
    source: str
    split: str
    data: Mapping[str, np.ndarray]


def read_normal_runs(directory):
    directory = Path(directory).resolve()
    if directory.name not in ("normal_train", "normal_calibration", "test_faultfree"):
        raise TemperatureError("仅接受正常训练、正常校准或独立正常测试分区")
    groups = {}
    for batch in iter_prepared_chunks(directory.parent, directory.name):
        if "X_raw" not in batch:
            raise TemperatureError("需要预处理时保留的原始单位 X_raw")
        if np.any(batch["y_detect"] != 0) or np.any(batch["y_class"] != 0):
            raise TemperatureError("正常训练/评估分区中存在故障标签")
        names = tuple(str(x) for x in batch["feature_names"])
        raw = np.asarray(batch["X_raw"], dtype=float)
        t = np.asarray(batch["time_h"], dtype=float)
        if raw.shape != (len(t), len(names)) or len(set(names)) != len(names):
            raise TemperatureError("X_raw 列布局无效")
        source = str(batch["source"])
        row = groups.setdefault(source, {"names": names, "raw": [], "time": []})
        if row["names"] != names:
            raise TemperatureError("同一来源的数据块列顺序不一致")
        row["raw"].append(raw)
        row["time"].append(t)
    runs = []
    for source, row in groups.items():
        raw = np.concatenate(row["raw"])
        data = {name: raw[:, i] for i, name in enumerate(row["names"])}
        data["time_h"] = np.concatenate(row["time"])
        times(data)
        runs.append(NormalRun(source, directory.name, data))
    if not runs:
        raise TemperatureError(f"{directory} 没有可用正常轨迹")
    return runs


@dataclass
class Input:
    column: str
    label: str
    unit: str
    group: str
    source: str = ""
    reference_column: str | None = None


@dataclass
class Device:
    key: str
    label: str
    module: str
    target: str
    inputs: list[Input]
    notes: list[str] = field(default_factory=list)
    independent_temperature: bool = True


def build_definitions(template, available):
    """由图谱模板选择上游输入；液位、压力及公用工程为显式设备配置。"""
    def stream_inputs(key):
        signals = template["streams"][key]["signals"]
        result = []
        for quantity in ("温度", "总质量流量"):
            choices = [s for s in signals.get(quantity, []) if s["id"] in available]
            if not choices:
                raise TemperatureError(f"{key} 缺少 {quantity}")
            s = choices[0]
            result.append(Input(s["id"], f"{key}.{quantity}", s["unit"], key, s.get("source", "")))
        comps = {s["component"]: s for s in signals.get("摩尔百分含量", []) if s["id"] in available}
        if len(comps) < 2:
            raise TemperatureError(f"{key} 缺少组成测点")
        reference = sorted(comps)[-1]
        for comp in sorted(comps):
            if comp == reference:
                continue
            s = comps[comp]
            result.append(Input(s["id"], f"{key}.组成.{comp}(参考{reference})", "mol%", key,
                                s.get("source", ""), comps[reference]["id"]))
        return result

    def extras(items):
        return [Input(col, label, unit, group) for col, label, unit, group in items]

    definitions = [
        Device("reactor", "M1 反应器", "M1", "XMEAS_09", stream_inputs("S6") + extras([
            ("SM_129", "反应器冷却水进口温度", "℃", "cooling"),
            ("SM_131", "反应器冷却水质量流量", "kg/h", "cooling"),
            ("XMEAS_08", "反应器液位", "%", "state"),
            ("XMEAS_07", "反应器压力", "kPa", "state"),
            ("XMV_12", "搅拌速度指令", "%", "command")]),
            ["闭环温度变化很小，需检查弱激励；不将出口同源温度作为输入"]),
        Device("condenser", "M2 冷凝器", "M2", "SM_055", stream_inputs("S7") + extras([
            ("SM_134", "冷凝器冷却水进口温度", "℃", "cooling"),
            ("SM_136", "冷凝器冷却水质量流量", "kg/h", "cooling")]),
            ["SM_055 为出口物流焓反算温度，拟合仅评价派生信号一致性，不能视为独立设备测温验证"], False),
        Device("separator", "M2 气液分离器", "M2", "XMEAS_11", stream_inputs("S14") + extras([
            ("XMEAS_12", "分离器液位", "%", "state"),
            ("XMEAS_13", "分离器压力", "kPa", "state")]),
            ["入口 SM_055 为焓反算信号；气液相变影响由局部温度响应近似表示"]),
        Device("stripper", "M3 蒸馏/汽提塔", "M3", "XMEAS_18", stream_inputs("S4") + stream_inputs("S10") + extras([
            ("XMEAS_19", "再沸器蒸汽质量流量", "kg/h", "steam"),
            ("XMEAS_15", "汽提塔液位", "%", "state")]),
            ["采用塔底代表温度；各输入共享一个主导时间常数"]),
        Device("feed_mixer", "S1/S2/S3/S8/S5 → S6 汇合点", "interface", "SM_033",
               sum((stream_inputs(s) for s in ("S1", "S2", "S3", "S8", "S5")), []),
               ["按图谱 S5 在 S8 净循环流量测点下游汇入，单列其输入",
                "混合节点的一阶系数表示等效传输/测量响应，不解释为设备储热能力"]),
    ]
    for dev in definitions:
        required = {dev.target, *(x.column for x in dev.inputs)}
        if required - set(available):
            raise TemperatureError(f"{dev.key} 缺少 {sorted(required - set(available))}")
        if len({x.column for x in dev.inputs}) != len(dev.inputs):
            raise TemperatureError("输入列重复")
    return {d.key: d for d in definitions}


@dataclass
class Parameters:
    target_reference_c: float
    input_reference: list[float]
    input_scale: list[float]
    a: float
    c_c: float
    b: list[float]
    delays_steps: list[int]
    sample_s: float
    ridge: float
    diagnostics: dict
    training_intervals: list[dict]
    selection: dict = field(default_factory=dict)
    calibration: dict | None = None

    def interpretation(self, device):
        tau = -self.sample_s / math.log(self.a)
        rows = []
        # 仅用于解释性核查，不约束闭环数据回归；固定其它条件下的通常方向。
        expected_sign = {"SM_129": 1, "SM_131": -1, "SM_134": 1, "SM_136": -1, "XMEAS_19": 1}
        for j, spec in enumerate(device.inputs):
            rows.append({**asdict(spec), "reference": self.input_reference[j],
                         "b_c_per_input_unit": self.b[j], "K_c_per_input_unit": self.b[j] / (1-self.a),
                         "gain_unit": f"℃/({spec.unit})", "d_steps": self.delays_steps[j],
                         "theta_s": self.delays_steps[j] * self.sample_s,
                         "training_std": self.input_scale[j],
                         "expected_sign_if_others_fixed": expected_sign.get(spec.column),
                         "direction_matches_expectation": (self.b[j]*expected_sign[spec.column]>0
                             if spec.column in expected_sign and self.diagnostics["input_status"][j]=="estimated" else None),
                         "status": self.diagnostics["input_status"][j]})
        warnings = list(device.notes)
        for row in rows:
            if row["direction_matches_expectation"] is False:
                warnings.append(f"{row['column']} 增益符号与固定其它条件时的通常方向不符，需核查闭环相关及弱激励，不宜直接因果解释")
        if tau < self.sample_s:
            warnings.append("tau 小于采样间隔，无法凭当前采样可靠分辨该时间常数，响应接近静态关系")
        if self.diagnostics["a_at_bound"]:
            warnings.append("a 命中稳定性约束边界，时间常数解释应谨慎")
        if not self.diagnostics["full_rank"]:
            warnings.append("输入设计矩阵秩亏，有效增益不能独立辨识")
        if self.diagnostics["target_training_std_c"] < .05:
            warnings.append("正常目标温度变化小于 0.05℃，需用动态工况进一步验证增益；此值为提示标准")
        warnings.append("K 为闭环正常工况下的条件响应增益，不自动代表因果效应；所有输入共享 tau")
        return {"tau_s": tau, "tau_min": tau/60, "a": self.a, "c_c": self.c_c,
                "equilibrium_at_reference_c": self.target_reference_c + self.c_c/(1-self.a),
                "inputs": rows, "warnings": warnings}


def sampling_seconds(runs):
    values = [float(np.median(np.diff(times(run.data))) * 3600) for run in runs]
    sample = float(np.median(values))
    if sample <= 0 or not np.allclose(values, sample, atol=1e-5, rtol=1e-5):
        raise TemperatureError("各轨迹采样间隔不一致")
    return sample


def steps(seconds, sample, name):
    value = number(seconds, name)
    integer = int(round(value/sample))
    if value < 0 or abs(value - integer*sample) > max(1e-5, sample*1e-5):
        raise TemperatureError(f"{name} 必须为采样间隔的非负整数倍")
    return integer


def arrays(device, data):
    t = times(data)
    y = column(data, device.target)
    u = np.column_stack([column(data, x.column) for x in device.inputs])
    if len(y) != len(t) or len(u) != len(t):
        raise TemperatureError("温度、输入和时间未对齐")
    return t, y, u


def valid_prefix(t, y, u, sample):
    finite = np.isfinite(y) & np.isfinite(u).all(axis=1)
    invalid = np.r_[0, np.cumsum(~finite)]
    gap = np.r_[0, np.cumsum(~np.isclose(np.diff(t)*3600, sample, atol=1e-5, rtol=1e-5))]
    return invalid, gap


def regression_rows(device, runs, delays, sample):
    history = max(delays, default=0)
    ys, zs, us = [], [], []
    for run in runs:
        t, y, u = arrays(device, run.data)
        k = np.arange(history, len(t)-1)
        invalid, gap = valid_prefix(t, y, u, sample)
        valid = ((invalid[k+2] == invalid[k-history]) & (gap[k+1] == gap[k-history]))
        k = k[valid]
        if len(k):
            ys.append(y[k+1]);zs.append(y[k])
            us.append(u[k[:, None]-np.asarray(delays)[None, :], np.arange(u.shape[1])[None, :]])
    if not ys:
        raise TemperatureError("没有完整回归样本")
    return np.concatenate(ys), np.concatenate(zs), np.vstack(us)


def fit_parameters(device, runs, delays, sample, ridge):
    y, z, u = regression_rows(device, runs, delays, sample)
    if len(y) < max(50, 3*(u.shape[1]+2)):
        raise TemperatureError("正常训练数据不足")
    reference = float(z.mean())
    u0, scale = u.mean(axis=0), u.std(axis=0)
    active = scale > 1e-10*np.maximum(1, np.max(np.abs(u), axis=0))
    status = ["estimated" if flag else "unexcited_fixed_zero" for flag in active]
    # 排除与目标同源的输入，以及训练中完全重复的输入，避免复制目标或增益任意分摊。
    for j in range(u.shape[1]):
        target_alias = all(np.allclose(column(run.data, device.inputs[j].column),
                                       column(run.data, device.target), atol=1e-10, rtol=0)
                           for run in runs)
        if target_alias:
            active[j] = False;status[j] = "target_alias_excluded"
        for prior in np.flatnonzero(active[:j]):
            if np.allclose(u[:, j], u[:, prior], atol=1e-10, rtol=0):
                active[j] = False;status[j] = f"duplicate_of_{device.inputs[prior].column}"
                break
    safe_scale = np.where(scale > 0, scale, 1)
    x = (u[:, active]-u0[active])/safe_scale[active]
    yc, zc = y-y.mean(), z-z.mean()
    gram = x.T@x/len(y) + ridge*np.eye(x.shape[1])
    gy, gz = x.T@yc/len(y), x.T@zc/len(y)
    if x.shape[1]:
        pair = np.linalg.lstsq(gram, np.column_stack((gy, gz)), rcond=None)[0]
        by, bz = pair[:, 0], pair[:, 1]
    else:
        by = bz = np.empty(0)
    denominator = float(zc@zc/len(y) - gz@bz)
    raw_a = float((zc@yc/len(y) - gz@by)/denominator) if denominator > 1e-15 else 0.0
    a = float(np.clip(raw_a, 1e-6, 1-1e-6))
    b = np.zeros(u.shape[1]);b[active] = (by-a*bz)/safe_scale[active]
    c = float(y.mean()-reference)  # z-reference 的训练均值为零，u-u0 同理。
    design = np.column_stack((zc/max(float(zc.std()), 1e-12), x))
    rank = int(np.linalg.matrix_rank(design))
    singular = np.linalg.svd(design, compute_uv=False)
    condition = float(singular[0]/singular[-1]) if len(singular) and singular[-1] > 1e-12 else None
    diagnostics = {"rows": len(y), "input_status": status, "unconstrained_a": raw_a,
                   "a_at_bound": a in (1e-6, 1-1e-6), "design_rank": rank,
                   "design_columns": design.shape[1], "full_rank": rank == design.shape[1],
                   "condition_number": condition, "target_training_std_c": float(z.std()),
                   "objective": "mean squared one-step error + ridge * squared standardized input coefficients"}
    intervals = [{"source": run.source, "start_h": float(times(run.data)[0]),
                  "end_h": float(times(run.data)[-1])} for run in runs]
    return Parameters(reference, u0.tolist(), scale.tolist(), a, c, b.tolist(), list(delays),
                      sample, ridge, diagnostics, intervals)


def predict_windows(device, parameter, data, *, horizon_s=60, stride_s=None, warmup_steps=None, observed_data=None):
    """事后窗口递推：每窗只读取一次实际初温，窗口内不再读取目标温度来推进预测。"""
    t, y_input, u = arrays(device, data)
    if observed_data is not None:
        ot, y, _ = arrays(device, observed_data)
        if not np.array_equal(t, ot):
            raise TemperatureError("替换前后时间网格不同")
    else:
        y = y_input
    nstep = steps(horizon_s, parameter.sample_s, "horizon_s")
    stride = steps(horizon_s if stride_s is None else stride_s, parameter.sample_s, "stride_s")
    if nstep < 1 or stride < 1:
        raise TemperatureError("窗口和步长至少为一个采样间隔")
    history = max(parameter.delays_steps, default=0)
    first = max(history, history if warmup_steps is None else warmup_steps)
    starts = np.arange(first, len(t)-nstep, stride, dtype=int)
    invalid, gap = valid_prefix(t, y, u, parameter.sample_s)
    end = starts+nstep
    good = ((invalid[end+1] == invalid[starts-history]) & (gap[end] == gap[starts-history]))
    skipped = int((~good).sum())
    starts = starts[good]
    if not len(starts):
        raise TemperatureError(f"{device.key} 没有完整有效递推窗口")
    idx = starts[:, None]+np.arange(nstep+1)[None, :]
    pred = np.empty(idx.shape)
    pred[:, 0] = y[starts]
    one = np.empty((len(starts), nstep))
    delays = np.asarray(parameter.delays_steps)
    b = np.asarray(parameter.b)
    u0 = np.asarray(parameter.input_reference)
    ref = parameter.target_reference_c
    for offset in range(nstep):
        k = starts+offset
        lagged = u[k[:, None]-delays[None, :], np.arange(len(b))[None, :]]
        force = parameter.c_c + (lagged-u0)@b
        pred[:, offset+1] = ref+parameter.a*(pred[:, offset]-ref)+force
        one[:, offset] = ref+parameter.a*(y[k]-ref)+force
    observed = y[idx[:, 1:]]
    result = {"time_h": t[idx[:, 1:]], "observed_c": observed,
              "predicted_c": pred[:, 1:], "residual_c": observed-pred[:, 1:],
              "one_step_residual_c": observed-one,
              "persistence_residual_c": observed-y[starts, None],
              "start_h": t[starts], "end_h": t[starts+nstep],
              "initial_temperature_c": y[starts], "skipped_windows": skipped}
    if parameter.calibration is not None:
        cal = parameter.calibration
        result["score"] = np.abs(result["residual_c"]-cal["center_c"])/cal["scale_c"]
    return result


def summary(rows, parameter):
    def join(key):
        return np.concatenate([r[key].ravel() for r in rows])
    residual, observed, predicted = join("residual_c"), join("observed_c"), join("predicted_c")
    mse = float(np.mean(residual**2))
    persistence = float(np.mean(join("persistence_residual_c")**2))
    reference = float(np.mean((observed-parameter.target_reference_c)**2))
    correlation = float(np.corrcoef(observed, predicted)[0, 1]) if np.std(observed)>1e-12 and np.std(predicted)>1e-12 else None
    report = {"temperature_unit": "degC", "residual_unit": "degC_difference",
              "prediction_mode": "recursive windows; actual initial temperature only; recorded input trajectory",
              "windows": sum(len(r["start_h"]) for r in rows), "points_excluding_initial": len(residual),
              "skipped_windows": sum(r["skipped_windows"] for r in rows),
              "rmse_c": math.sqrt(mse), "mae_c": float(np.mean(np.abs(residual))),
              "bias_c": float(np.mean(residual)), "max_abs_c": float(np.max(np.abs(residual))),
              "one_step_rmse_c": float(np.sqrt(np.mean(join("one_step_residual_c")**2))),
              "persistence_rmse_c": math.sqrt(persistence), "training_mean_rmse_c": math.sqrt(reference),
              "skill_vs_persistence": 1-mse/persistence if persistence>0 else None,
              "skill_vs_training_mean": 1-mse/reference if reference>0 else None,
              "temperature_correlation": correlation,
              "improves_both_baselines": mse < persistence and mse < reference}
    if parameter.calibration:
        cal = parameter.calibration
        report.update({"exceedance_rate": float(np.mean(np.abs(residual-cal["center_c"])>cal["limit_c"])),
                       "centered_rmse_c": float(np.sqrt(np.mean((residual-cal["center_c"])**2)))})
    return report


class TemperatureSystem:
    def __init__(self, devices, *, horizon_s=60, stride_s=None):
        self.devices = devices
        self.horizon_s = number(horizon_s, "horizon_s", positive=True)
        self.stride_s = number(horizon_s if stride_s is None else stride_s, "stride_s", positive=True)
        self.parameters: dict[str, Parameters] = {}

    def fit(self, runs, *, delays_s=(0, 1, 5, 15, 30, 60), ridges=(.0001, .01, 1.), fixed_delays=None):
        runs = list(runs)
        if not runs or any(r.split != "normal_train" for r in runs):
            raise TemperatureError("参数拟合仅接受 normal_train")
        if len({r.source for r in runs}) != len(runs):
            raise TemperatureError("每条训练轨迹应有唯一来源")
        sample = sampling_seconds(runs)
        delays = sorted({steps(x, sample, "delay_s") for x in delays_s})
        penalties = sorted({number(x, "ridge", positive=True) for x in ridges})
        if not delays or not penalties:
            raise TemperatureError("延迟和正则候选不能为空")
        fixed_delays = fixed_delays or {}
        prefix, holdout = [], []
        for run in runs:
            cut = int(len(times(run.data))*.7)
            prefix.append(NormalRun(run.source, "normal_train", {k:v[:cut] for k,v in run.data.items()}))
            holdout.append(NormalRun(run.source, "internal_validation", {k:v[cut:] for k,v in run.data.items()}))
        fitted = {}
        for key, device in self.devices.items():
            fixed = fixed_delays.get(key, {})
            unknown = set(fixed)-{x.column for x in device.inputs}
            if unknown:
                raise TemperatureError(f"{key} 固定延迟包含未知输入 {unknown}")
            configs = [tuple(steps(fixed[x.column], sample, x.column) if x.column in fixed else d
                             for x in device.inputs) for d in delays]
            configs = list(dict.fromkeys(configs))
            warmup = max(max(d) for d in configs)
            scores, best = [], None
            for d in configs:
                for ridge in penalties:
                    p = fit_parameters(device, prefix, d, sample, ridge)
                    rows = [predict_windows(device, p, r.data, horizon_s=self.horizon_s,
                                            stride_s=self.stride_s, warmup_steps=warmup) for r in holdout]
                    metrics = summary(rows, p)
                    record = {"delays_steps": list(d), "ridge": ridge, "rmse_c": metrics["rmse_c"],
                              "persistence_rmse_c": metrics["persistence_rmse_c"],
                              "training_mean_rmse_c": metrics["training_mean_rmse_c"],
                              "a": p.a, "tau_s": -sample/math.log(p.a)}
                    scores.append(record)
                    if best is None or record["rmse_c"] < best[0]:
                        best = (record["rmse_c"], d, ridge, p, metrics)
            _, d, ridge, partial, metrics = best
            final = fit_parameters(device, runs, d, sample, ridge)
            final.selection = {"method": "chronological 70/30 holdout entirely inside normal_train",
                               "objective": "recursive window RMSE, initial points excluded",
                               "delay_policy": "shared selected delay; explicit per-input overrides if supplied",
                               "fixed_delays_s": fixed, "candidates": scores,
                               "internal_validation": metrics, "prefix_fit_a": partial.a,
                               "full_fit_a": final.a, "validation_warmup_steps": warmup}
            fitted[key] = final
            print(f"fitted {key}: a={final.a:.6g}, tau={-sample/math.log(final.a):.5g}s, ridge={ridge:g}", file=sys.stderr)
        self.parameters = fitted
        return self

    def predict(self, key, data, *, observed_data=None):
        return predict_windows(self.devices[key], self.parameters[key], data, horizon_s=self.horizon_s,
                               stride_s=self.stride_s, observed_data=observed_data)

    def _check_disjoint(self, key, runs):
        for run in runs:
            t = times(run.data)
            for interval in self.parameters[key].training_intervals:
                if run.source == interval["source"] and t[0] <= interval["end_h"] and t[-1] >= interval["start_h"]:
                    raise TemperatureError("校准/验证区间与参数拟合区间重叠")

    def metrics(self, key, runs, *, independent=True):
        runs = list(runs)
        if not runs:
            raise TemperatureError("没有评估轨迹")
        if independent:
            self._check_disjoint(key, runs)
        report = summary([self.predict(key, r.data) for r in runs], self.parameters[key])
        report["independent_of_fit"] = independent
        report["independent_temperature"] = self.devices[key].independent_temperature
        return report

    def calibrate(self, runs, *, quantile=.99):
        runs = list(runs)
        q = number(quantile, "quantile")
        if not 0<q<1 or not runs or any(r.split != "normal_calibration" for r in runs):
            raise TemperatureError("校准仅接受 normal_calibration，分位数应在0和1之间")
        results = {}
        for key in self.devices:
            self._check_disjoint(key, runs)
            residual = np.concatenate([self.predict(key, r.data)["residual_c"].ravel() for r in runs])
            center = float(np.median(residual))
            limit = float(np.quantile(np.abs(residual-center), q))
            results[key] = {"center_c": center, "limit_c": limit, "scale_c": max(limit, 1e-12),
                            "lower_c": center-limit, "upper_c": center+limit, "quantile": q,
                            "points": len(residual), "mode": "pointwise recursive residual band",
                            "note": "校准集约1-quantile超限为分位数构造结果，不代表独立检测性能"}
        for key, calibration in results.items():
            self.parameters[key].calibration = calibration
        return results

    def rank_replacements(self, key, actual, normal):
        """正常参考由外部 baseline 对齐提供；单组分替换由参考组分补偿以保持原总和。"""
        if not np.array_equal(times(actual), times(normal)):
            raise TemperatureError("正常参考与实际轨迹必须使用相同时间网格")
        p, device = self.parameters[key], self.devices[key]
        if p.calibration is None:
            raise TemperatureError("先校准残差尺度")
        original = self.predict(key, actual)
        r0 = original["residual_c"]
        center, scale = p.calibration["center_c"], p.calibration["scale_c"]
        results = []
        for spec in device.inputs:
            replacement = {spec.column: column(normal, spec.column)}
            if spec.reference_column:
                replacement[spec.reference_column] = (column(actual, spec.reference_column)
                    +column(actual, spec.column)-replacement[spec.column])
                if np.any((replacement[spec.column]<0) | (replacement[spec.column]>100)
                          | (replacement[spec.reference_column]<0) | (replacement[spec.reference_column]>100)):
                    results.append({"candidate": spec.column, "status": "invalid_composition_replacement"})
                    continue
            changed = self.predict(key, {**actual, **replacement}, observed_data=actual)
            if not np.array_equal(changed["time_h"], original["time_h"]):
                raise TemperatureError("替换前后有效窗口不同")
            rj = changed["residual_c"]
            results.append({"candidate": spec.column, "label": spec.label, "status": "ok",
                            "replacement_columns": list(replacement),
                            "influence_score": float(np.mean(np.abs(rj-r0))/scale),
                            "signed_residual_change_c": float(np.mean(np.abs(rj-center)-np.abs(r0-center))),
                            "interpretation": "条件贡献证据，需要候选自身异常及传播路径验证"})
        return sorted(results, key=lambda x:x.get("influence_score", -1), reverse=True)

    def save(self, path):
        write_json(path, {"format_version": VERSION, "temperature_unit": "degC", "time_unit": "s",
                          "horizon_s": self.horizon_s, "stride_s": self.stride_s,
                          "devices": {k:asdict(v) for k,v in self.devices.items()},
                          "parameters": {k:asdict(v) for k,v in self.parameters.items()}})

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format_version") != VERSION or data.get("temperature_unit") != "degC" or data.get("time_unit") != "s":
            raise TemperatureError("不支持的模型格式或单位")
        devices = {}
        for key, row in data["devices"].items():
            devices[key] = Device(**{**row, "inputs": [Input(**x) for x in row["inputs"]]})
        system = cls(devices, horizon_s=data["horizon_s"], stride_s=data["stride_s"])
        system.parameters = {key:Parameters(**row) for key,row in data["parameters"].items()}
        if set(devices) != set(system.parameters):
            raise TemperatureError("设备与参数集合不一致")
        for key, p in system.parameters.items():
            if not 0<number(p.a, "a")<1:
                raise TemperatureError("a 必须满足 0<a<1")
            number(p.sample_s, "sample_s", positive=True)
            number(p.c_c, "c_c");number(p.target_reference_c, "target_reference_c")
            for values in [p.b, p.input_reference, p.input_scale, p.delays_steps]:
                if len(values) != len(devices[key].inputs) or not np.isfinite(values).all():
                    raise TemperatureError("输入参数维度或数值无效")
            if any(int(d)!=d or d<0 for d in p.delays_steps):
                raise TemperatureError("延迟应为非负整数")
            if p.calibration:
                number(p.calibration["center_c"], "center_c")
                number(p.calibration["scale_c"], "scale_c", positive=True)
        return system


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    fd, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def plot_calibration(system, runs, output):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"costep-temperature-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    path = Path("C:/Windows/Fonts/msyh.ttc")
    font = FontProperties(fname=str(path)) if path.exists() else FontProperties()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10})
    output = Path(output)
    overview, grid = plt.subplots(len(system.devices), 2, figsize=(18, 4*len(system.devices)), squeeze=False)
    artifacts = {}
    for row_index, (key, dev) in enumerate(system.devices.items()):
        p = system.parameters[key]
        if not p.calibration:
            raise TemperatureError("绘图前需要完成校准")
        cal = p.calibration
        rows = [system.predict(key, run.data) for run in runs]
        folder = output/key
        folder.mkdir(parents=True, exist_ok=True)
        metrics = summary(rows, p)
        note = f"a={p.a:.5g}；τ={-p.sample_s/math.log(p.a):.5g}s；窗口 {system.horizon_s:g}s，仅初温实测初始化"
        if not dev.independent_temperature:
            note += "\n目标为焓反算温度，独立物理解释受限"
        def decorate(ax, ylabel):
            ax.set_xlabel("时间 / h", fontproperties=font)
            ax.set_ylabel(ylabel, fontproperties=font)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.grid(alpha=.18)
            ax.legend(prop=font, loc="upper right")
            ax.text(.01, .015, note, transform=ax.transAxes, fontproperties=font, fontsize=8,
                    bbox={"facecolor":"white", "alpha":.85, "edgecolor":"none"})
            ax.margins(y=.15)
        def residual(ax):
            for i, r in enumerate(rows):
                ax.plot(r["time_h"].ravel(), r["residual_c"].ravel(), lw=.65, color="#255e8b", label="原始温度残差" if i==0 else None)
            ax.axhspan(cal["lower_c"], cal["upper_c"], color="#dca244", alpha=.23, label=f"{cal['quantile']:.0%}正常阈值带")
            ax.axhline(cal["center_c"], color="#287a54", lw=1.2, label="正常中心")
            for bound in (cal["lower_c"], cal["upper_c"]):
                ax.axhline(bound, color="#b27921", ls="--", lw=.8)
            ax.set_title(dev.label+f"｜残差中心 {cal['center_c']:.4g}℃，阈值半宽 {cal['limit_c']:.4g}℃", fontproperties=font)
            decorate(ax, "实测 − 递推预测 / ℃")
        def comparison(ax, *, zoom=False):
            zoom_start = min(r["time_h"].min() for r in rows)
            for i, r in enumerate(rows):
                time = r["time_h"].ravel()
                mask = time<=zoom_start+600/3600 if zoom else np.ones(len(time), dtype=bool)
                ax.plot(time[mask], r["observed_c"].ravel()[mask], lw=.8, alpha=.75, color="#255e8b", label="实际温度" if i==0 else None)
                ax.plot(time[mask], r["predicted_c"].ravel()[mask], lw=.8, color="#d56d32", label="窗口递推预测温度" if i==0 else None)
                if zoom:
                    baseline = (r["observed_c"]-r["persistence_residual_c"]).ravel()
                    ax.plot(time[mask], baseline[mask], lw=.8, ls="--", color="#3f8468", label="保持窗口初温基线" if i==0 else None)
            title = "前10分钟局部对比" if zoom else f"校准 RMSE={metrics['rmse_c']:.5g}℃；保持初温基线 {metrics['persistence_rmse_c']:.5g}℃"
            ax.set_title(dev.label+"｜"+title, fontproperties=font)
            decorate(ax, "温度 / ℃")
        fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
        residual(ax)
        fig.savefig(folder/"calibration_residual.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
        comparison(axes[0]);comparison(axes[1], zoom=True)
        fig.savefig(folder/"calibration_temperature.png", dpi=160)
        plt.close(fig)
        residual(grid[row_index, 0]);comparison(grid[row_index, 1])
        np.savez_compressed(folder/"calibration_windows.npz", **{
            f"run{i}_{name}":value for i,r in enumerate(rows) for name,value in r.items() if isinstance(value, np.ndarray)})
        artifacts[key] = {"residual_plot": str(folder/"calibration_residual.png"),
                          "temperature_plot": str(folder/"calibration_temperature.png")}
    overview.tight_layout(pad=2)
    overview.savefig(output/"calibration_overview.png", dpi=140)
    plt.close(overview)
    return {"overview": str(output/"calibration_overview.png"), "devices": artifacts}


def export_interpretation(system, output):
    report = {key:p.interpretation(system.devices[key]) for key,p in system.parameters.items()}
    write_json(Path(output)/"interpretation.json", report)
    with (Path(output)/"input_gains.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        keys = ["device", "column", "label", "unit", "b_c_per_input_unit", "K_c_per_input_unit", "d_steps", "theta_s", "training_std", "status", "expected_sign_if_others_fixed", "direction_matches_expectation"]
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for key, item in report.items():
            for row in item["inputs"]:
                writer.writerow({"device":key, **row})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="拟合一阶 ARX、校准温度残差并生成图形")
    train.add_argument("--train-dir", type=Path, default=DATA/"normal_train")
    train.add_argument("--calibration-dir", type=Path, default=DATA/"normal_calibration")
    train.add_argument("--template", type=Path, default=TEMPLATE)
    train.add_argument("--output", type=Path, default=OUTPUT)
    train.add_argument("--horizon-s", type=float, default=60)
    train.add_argument("--stride-s", type=float)
    train.add_argument("--delay-seconds", type=float, nargs="+", default=[0, 1, 5, 15, 30, 60])
    train.add_argument("--ridges", type=float, nargs="+", default=[.0001, .01, 1])
    train.add_argument("--delays-json", type=Path)
    train.add_argument("--quantile", type=float, default=.99)
    validate = commands.add_parser("validate", help="冻结参数及阈值，评估独立正常测试数据")
    validate.add_argument("--model", type=Path, default=OUTPUT/"parameters.json")
    validate.add_argument("--data-dir", type=Path, default=DATA/"test_faultfree")
    validate.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            system = TemperatureSystem.load(args.model)
            runs = read_normal_runs(args.data_dir)
            report = {key:system.metrics(key, runs) for key in system.devices}
            write_json(args.output or args.model.parent/"validation.json", report)
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            training, calibration = read_normal_runs(args.train_dir), read_normal_runs(args.calibration_dir)
            template = json.loads(args.template.read_text(encoding="utf-8"))
            devices = build_definitions(template, training[0].data)
            system = TemperatureSystem(devices, horizon_s=args.horizon_s, stride_s=args.stride_s)
            fixed = json.loads(args.delays_json.read_text(encoding="utf-8")) if args.delays_json else None
            system.fit(training, delays_s=args.delay_seconds, ridges=args.ridges, fixed_delays=fixed)
            system.calibrate(calibration, quantile=args.quantile)
            args.output.mkdir(parents=True, exist_ok=True)
            system.save(args.output/"parameters.json")
            interpretation = export_interpretation(system, args.output)
            report = {key:{"device":asdict(dev), "parameters":asdict(system.parameters[key]),
                           "interpretation":interpretation[key],
                           "training":system.metrics(key, training, independent=False),
                           "calibration":system.metrics(key, calibration)} for key,dev in devices.items()}
            plots = plot_calibration(system, calibration, args.output)
            write_json(args.output/"report.json", {"complete":True, "models":report, "plots":plots})
            print(json.dumps({key:{"a":system.parameters[key].a, "tau_s":interpretation[key]["tau_s"],
                                   "calibration":report[key]["calibration"]} for key in devices}, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()

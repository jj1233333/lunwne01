"""同工况正常窗口匹配，供第二层模板共用。

实际轨迹与参考轨迹采用 time_h 和原始物理信号。用 CCA 起报点之前的
历史做匹配，在所有合格正常历史窗口上拟合一套均值与标准差，再用欧氏
距离选一个近邻。返回该近邻从匹配时刻开始的候选变量轨迹。

    matcher = BaselineMatcher.from_preprocessed(
        match_features=("XMEAS_08", "SM_035"),
        candidate_columns=("SM_129",),
    )
    result = matcher.match(actual, cca_alarm_or_llm_plan, ("SM_129",))

预处理 NPZ 只读取 time_h、feature_names、X_raw。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "data/COSTE_cca_preprocessed"
NORMAL_SPLITS = ("normal_train", "normal_calibration")
SECONDS_PER_HOUR = 3600.0
TIME_TOLERANCE_S = 1e-6
PHYSICAL_ID = re.compile(r"(?:SM_\d{3}|XMEAS_\d{2}|XMV_\d{2})")


def _columns(names: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(names, str):
        names = (names,)
    output = tuple(dict.fromkeys(names))
    if any(not isinstance(name, str) or not PHYSICAL_ID.fullmatch(name) for name in output):
        raise ValueError("匹配变量须使用 SM、XMEAS 或 XMV 物理信号编号")
    return output


def _condition_key(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("工况名称必须是非空字符串")
    key = re.sub(r"[^a-z0-9]", "", name.casefold())
    if (key.startswith(("fault", "idv", "scenario", "label", "anomaly", "groundtruth"))
            or key in {"randomseed", "ytrue", "yclass", "ydetect"}
            or name.strip().startswith(("故障", "异常", "标签"))):
        raise ValueError("故障标签及仿真标签元数据不能用作工况")
    return name


@dataclass(frozen=True)
class NormalRun:
    run_id: str
    data: Any
    conditions: Mapping[str, Any] = field(default_factory=dict)
    split: str | None = None


def read_physical_runs(directory: Path | str, columns: str | Iterable[str], *,
                       split: str | None = None,
                       conditions_by_run: Mapping[str, Mapping[str, Any]] | None = None) -> list[NormalRun]:
    """Read selected columns and merge chunks within a run and split."""
    columns = _columns(columns)
    if not columns:
        raise ValueError("读取轨迹至少需要一个物理测点")
    directory = Path(directory).resolve()
    by_folder = defaultdict(list)
    for path in sorted(directory.rglob("part_*.npz")):
        by_folder[path.parent].append(path)
    if not by_folder:
        raise ValueError(f"目录没有预处理的 part_*.npz：{directory}")
    runs = []
    for folder, chunks in by_folder.items():
        times, matrices = [], []
        for path in chunks:
            with np.load(path, allow_pickle=False) as data:
                names = data["feature_names"].tolist()
                time = np.asarray(data["time_h"], dtype=float)
                raw = data["X_raw"]
                if (not isinstance(names, list) or len(names) != len(set(names)) or time.ndim != 1
                        or raw.shape != (len(time), len(names))):
                    raise ValueError(f"预处理数据列或时序不一致：{path}")
                missing = sorted(set(columns) - set(names))
                if missing:
                    raise ValueError(f"缺少物理数据列 {missing}：{path}")
                matrices.append(raw[:, [names.index(name) for name in columns]].astype(float, copy=False))
                times.append(time)
        values = np.concatenate(matrices, axis=0)
        data = {"time_h": np.concatenate(times), **dict(zip(columns, values.T))}
        _times(data)
        tags = (conditions_by_run or {}).get(folder.name, {})
        runs.append(NormalRun(folder.name, data, tags, split))
    return runs


def _times(data):
    values = np.asarray(data["time_h"], dtype=float)
    if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
            or np.any(np.diff(values) <= 0)):
        raise ValueError("单段轨迹至少需要两个严格递增的 time_h")
    return values


def _slice_values(data, columns, section, n):
    arrays = []
    for name in columns:
        values = data[name]
        if len(values) != n:
            raise ValueError(f"{name} 与 time_h 未对齐")
        sliced = values.iloc[section] if hasattr(values, "iloc") else values[section]
        series = np.asarray(sliced, dtype=float)
        if series.ndim != 1:
            raise ValueError(f"{name} 必须是一维物理信号")
        arrays.append(series)
    return np.column_stack(arrays)


def _seconds(time_h):
    return (time_h - time_h[0]) * SECONDS_PER_HOUR


def _time_index(time_h, point):
    if not math.isfinite(point):
        raise ValueError("诊断窗口和报警时刻必须是有限数")
    i = int(np.searchsorted(time_h, point))
    available = [j for j in (i - 1, i) if 0 <= j < len(time_h)]
    idx = min(available, key=lambda j: abs(time_h[j] - point))
    if abs(time_h[idx] - point) * SECONDS_PER_HOUR > TIME_TOLERANCE_S:
        raise ValueError("确认时刻和诊断窗口端点必须对应实际采样点")
    return idx


def _gap_counts(time_s, maximum_s):
    increments = np.diff(time_s)
    limit = 1.5 * float(np.median(increments)) if maximum_s is None else maximum_s
    return np.r_[0, np.cumsum(increments > limit + TIME_TOLERANCE_S)]


def _history_starts(time_s, anchors, length_s):
    desired = time_s[anchors] - length_s
    starts = np.searchsorted(time_s, desired - TIME_TOLERANCE_S)
    covered = (desired >= time_s[0] - TIME_TOLERANCE_S) & (starts < anchors)
    return starts, covered


def _block_counts(mask):
    return np.r_[0, np.cumsum(mask)]


@dataclass(frozen=True)
class _Windows:
    run: NormalRun
    time_h: np.ndarray
    time_s: np.ndarray
    anchors: np.ndarray
    starts: np.ndarray
    means: np.ndarray
    candidate_values: np.ndarray


class BaselineMatcher:
    """Select the single nearest eligible normal window."""

    def __init__(self, normal_runs: Iterable[NormalRun], match_features: str | Iterable[str], *,
                 history_s: float = 50.0, lookback_s: float = 30.0, step: int = 1,
                 max_gap_s: float | None = None, exact_conditions: Iterable[str] = (),
                 condition_tolerances: Mapping[str, float] | None = None):
        self.runs = tuple(normal_runs)
        self.features = _columns(match_features)
        if not self.runs or not self.features:
            raise ValueError("必须提供正常参考轨迹及工况匹配特征")
        if not math.isfinite(history_s) or history_s <= 0:
            raise ValueError("history_s 必须大于零")
        if not math.isfinite(lookback_s) or lookback_s < 0:
            raise ValueError("lookback_s 不能为负")
        if type(step) is not int or step < 1:
            raise ValueError("step 必须是正整数采样点数")
        if max_gap_s is not None and (not math.isfinite(max_gap_s) or max_gap_s <= 0):
            raise ValueError("max_gap_s 必须大于零")
        self.history_s, self.lookback_s, self.step = history_s, lookback_s, step
        self.max_gap_s = max_gap_s
        self.exact_conditions = tuple(_condition_key(key) for key in exact_conditions)
        self.condition_tolerances = {
            _condition_key(key): float(value) for key, value in (condition_tolerances or {}).items()
        }
        if set(self.exact_conditions) & self.condition_tolerances.keys():
            raise ValueError("同一个工况字段不可同时使用精确匹配和容差匹配")
        if any(not math.isfinite(value) or value < 0 for value in self.condition_tolerances.values()):
            raise ValueError("工况容差必须是非负有限数")

    @classmethod
    def from_preprocessed(cls, *, match_features: str | Iterable[str],
                          candidate_columns: str | Iterable[str],
                          data_root: Path | str = DEFAULT_DATA_ROOT,
                          reference_splits: Iterable[str] = NORMAL_SPLITS,
                          conditions_by_run: Mapping[str, Mapping[str, Any]] | None = None,
                          **settings):
        """Load the project normal library. Validations can select normal_train only."""
        features, candidates = _columns(match_features), _columns(candidate_columns)
        names = tuple(dict.fromkeys((*features, *candidates)))
        splits = tuple(dict.fromkeys(reference_splits))
        if not splits or set(splits) - set(NORMAL_SPLITS):
            raise ValueError("正常参考库只接受 normal_train 和 normal_calibration")
        runs = [
            run for split in splits
            for run in read_physical_runs(Path(data_root) / split, names, split=split,
                                          conditions_by_run=conditions_by_run)
        ]
        return cls(runs, features, **settings)

    def _compatible(self, actual, reference):
        for key in self.exact_conditions:
            if key not in actual or key not in reference or actual[key] != reference[key]:
                return False
        for key, tolerance in self.condition_tolerances.items():
            if key not in actual or key not in reference:
                return False
            delta = float(actual[key]) - float(reference[key])
            if not math.isfinite(delta) or abs(delta) > tolerance:
                return False
        return True

    def _windows(self, run, features, candidates, horizon_s, nonnegative):
        time_h = _times(run.data)
        time_s = _seconds(time_h)
        try:
            states = _slice_values(run.data, features, slice(None), len(time_h))
            targets = _slice_values(run.data, candidates, slice(None), len(time_h))
        except KeyError:
            return None
        anchors = np.arange(0, len(time_s), self.step)
        starts, covered = _history_starts(time_s, anchors, self.history_s)
        raw_end = time_s[anchors] + horizon_s
        ends = np.minimum(np.searchsorted(time_s, raw_end - TIME_TOLERANCE_S), len(time_s) - 1)
        gaps = _gap_counts(time_s, self.max_gap_s)
        good_state = np.isfinite(states).all(axis=1)
        bad_history = _block_counts(~good_state)
        bad_target = ~np.isfinite(targets).all(axis=1)
        for j, name in enumerate(candidates):
            if name in nonnegative:
                bad_target |= targets[:, j] < 0
        bad_followup = _block_counts(bad_target)
        keep = (covered & (raw_end <= time_s[-1] + TIME_TOLERANCE_S)
                & (gaps[starts] == gaps[ends])
                & (bad_history[starts] == bad_history[anchors + 1])
                & (bad_followup[anchors] == bad_followup[ends + 1]))
        anchors, starts = anchors[keep], starts[keep]
        if not len(anchors):
            return None
        # Center before the cumulative sum to reduce loss of precision on constant features.
        first_valid = np.flatnonzero(good_state)[0]
        offset = states[first_valid]
        sums = np.vstack((np.zeros(len(features)), np.cumsum(
            np.where(good_state[:, None], states - offset, 0.0), axis=0,
        )))
        means = (sums[anchors + 1] - sums[starts]) / (anchors - starts + 1)[:, None] + offset
        return _Windows(run, time_h, time_s, anchors, starts, means, targets)

    def match(self, actual, alarm: Mapping, candidate_columns: str | Iterable[str], *,
              window: tuple[float, float] | None = None, cutoff_time_h: float | None = None,
              conditions: Mapping | None = None, actual_run_id: str | None = None,
              exclude_features: str | Iterable[str] = (), max_distance: float | None = None,
              nonnegative_candidates: str | Iterable[str] = ()) -> dict:
        """Return time-aligned replacements in original units, or why none match.

        window is an optional subwindow inside the matched time through CCA confirmation.
        NormalRun supplies the run ID and conditions automatically. For raw mappings,
        actual_run_id excludes overlapping reference segments from the indicated run.
        """
        candidates = _columns(candidate_columns)
        if not candidates:
            raise ValueError("至少需要一个待替换的物理变量")
        nonnegative = set(_columns(nonnegative_candidates))
        if nonnegative - set(candidates):
            raise ValueError("nonnegative_candidates 必须包含在待替换变量中")
        if max_distance is not None and (not math.isfinite(max_distance) or max_distance < 0):
            raise ValueError("max_distance 必须是非负有限数")
        event = alarm["alarm"] if "alarm" in alarm else alarm
        onset, confirm = float(event["onset_time_h"]), float(event["confirmation_time_h"])
        if not math.isfinite(onset) or not math.isfinite(confirm) or confirm < onset:
            raise ValueError("CCA 报警起点与确认时刻无效")
        if cutoff_time_h is not None and not math.isfinite(cutoff_time_h):
            raise ValueError("cutoff_time_h 必须是有限数")
        if isinstance(actual, NormalRun):
            actual_run_id = actual.run_id if actual_run_id is None else actual_run_id
            conditions = actual.conditions if conditions is None else conditions
            actual = actual.data
        excluded = set(_columns(exclude_features)) | set(candidates) | {event.get("variable_id")}
        features = tuple(name for name in self.features if name not in excluded)
        result = dict(status="blocked", reason=None, trajectory=None, match_features=features,
                      candidate_columns=candidates, match_time_h=None, history_window_h=None, window_h=None,
                      distance=None, max_distance=max_distance, reference=None, reference_window_count=0,
                      normalization=None, condition_filter=bool(self.exact_conditions or self.condition_tolerances))

        def blocked(reason):
            return {**result, "reason": reason}

        if not features:
            return blocked("no_unaffected_matching_features")
        time_h = _times(actual)
        time_s = _seconds(time_h)
        confirm_idx = _time_index(time_h, confirm)
        cutoff = onset - self.lookback_s / SECONDS_PER_HOUR
        if cutoff_time_h is not None:
            cutoff = min(cutoff, cutoff_time_h)
        tol_h = TIME_TOLERANCE_S / SECONDS_PER_HOUR
        anchor = min(np.searchsorted(time_h, cutoff + tol_h, side="right"),
                     np.searchsorted(time_h, onset - tol_h)) - 1
        if anchor < 1:
            return blocked("insufficient_pre_alarm_history")
        result["match_time_h"] = float(time_h[anchor])
        starts, covered = _history_starts(time_s, np.array([anchor]), self.history_s)
        if not covered[0]:
            return blocked("insufficient_pre_alarm_history")
        start = int(starts[0])
        result["history_window_h"] = (float(time_h[start]), float(time_h[anchor]))
        first, last = (anchor, confirm_idx) if window is None else (
            _time_index(time_h, float(window[0])), _time_index(time_h, float(window[1]))
        )
        if not anchor <= first < last <= confirm_idx:
            raise ValueError("诊断子窗口须位于匹配时刻至报警确认时刻之间且至少有两个采样点")
        result["window_h"] = (float(time_h[first]), float(time_h[last]))
        gap_limit = (self.max_gap_s if self.max_gap_s is not None
                     else 1.5 * float(np.median(np.diff(time_s[:anchor + 1]))))
        gaps = _gap_counts(time_s[:last + 1], gap_limit)
        if gaps[start] != gaps[anchor]:
            return blocked("gap_in_actual_history")
        if gaps[anchor] != gaps[last]:
            return blocked("gap_in_actual_diagnostic_window")
        try:
            observed = _slice_values(actual, features, slice(start, anchor + 1), len(time_h))
        except KeyError:
            return blocked("missing_actual_matching_feature")
        if not np.isfinite(observed).all():
            return blocked("invalid_actual_matching_feature")
        query = observed.mean(axis=0)
        offsets = time_s[first:last + 1] - time_s[anchor]

        runs = []
        for run in self.runs:
            if run.data is actual or not self._compatible(conditions or {}, run.conditions):
                continue
            if actual_run_id is not None and run.run_id == actual_run_id:
                ref_time = _times(run.data)
                if max(ref_time[0], time_h[0]) <= min(ref_time[-1], time_h[-1]):
                    continue
            runs.append(run)
        if not runs:
            return blocked("no_compatible_normal_run")
        references = []
        for run in runs:
            windows = self._windows(run, features, candidates, offsets[-1], nonnegative)
            if windows is not None:
                references.append(windows)
        if not references:
            return blocked("no_complete_normal_window")
        return self._nearest(result, references, query, time_h[first:last + 1], offsets)

    @staticmethod
    def _nearest(result, references, query, grid, offsets):
        features, candidates = result["match_features"], result["candidate_columns"]
        values = np.concatenate([reference.means for reference in references])
        center = values.mean(axis=0)
        std = (values - center).std(axis=0)
        constant = std <= 16 * np.finfo(float).eps * np.maximum(1, np.abs(center))
        scale = np.where(constant, 1.0, std)
        normal_z = (values - center) / scale
        actual_z = (query - center) / scale
        distances = np.linalg.norm(normal_z - actual_z, axis=1)
        best = int(np.argmin(distances))
        distance = float(distances[best])
        counts = np.cumsum([len(ref.anchors) for ref in references])
        ref_index = int(np.searchsorted(counts, best, side="right"))
        ref = references[ref_index]
        local_index = best - (int(counts[ref_index - 1]) if ref_index else 0)
        anchor = ref.anchors[local_index]
        start = ref.starts[local_index]
        requested = ref.time_s[anchor] + offsets
        reference = dict(run_id=ref.run.run_id, split=ref.run.split,
                         history_window_h=(float(ref.time_h[start]), float(ref.time_h[anchor])),
                         match_time_h=float(ref.time_h[anchor]),
                         replacement_window_h=(float(ref.time_h[0] + requested[0] / SECONDS_PER_HOUR),
                                               float(ref.time_h[0] + requested[-1] / SECONDS_PER_HOUR)))
        result.update(distance=distance, reference=reference, reference_window_count=len(values),
                      normalization={
                          name: dict(mean=float(center[j]), std=float(std[j]), scale=float(scale[j]),
                                     actual_history_mean=float(query[j]),
                                     reference_history_mean=float(ref.means[local_index, j]))
                          for j, name in enumerate(features)
                      })
        if result["max_distance"] is not None and distance > result["max_distance"]:
            return {**result, "reason": "distance_exceeds_limit"}
        # Snap roundoff around measured times to avoid interpolating into an adjacent invalid value.
        right = np.minimum(np.searchsorted(ref.time_s, requested), len(ref.time_s) - 1)
        left = np.maximum(right - 1, 0)
        nearest = np.where(abs(ref.time_s[left] - requested) <= abs(ref.time_s[right] - requested), left, right)
        requested = np.where(abs(ref.time_s[nearest] - requested) <= TIME_TOLERANCE_S,
                             ref.time_s[nearest], requested)
        trajectory = {"time_h": grid.copy()}
        for j, name in enumerate(candidates):
            trajectory[name] = np.interp(requested, ref.time_s, ref.candidate_values[:, j])
        return {**result, "status": "ready", "trajectory": trajectory}

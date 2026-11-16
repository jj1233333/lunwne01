"""由 Neo4j dump 建立对象级经典 CCA，并校准原单位的逐变量预测残差。

在项目根目录运行：
    .venv/Scripts/python.exe src/costep_diagnosis/layer1/cca.py train
    .venv/Scripts/python.exe src/costep_diagnosis/layer1/cca.py predict --split test_fault

P 包含同一设备/管线直接关联的检测变量和操作参数的前 window 个采样点，
F 为紧接着的检测值。训练只使用 normal_train 及其已有标准化参数。
normal_calibration 上逐变量拟合 b=median(e)、q=quantile(abs(e-b), .99)。
score=abs(e-b)/q，direction 取 e-b 的符号；默认连续 3 个有效点 score>1 报警。
每段连续超限只生成一个事件，同时保留起点、确认点和逐点残差，供第二层追溯。
故障推理按原始轨迹分别导出 events.jsonl、逐点 NPZ 和 report.json；
每条轨迹的 COMPLETE.json 写入后，该目录即可交给第二层读取。
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable, Iterator

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from costep_diagnosis.data_convert.data import (  # noqa: E402
    FEATURE_COLUMNS, NormalScaler, iter_prepared_chunks,
)

DEFAULT_DATA = PROJECT_ROOT / "data" / "COSTE_cca_preprocessed"
DEFAULT_DUMP = PROJECT_ROOT / "data" / "kg" / "costep-2026-11-10T07-12-29.dump"
DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "layer1_cca"
FORMAT_VERSION = 1


class CCAError(ValueError):
    """图谱、数据或模型不满足第一层接口约定。"""


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


# 使用与 dump 兼容的 Neo4j 官方库读取隔离副本，不启动网络服务。
GRAPH_READER_JAVA = r'''
import java.nio.file.Path;
import java.util.*;
import org.neo4j.dbms.api.*;
import org.neo4j.configuration.GraphDatabaseSettings;
import org.neo4j.graphdb.*;
import com.fasterxml.jackson.databind.ObjectMapper;

class ReadCcaGraph {
    @SuppressWarnings("unchecked")
    public static void main(String[] args) throws Exception {
        Neo4jDatabaseManagementServiceBuilder builder;
        try {
            builder = (Neo4jDatabaseManagementServiceBuilder) Class
                .forName("com.neo4j.dbms.api.EnterpriseDatabaseManagementServiceBuilder")
                .getConstructor(Path.class).newInstance(Path.of(args[0]));
            var replication = (org.neo4j.graphdb.config.Setting<Boolean>) Class
                .forName("com.neo4j.configuration.EnterpriseEditionInternalSettings")
                .getField("replication_enabled").get(null);
            builder.setConfig(replication, false);
            var backup = (org.neo4j.graphdb.config.Setting<Boolean>) Class
                .forName("com.neo4j.configuration.OnlineBackupSettings")
                .getField("online_backup_enabled").get(null);
            builder.setConfig(backup, false);
        } catch (ClassNotFoundException absent) {
            builder = new DatabaseManagementServiceBuilder(Path.of(args[0]));
        }
        var management = builder
            .setConfig(GraphDatabaseSettings.initial_default_database, "costep")
            .setConfig(GraphDatabaseSettings.pagecache_memory, 67108864L)
            .build();
        try {
            var nodes = new ArrayList<Map<String, Object>>();
            var edges = new ArrayList<Map<String, Object>>();
            GraphDatabaseService database = null;
            long deadline = System.nanoTime() + 30000000000L;
            while (database == null && System.nanoTime() < deadline) {
                try {
                    database = management.database("costep");
                } catch (DatabaseNotFoundException initializing) {
                    Thread.sleep(100);
                }
            }
            if (database == null) throw new IllegalStateException("Default costep database was not initialized");
            if (!database.isAvailable(30000)) throw new IllegalStateException("Loaded costep database is unavailable");
            try (var tx = database.beginTx()) {
                try (var allNodes = tx.getAllNodes()) {
                    for (var node : allNodes) {
                        var labels = new ArrayList<String>();
                        for (var label : node.getLabels()) labels.add(label.name());
                        nodes.add(Map.of("id", node.getElementId(), "labels", labels,
                            "properties", node.getAllProperties()));
                    }
                }
                try (var allEdges = tx.getAllRelationships()) {
                    for (var edge : allEdges) {
                        edges.add(Map.of("source", edge.getStartNode().getElementId(),
                            "target", edge.getEndNode().getElementId(),
                            "type", edge.getType().name(), "properties", edge.getAllProperties()));
                    }
                }
            }
            new ObjectMapper().writeValue(Path.of(args[1]).toFile(), Map.of("nodes", nodes, "edges", edges));
        } finally {
            management.shutdown();
        }
    }
}
'''


def read_dump(dump: Path, neo4j_home: Path | None = None) -> dict:
    """把 dump 加载到全新的临时实例，导出节点和关系后关闭并清理副本。"""
    dump = Path(dump).resolve()
    if not dump.is_file():
        raise CCAError(f"图谱 dump 不存在：{dump}")
    if neo4j_home is None:
        desktop_cache = Path.home() / ".Neo4jDesktop2" / "Cache" / "dbmss"
        desktop_versions = sorted(desktop_cache.glob("neo4j-enterprise-*"), reverse=True)
        executable = shutil.which("neo4j-admin.bat") or shutil.which("neo4j-admin")
        if desktop_versions:
            neo4j_home = desktop_versions[0]
        elif executable:
            neo4j_home = Path(executable).resolve().parents[1]
        elif os.getenv("NEO4J_HOME"):
            neo4j_home = Path(os.environ["NEO4J_HOME"])
    if neo4j_home is None or not (Path(neo4j_home) / "lib").is_dir():
        raise CCAError("读取二进制 dump 需要本机 Neo4j 库，请用 --neo4j-home 指定安装目录")
    java = shutil.which("java")
    if java is None:
        raise CCAError("读取 dump 需要 Neo4j 支持的 Java，请把 java 加入 PATH")
    classpath = str(Path(neo4j_home).resolve() / "lib" / "*")
    runtime_parent = PROJECT_ROOT / "outputs"
    runtime_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="costep-cca-graph-", dir=runtime_parent) as folder:
        home = Path(folder)
        conf, imports = home / "conf", home / "import"
        conf.mkdir()
        imports.mkdir()
        (conf / "neo4j.conf").write_text(
            "server.memory.heap.initial_size=128m\nserver.memory.heap.max_size=512m\n"
            "server.memory.pagecache.size=64m\n", encoding="utf-8",
        )
        shutil.copyfile(dump, imports / "costep.dump")
        environment = os.environ.copy()
        environment.update(NEO4J_HOME=str(home), NEO4J_CONF=str(conf))
        helper = home / "ReadCcaGraph.java"
        helper.write_text(GRAPH_READER_JAVA, encoding="utf-8")
        snapshot = home / "graph.json"
        commands = [
            [java, "-Xmx512m", "-cp", classpath, "org.neo4j.cli.AdminTool", "database", "load",
             "costep", f"--from-path={imports}"],
            [java, "-Xmx512m", "-Dfile.encoding=UTF-8", "-cp", classpath,
             str(helper), str(home), str(snapshot)],
        ]
        for command in commands:
            result = subprocess.run(command, env=environment, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=180)
            if result.returncode:
                raise CCAError(f"隔离读取 dump 失败：\n{result.stdout[-1500:]}\n"
                               f"{result.stderr[:1800]}\n{result.stderr[-1800:]}")
        graph = json.loads(snapshot.read_text(encoding="utf-8"))
    graph["provenance"] = {"dump": str(dump), "sha256": hashlib.sha256(dump.read_bytes()).hexdigest(),
                           "neo4j_home": str(Path(neo4j_home).resolve())}
    return graph


@dataclass
class ObjectSpec:
    name: str
    kind: str
    modules: list[str]
    history_ids: list[str]
    target_ids: list[str]
    variable_names: list[str]
    units: list[str]


def graph_objects(graph: dict, feature_names: list[str], excluded: list[str]) -> tuple[list[ObjectSpec], dict]:
    """按检测位置、操作位置直接关系分组；模块来自显式的包含关系。"""
    nodes = {node["id"]: node for node in graph["nodes"]}
    locations = {key: node for key, node in nodes.items()
                 if set(node["labels"]) & {"设备", "管线"}}
    measurements = {key: [] for key in locations}
    controls = {key: [] for key in locations}
    modules = {key: set() for key in locations}
    unknown = set()
    used = set()
    for edge in graph["edges"]:
        source, target = nodes[edge["source"]], nodes[edge["target"]]
        if edge["target"] not in locations:
            continue
        props = source["properties"]
        if edge["type"] == "包含" and "模块" in source["labels"]:
            modules[edge["target"]].add(props["模块名称"])
        if edge["type"] == "检测位置" and "检测变量" in source["labels"]:
            variable_id = props["变量ID"]
            if variable_id in feature_names:
                measurements[edge["target"]].append(props)
                used.add(variable_id)
            elif variable_id not in excluded:
                unknown.add(variable_id)
        if edge["type"] == "操作位置" and "操作参数" in source["labels"]:
            variable_id = props["参数ID"]
            if variable_id in feature_names:
                controls[edge["target"]].append(variable_id)
                used.add(variable_id)
            else:
                unknown.add(variable_id)
    if unknown:
        raise CCAError(f"图谱变量缺少数据列且不在预处理排除清单中：{sorted(unknown)}")
    specs, skipped = [], []
    order = {name: i for i, name in enumerate(feature_names)}
    for key, node in locations.items():
        kind = "设备" if "设备" in node["labels"] else "管线"
        name = node["properties"][f"{kind}名称"]
        targets = {item["变量ID"]: item for item in measurements[key]}
        if not targets:
            skipped.append({"object_name": name, "object_kind": kind, "reason": "没有保留的直接检测变量",
                            "control_ids": sorted(set(controls[key]))})
            continue
        target_ids = sorted(targets, key=order.__getitem__)
        history_ids = sorted(set(target_ids) | set(controls[key]), key=order.__getitem__)
        specs.append(ObjectSpec(name, kind, sorted(modules[key]), history_ids, target_ids,
                                [targets[key]["变量名称"] for key in target_ids],
                                [targets[key].get("单位") or "" for key in target_ids]))
    specs.sort(key=lambda spec: (spec.kind, spec.name))
    if not specs:
        raise CCAError("图谱中没有能够与预处理数据对齐的检测位置关系")
    audit = {"grouping": "direct_detection_and_operation_location", "skipped_objects": skipped,
             "objects_without_modules": [spec.name for spec in specs if not spec.modules],
             "features_without_location": sorted(set(feature_names) - used),
             "features_not_used_by_any_model": sorted(set(feature_names) -
                                                       {name for spec in specs for name in spec.history_ids}),
             "excluded_feature_ids": excluded}
    return specs, audit


@dataclass
class Trajectory:
    source: str
    split: str
    raw: np.ndarray
    scaled: np.ndarray
    time_h: np.ndarray
    row_offset: int
    sample_seconds: float
    labels: np.ndarray


def read_trajectories(data: Path, split: str, scaler: NormalScaler,
                      source_filter: str | None = None) -> Iterator[Trajectory]:
    """合并同一轨迹的相邻存储块；分区之间始终分别读取。"""
    report = json.loads((Path(data) / "report.json").read_text(encoding="utf-8"))
    sources = {item["source"]: item for item in report["files"]}
    blocks = []
    previous_source = None

    def assemble(source: str, parts: list[dict]) -> Trajectory:
        times = np.concatenate([part["time_h"] for part in parts]).astype(float)
        values = np.concatenate([part.get("X_raw", part["X"].astype(float) * scaler.scale + scaler.mean)
                                 for part in parts])
        if values.shape != (len(times), len(FEATURE_COLUMNS)):
            raise CCAError(f"{source} 的数据形状不正确")
        for part in parts:
            if list(part["feature_names"]) != list(FEATURE_COLUMNS):
                raise CCAError(f"{source} 的 NPZ 特征顺序不一致")
        if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise CCAError(f"{source} 时间必须有限且严格递增")
        labels = np.concatenate([part["y_class"] for part in parts])
        if split.startswith("normal_") and (labels != 0).any():
            raise CCAError(f"{split} 包含故障标签，不能用于训练或校准")
        summary = sources[source]
        intervals = summary.get("sample_interval_seconds")
        if len(times) > 1:
            step = float(np.median(np.diff(times)) * 3600)
        elif intervals:
            step = float(intervals[0])
        else:
            raise CCAError(f"{source} 至少需要两个时间点或明确的采样间隔")
        if not np.isfinite(step) or step <= 0:
            raise CCAError(f"{source} 无法确定采样间隔")
        selections = summary.get("normal_groups", []) if split.startswith("normal_") else [summary]
        selection = next(item for item in selections if item["split"] == split)
        return Trajectory(source, split, values, (values - scaler.mean) / scaler.scale, times,
                          int(selection.get("row_start", 0)), step, labels)

    for block in iter_prepared_chunks(Path(data), split):
        source = str(block["source"])
        if source_filter and source_filter not in source:
            continue
        if previous_source is not None and source != previous_source:
            yield assemble(previous_source, blocks)
            blocks = []
        previous_source = source
        blocks.append(block)
    if blocks:
        yield assemble(previous_source, blocks)


def make_pairs(trajectory: Trajectory, history_indices: list[int], target_indices: list[int],
               window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每段连续有效数据内构造 P(t) -> y(t+1)，历史按时间、变量顺序展开。"""
    selected = trajectory.scaled[:, sorted(set(history_indices + target_indices))]
    valid = np.isfinite(selected).all(axis=1)
    dt = np.diff(trajectory.time_h) * 3600
    continuous = np.isclose(dt, trajectory.sample_seconds, rtol=1e-5, atol=1e-6)
    breaks = np.r_[True, ~continuous | ~valid[:-1] | ~valid[1:], True]
    boundaries = np.flatnonzero(breaks)
    histories, futures, indices = [], [], []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if not valid[start] or stop - start <= window:
            continue
        values = trajectory.scaled[start:stop, history_indices]
        windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)[:-1]
        histories.append(windows.transpose(0, 2, 1).reshape(len(windows), -1))
        sample_indices = np.arange(start + window, stop)
        futures.append(trajectory.scaled[sample_indices][:, target_indices])
        indices.append(sample_indices)
    if not histories:
        return (np.empty((0, window * len(history_indices))),
                np.empty((0, len(target_indices))), np.empty(0, dtype=int))
    return np.concatenate(histories), np.concatenate(futures), np.concatenate(indices)


def covariance_whitener(covariance: np.ndarray, samples: int) -> tuple[np.ndarray, int]:
    """经典 CCA 的有效秩白化。零特征值方向移除，不添加岭惩罚。"""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    cutoff = max(samples, len(eigenvalues)) * np.finfo(float).eps * max(float(eigenvalues[-1]), 0)
    keep = eigenvalues > cutoff
    return eigenvectors[:, keep] / np.sqrt(eigenvalues[keep]), int(keep.sum())


@dataclass
class CCAModel:
    spec: ObjectSpec
    window: int
    p_mean: np.ndarray
    A: np.ndarray
    D: np.ndarray
    B: np.ndarray
    c: np.ndarray
    correlations: np.ndarray
    p_rank: int
    f_rank: int
    training_rows: int
    center: np.ndarray
    quantile: np.ndarray
    threshold: np.ndarray
    floor: np.ndarray

    @classmethod
    def fit(cls, spec: ObjectSpec, window: int, history: np.ndarray, future: np.ndarray,
            components: int | None = None) -> CCAModel:
        if len(history) < 3 or len(history) != len(future):
            raise CCAError(f"{spec.name} 至少需要 3 个有效训练配对")
        p_mean, f_mean = history.mean(axis=0), future.mean(axis=0)
        p, f = history - p_mean, future - f_mean
        denominator = len(p) - 1
        wp, rank_p = covariance_whitener(p.T @ p / denominator, len(p))
        wf, rank_f = covariance_whitener(f.T @ f / denominator, len(f))
        available = min(rank_p, rank_f)
        if components is not None and (components < 1 or components > available):
            raise CCAError(f"{spec.name} 的成分数需在 1 与 {available} 之间")
        count = available if components is None else components
        if available:
            left, singular, right = np.linalg.svd(wp.T @ (p.T @ f / denominator) @ wf,
                                                 full_matrices=False)
            A, D = (wp @ left[:, :count]).T, (wf @ right[:count].T).T
            z = p @ A.T
            B = np.linalg.lstsq(z, f, rcond=None)[0].T
            correlations = singular[:count]
        else:
            A, D = np.empty((0, p.shape[1])), np.empty((0, f.shape[1]))
            B, correlations = np.empty((f.shape[1], 0)), np.empty(0)
        # z 在训练组中心化，截距为 F 的训练均值。
        zeros = np.zeros(f.shape[1])
        return cls(spec, window, p_mean, A, D, B, f_mean, correlations, rank_p, rank_f,
                   len(p), zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy())

    def predict(self, history: np.ndarray) -> np.ndarray:
        return ((history - self.p_mean) @ self.A.T) @ self.B.T + self.c

    def calibrate(self, residual: np.ndarray, quantile: float, physical_scale: np.ndarray) -> None:
        if not len(residual) or not np.isfinite(residual).all():
            raise CCAError(f"{self.spec.name} 缺少有限的正常校准残差")
        self.center = np.median(residual, axis=0)
        self.quantile = np.quantile(np.abs(residual - self.center), quantile, axis=0)
        # q=0 时保存原 q，并用浮点精度容差处理除零；不将极小数值噪声放大为无穷分数。
        self.floor = 64 * np.finfo(float).eps * np.maximum(1, physical_scale)
        self.threshold = np.maximum(self.quantile, self.floor)

    def score(self, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if (self.threshold <= 0).any():
            raise CCAError(f"{self.spec.name} 尚未校准")
        deviation = residual - self.center
        return np.abs(deviation) / self.threshold, np.sign(deviation).astype(np.int8)

    def save(self, folder: Path) -> None:
        folder.mkdir()
        metadata = {"spec": asdict(self.spec), "window": self.window, "p_rank": self.p_rank,
                    "f_rank": self.f_rank, "training_rows": self.training_rows}
        write_json(folder / "metadata.json", metadata)
        np.savez_compressed(folder / "parameters.npz", **{name: getattr(self, name) for name in
                            ("p_mean", "A", "D", "B", "c", "correlations", "center", "quantile", "threshold", "floor")})

    @classmethod
    def load(cls, folder: Path) -> CCAModel:
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        metadata["spec"] = ObjectSpec(**metadata["spec"])
        with np.load(folder / "parameters.npz", allow_pickle=False) as archive:
            return cls(**metadata, **{key: archive[key] for key in archive.files})


def exceedance_runs(exceed: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """连续有效超限段 [start, stop)；有效预测之间的缺口会重置持续计数。"""
    if not len(exceed):
        return np.empty(0, int), np.empty(0, int)
    joined = np.diff(indices) == 1
    starts = np.flatnonzero(exceed & np.r_[True, ~exceed[:-1] | ~joined])
    stops = np.flatnonzero(exceed & np.r_[~exceed[1:] | ~joined, True]) + 1
    return starts, stops


def alarm_mask(scores: np.ndarray, indices: np.ndarray, consecutive: int) -> np.ndarray:
    if consecutive < 1:
        raise CCAError("持续采样点数必须至少为 1")
    active = np.zeros(scores.shape, dtype=bool)
    for j in range(scores.shape[1]):
        starts, stops = exceedance_runs(scores[:, j] > 1, indices)
        for start, stop in zip(starts, stops):
            if stop - start >= consecutive:
                active[start + consecutive - 1:stop, j] = True
    return active


def model_predictions(model: CCAModel, trajectory: Trajectory, scaler: NormalScaler) -> dict:
    order = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
    history_indices = [order[name] for name in model.spec.history_ids]
    target_indices = [order[name] for name in model.spec.target_ids]
    p, _, indices = make_pairs(trajectory, history_indices, target_indices, model.window)
    prediction = model.predict(p) * scaler.scale[target_indices] + scaler.mean[target_indices]
    actual = trajectory.raw[indices][:, target_indices]
    return {"indices": indices, "actual": actual, "prediction": prediction,
            "residual": actual - prediction}


def alarm_events(model: CCAModel, trajectory: Trajectory, values: dict, scores: np.ndarray,
                 direction: np.ndarray, consecutive: int) -> Iterator[dict]:
    """事件数值对应确认时刻；起点可按 source_row 回查逐点 NPZ，不引用确认后的观测。"""
    indices = values["indices"]
    for j, variable_id in enumerate(model.spec.target_ids):
        starts, stops = exceedance_runs(scores[:, j] > 1, indices)
        for start, stop in zip(starts, stops):
            if stop - start < consecutive:
                continue
            confirm = start + consecutive - 1
            point = int(indices[confirm])
            onset = int(indices[start])
            sign = int(direction[confirm, j])
            yield {
                "source": trajectory.source, "split": trajectory.split,
                "variable_id": variable_id, "variable_name": model.spec.variable_names[j],
                "object_name": model.spec.name, "object_kind": model.spec.kind,
                "module_names": model.spec.modules,
                "module_status": "explicit_graph_membership" if model.spec.modules else "missing_in_graph",
                "onset_time_h": float(trajectory.time_h[onset]),
                "confirmation_time_h": float(trajectory.time_h[point]),
                "onset_time_seconds": float(trajectory.time_h[onset] * 3600),
                "confirmation_time_seconds": float(trajectory.time_h[point] * 3600),
                "onset_source_row": trajectory.row_offset + onset,
                "confirmation_source_row": trajectory.row_offset + point,
                "direction": "偏高" if sign > 0 else "偏低",
                "direction_basis": "原单位预测残差减去正常校准残差中位数",
                "onset_direction": "偏高" if direction[start, j] > 0 else "偏低",
                "score": float(scores[confirm, j]),
                "actual": float(values["actual"][confirm, j]),
                "prediction": float(values["prediction"][confirm, j]),
                "raw_residual": float(values["residual"][confirm, j]),
                "centered_residual": float(values["residual"][confirm, j] - model.center[j]),
                "residual_center": float(model.center[j]),
                "threshold": float(model.threshold[j]), "unit": model.spec.units[j],
                "consecutive_required": consecutive,
            }


def evaluation_report(totals: dict, object_reports: list[dict], selected: int) -> dict:
    """用同一统计口径生成单条轨迹及整批数据的报告。"""
    rules = {}
    for k, item in totals.items():
        rules[str(k)] = {
            **item,
            "false_alarm_rate": (item["false_alarm_points"] / item["normal_points"]
                                 if item["normal_points"] else None),
            "false_alarm_events_per_hour": (item["false_alarm_events"] / item["normal_hours"]
                                            if item["normal_hours"] else None),
        }
    return {"selected_consecutive": selected, "rules": rules, "objects": object_reports,
            "false_alarm_definition": "正常有效时刻任一对象任一变量处于已确认报警的比例；持续确认前不计报警",
            "event_definition": "整体报警掩码的连续片段；非正常样本、无有效预测或时间缺口切断事件"}


def evaluate(models: list[CCAModel], trajectories: Iterable[Trajectory], scaler: NormalScaler,
             consecutive_values: list[int], selected: int, output: Path | None = None,
             *, per_trajectory_output: bool = False) -> dict:
    """固定模型计算残差与报警；故障模式在每条轨迹完成后独立发布第二层输入。"""
    candidates = sorted(set(consecutive_values + [selected]))
    totals = {k: {"valid_points": 0, "normal_points": 0, "false_alarm_points": 0,
                  "false_alarm_events": 0, "normal_hours": 0.0} for k in candidates}
    object_reports = []
    trajectory_reports = []
    trajectory_count = 0
    marker = None
    with ExitStack() as batch_files:
        batch_event_file = None
        if output is not None:
            output.mkdir(parents=True)
            marker = output / "INCOMPLETE.txt"
            marker.write_text("残差和事件尚未完整导出。", encoding="utf-8")
            if not per_trajectory_output:
                batch_event_file = batch_files.enter_context((output / "events.jsonl").open("w", encoding="utf-8"))
        for trajectory in trajectories:
            trajectory_count += 1
            combined = {k: np.zeros(len(trajectory.time_h), bool) for k in candidates}
            eligible = np.zeros(len(trajectory.time_h), bool)
            case_objects = []
            point_files = []
            event_count = 0
            destination = None
            case_marker = None
            with ExitStack() as case_files:
                event_file = batch_event_file
                if output is not None:
                    destination = output / Path(trajectory.source).stem
                    destination.mkdir()
                    if per_trajectory_output:
                        case_marker = destination / "INCOMPLETE.txt"
                        case_marker.write_text("本轨迹的残差和事件正在导出。", encoding="utf-8")
                        event_file = case_files.enter_context((destination / "events.jsonl").open("w", encoding="utf-8"))
                        print(f"[{trajectory_count}] 开始检测：{trajectory.source}", flush=True)
                for number, model in enumerate(models):
                    values = model_predictions(model, trajectory, scaler)
                    indices = values["indices"]
                    if not len(indices):
                        case_objects.append({"source": trajectory.source, "object_name": model.spec.name,
                                             "valid_points": 0, "reason": "没有完整的连续有效窗口"})
                        continue
                    eligible[indices] = True
                    scores, direction = model.score(values["residual"])
                    normal = trajectory.labels[indices] == 0
                    masks = {}
                    rules = {}
                    for k in candidates:
                        active = alarm_mask(scores, indices, k)
                        masks[k] = active
                        combined[k][indices] |= active.any(axis=1)
                        rules[str(k)] = {
                            "normal_variable_points": int(normal.sum() * scores.shape[1]),
                            "false_alarm_variable_points": int(active[normal].sum()),
                            "normal_object_points": int(normal.sum()),
                            "false_alarm_object_points": int(active[normal].any(axis=1).sum()),
                        }
                    case_objects.append({"source": trajectory.source, "object_name": model.spec.name,
                                         "target_ids": model.spec.target_ids, "valid_points": len(indices),
                                         "raw_exceedance_points_by_variable": (scores > 1).sum(axis=0).tolist(),
                                         "rules": rules})
                    if destination is not None:
                        filename = f"object_{number:03d}.npz"
                        np.savez_compressed(destination / filename,
                            time_h=trajectory.time_h[indices], source_rows=indices + trajectory.row_offset,
                            variable_ids=np.array(model.spec.target_ids), units=np.array(model.spec.units),
                            object_name=np.array(model.spec.name), module_names=np.array(model.spec.modules),
                            actual=values["actual"], prediction=values["prediction"],
                            raw_residual=values["residual"], centered_residual=values["residual"] - model.center,
                            score=scores, direction=direction, exceed=scores > 1, alarm=masks[selected],
                            residual_center=model.center, threshold=model.threshold,
                            fault_label=trajectory.labels[indices])
                        point_files.append({"object_name": model.spec.name, "file": filename})
                        for event in alarm_events(model, trajectory, values, scores, direction, selected):
                            event_file.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                            event_count += 1
            normal = eligible & (trajectory.labels == 0)
            case_totals = {}
            time_break = np.r_[True, ~np.isclose(np.diff(trajectory.time_h) * 3600,
                                               trajectory.sample_seconds, rtol=1e-5, atol=1e-6)]
            for k in candidates:
                active = combined[k] & normal
                # 时间缺口及非正常标签均切断正常误报事件。
                starts = active & (np.r_[True, ~active[:-1]] | time_break)
                case_totals[k] = {
                    "valid_points": int(eligible.sum()),
                    "normal_points": int(normal.sum()),
                    "false_alarm_points": int(active.sum()),
                    "false_alarm_events": int(starts.sum()),
                    "normal_hours": float(normal.sum() * trajectory.sample_seconds / 3600),
                }
                for key, value in case_totals[k].items():
                    totals[k][key] += value
            if per_trajectory_output and destination is not None:
                case_report = evaluation_report(case_totals, case_objects, selected)
                case_report.update(source=trajectory.source, split=trajectory.split, alarm_event_count=event_count)
                write_json(destination / "report.json", case_report)
                completion = {"status": "complete", "source": trajectory.source, "split": trajectory.split,
                              "events_file": "events.jsonl", "report_file": "report.json",
                              "point_files": point_files, "alarm_event_count": event_count,
                              "selected_consecutive": selected}
                # 先关闭本轨迹的所有数据文件，再原子发布完成标记，供第二层独立读取。
                completion_temp = destination / "COMPLETE.json.tmp"
                write_json(completion_temp, completion)
                completion_temp.replace(destination / "COMPLETE.json")
                case_marker.unlink()
                trajectory_reports.append({"source": trajectory.source, "split": trajectory.split,
                                           "directory": destination.name, "alarm_event_count": event_count})
                print(f"[{trajectory_count}] 已保存第二层数据：{destination}（{event_count} 个报警事件）", flush=True)
            else:
                object_reports.extend(case_objects)
    if not trajectory_count:
        raise CCAError("所选数据分区或 source 没有轨迹")
    result = evaluation_report(totals, object_reports, selected)
    if per_trajectory_output:
        result["output_layout"] = "per_trajectory"
        result["trajectories"] = trajectory_reports
    if output is not None:
        write_json(output / "report.json", result)
        marker.unlink()
    return result


def train(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise CCAError(f"输出目录已存在，请指定新的 --output：{args.output}")
    args.output.mkdir(parents=True)
    marker = args.output / "INCOMPLETE.txt"
    marker.write_text("模型训练或导出未完成。", encoding="utf-8")
    scaler = NormalScaler.load(args.data / "scaler.json")
    report = json.loads((args.data / "report.json").read_text(encoding="utf-8"))
    if scaler.n_samples != report["normal_split"]["train_rows"]:
        raise CCAError("标准化拟合样本数与正常训练分区不一致")
    print("读取指定 dump 的隔离副本……", flush=True)
    graph = read_dump(args.graph_dump, args.neo4j_home)
    write_json(args.output / "graph_snapshot.json", graph)
    specs, audit = graph_objects(graph, list(FEATURE_COLUMNS), report["excluded_feature_names"])
    write_json(args.output / "graph_mapping.json", {"objects": [asdict(spec) for spec in specs], **audit})
    training = list(read_trajectories(args.data, "normal_train", scaler))
    calibration = list(read_trajectories(args.data, "normal_calibration", scaler))
    if not training or not calibration:
        raise CCAError("正常训练组和正常校准组都必须包含轨迹")
    index = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
    models, summaries = [], []
    for number, spec in enumerate(specs):
        hi, ti = [index[name] for name in spec.history_ids], [index[name] for name in spec.target_ids]
        paired = [make_pairs(trajectory, hi, ti, args.window) for trajectory in training]
        p, f = np.concatenate([item[0] for item in paired]), np.concatenate([item[1] for item in paired])
        model = CCAModel.fit(spec, args.window, p, f, args.components)
        training_error = (f - model.predict(p)) * scaler.scale[ti]
        rmse = np.sqrt(np.mean(training_error ** 2, axis=0))
        del paired, p, f, training_error
        residuals = [model_predictions(model, trajectory, scaler)["residual"] for trajectory in calibration]
        physical_scale = np.maximum(np.abs(scaler.mean[ti]), scaler.scale[ti])
        model.calibrate(np.concatenate(residuals), args.quantile, physical_scale)
        models.append(model)
        summaries.append({"object_name": spec.name, "history_ids": spec.history_ids,
                          "target_ids": spec.target_ids, "p_rank": model.p_rank,
                          "f_rank": model.f_rank, "components": len(model.A),
                          "training_pairs": model.training_rows, "training_rmse_raw_units": rmse.tolist(),
                          "canonical_correlations": model.correlations.tolist(),
                          "residual_center": model.center.tolist(), "q_quantile": model.quantile.tolist(),
                          "effective_threshold": model.threshold.tolist(),
                          "threshold_floor_used": (model.quantile < model.floor).tolist()})
        print(f"[{number + 1}/{len(specs)}] {spec.name}: P={args.window * len(hi)}, "
              f"F={len(ti)}, 成分={len(model.A)}, 配对={model.training_rows}", flush=True)
    candidates = sorted(set(args.consecutive_candidates + [args.consecutive]))
    calibration_check = evaluate(models, calibration, scaler, candidates, args.consecutive)
    selected = args.consecutive
    if args.max_calibration_fpr is not None:
        accepted = [k for k in candidates if calibration_check["rules"][str(k)]["false_alarm_rate"] is not None
                    and calibration_check["rules"][str(k)]["false_alarm_rate"] <= args.max_calibration_fpr]
        if not accepted:
            write_json(args.output / "calibration_candidates.json", calibration_check)
            raise CCAError("候选持续次数均未达到目标整体误报率，请扩大候选次数或调整分位数")
        selected = min(accepted)
    for number, model in enumerate(models):
        model.save(args.output / f"object_{number:03d}")
    # 保存原训练组标准化参数；预测不再依赖另一份可能被更新的 scaler.json。
    shutil.copyfile(args.data / "scaler.json", args.output / "scaler.json")
    manifest = {"format_version": FORMAT_VERSION, "data": str(args.data.resolve()),
                "feature_names": list(FEATURE_COLUMNS), "graph": graph["provenance"],
                "window": args.window, "quantile": args.quantile, "consecutive": selected,
                "consecutive_candidates": candidates, "max_calibration_fpr": args.max_calibration_fpr,
                "model_folders": [f"object_{i:03d}" for i in range(len(models))],
                "models": summaries, "graph_audit": audit,
                "calibration_scope": "单一工况；不同工况须分别训练和校准",
                "p_order": "从最早到最近的时刻，每时刻按 history_ids 顺序",
                "source_row_convention": "原 CSV 数据行从 0 开始计数，不包含表头",
                "score_definition": "abs(raw_residual - calibration_median) / effective_q",
                "persistence_definition": "score>1 的连续有效点；允许偏离方向变化，事件同时记录起点和确认点方向"}
    write_json(args.output / "manifest.json", manifest)
    result = evaluate(models, calibration, scaler, candidates, selected, args.output / "normal_calibration")
    print(f"校准组整体误报率：{result['rules'][str(selected)]['false_alarm_rate']:.6%}", flush=True)
    if not args.skip_normal_test:
        normal_test = list(read_trajectories(args.data, "test_faultfree", scaler))
        if not normal_test:
            raise CCAError("缺少独立 test_faultfree；可用 --skip-normal-test 只完成校准检查")
        check = evaluate(models, normal_test, scaler, [selected], selected, args.output / "test_faultfree")
        manifest["normal_test_rule_frozen"] = True
        print(f"独立正常测试组整体误报率：{check['rules'][str(selected)]['false_alarm_rate']:.6%}", flush=True)
    write_json(args.output / "manifest.json", manifest)
    marker.unlink()
    return manifest


def predict(args: argparse.Namespace) -> dict:
    if (args.model / "INCOMPLETE.txt").exists():
        raise CCAError("训练尚未完整结束，不能加载该模型")
    manifest = json.loads((args.model / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != FORMAT_VERSION or manifest["feature_names"] != list(FEATURE_COLUMNS):
        raise CCAError("模型版本或特征顺序不兼容")
    scaler = NormalScaler.load(args.model / "scaler.json")
    models = [CCAModel.load(args.model / folder) for folder in manifest["model_folders"]]
    data_scaler = NormalScaler.load(args.data / "scaler.json")
    if not np.array_equal(data_scaler.mean, scaler.mean) or not np.array_equal(data_scaler.scale, scaler.scale):
        raise CCAError("预测数据的标准化参数与训练模型不同，请使用同一套预处理参数")
    trajectories = read_trajectories(args.data, args.split, scaler, args.source)
    selected = manifest["consecutive"]
    return evaluate(models, trajectories, scaler, [selected], selected, args.output,
                    per_trajectory_output=args.split in {"train_fault", "test_fault"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="从指定 dump 提取图谱，训练、校准并检查正常误报")
    fit.add_argument("--data", type=Path, default=DEFAULT_DATA)
    fit.add_argument("--graph-dump", type=Path, default=DEFAULT_DUMP)
    fit.add_argument("--neo4j-home", type=Path)
    fit.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    fit.add_argument("--window", type=int, default=30)
    fit.add_argument("--components", type=int, default=None, help="默认保留至多 min(rank(P),rank(F)) 个成分")
    fit.add_argument("--quantile", type=float, default=.99)
    fit.add_argument("--consecutive", type=int, default=3)
    fit.add_argument("--consecutive-candidates", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    fit.add_argument("--max-calibration-fpr", type=float, help="可选，按校准组整体逐点误报率选择最小合格持续次数")
    fit.add_argument("--skip-normal-test", action="store_true")
    infer = commands.add_parser("predict", help="使用固定模型和固定报警规则输出残差及第二层事件")
    infer.add_argument("--data", type=Path, default=DEFAULT_DATA)
    infer.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    infer.add_argument("--split", choices=["normal_calibration", "test_faultfree", "test_fault", "train_fault"],
                       default="test_fault")
    infer.add_argument("--source", help="只处理 source 路径包含该字符串的轨迹")
    infer.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "layer1_cca_predictions",
                       help="新建输出目录；故障数据按原始轨迹分别保存第二层事件和逐点结果")
    args = parser.parse_args()
    if args.command == "train":
        if args.window < 1 or not 0 < args.quantile < 1 or args.consecutive < 1:
            parser.error("window、consecutive 必须为正整数，quantile 必须在 0 与 1 之间")
        if not args.consecutive_candidates or min(args.consecutive_candidates) < 1:
            parser.error("候选持续次数必须为正整数")
        if args.max_calibration_fpr is not None and not 0 <= args.max_calibration_fpr <= 1:
            parser.error("max-calibration-fpr 必须在 0 与 1 之间")
    try:
        result = train(args) if args.command == "train" else predict(args)
    except (CCAError, OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"CCA 处理失败：{exc}\n")
    print(f"完成，输出目录：{args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

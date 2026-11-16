"""对 test_fault 的每条轨迹执行第二层溯源测试，并汇总候选排名。

在项目根目录运行（Windows 下先 set PYTHONPATH=src）：

    python -m costep_diagnosis.trace_test --from-cca --output outputs/trace_test_20261116
    python -m costep_diagnosis.trace_test --only fault_01 --max-events 6 --output outputs/try1

报警来源有两种：--predictions 指向第一层产物目录（既支持按轨迹导出的
<trajectory>/events.jsonl，也支持单文件批量导出、按 source 字段切分）；
缺该轨迹的文件时回退到冻结 CCA 现场生成，--from-cca 则直接用冻结 CCA，
不写逐点 NPZ，避免每条轨迹数百 MB 的中间产物。

默认只取故障起始时刻之后的报警，且每个测点只保留首条，避免持续偏移被切成
上万条重复事件。输出目录必须不存在；每条轨迹各自一个子目录，内含 events/
plans/diagnoses 和 summary.json，最后写出 trace_summary.json 与
trace_candidates.csv。

如已有故障-根因对照表（CSV，列为 fault_id,expected_variable），
用 --mapping 指定后可额外统计命中率。
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT / "src"))

from costep_diagnosis import pipeline
from costep_diagnosis.layer2 import llm


DEFAULT_EVENTS = PROJECT / "outputs/cca_run2_test_fault_predictions"
DEFAULT_DATA_ROOT = PROJECT / "data/COSTE_cca_preprocessed/test_fault"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def trajectory_dirs(root: Path) -> list[Path]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"轨迹目录不存在：{root}")
    folders = sorted(folder for folder in root.iterdir()
                     if folder.is_dir() and any(folder.glob("part_*.npz")))
    if not folders:
        raise ValueError(f"{root} 下没有 part_*.npz 轨迹")
    return folders


def fault_start(folder: Path) -> tuple[float | None, int]:
    """读取首个非零标签的时刻与故障编号，不参与任何诊断计算。"""
    times, labels = [], []
    for path in sorted(folder.glob("part_*.npz")):
        with np.load(path, allow_pickle=False) as chunk:
            times.append(np.asarray(chunk["time_h"], dtype=float))
            labels.append(np.asarray(chunk["fault_label"], dtype=np.int64))
    label = np.concatenate(labels)
    time_h = np.concatenate(times)
    hit = np.flatnonzero(label != 0)
    return (float(time_h[hit[0]]) if len(hit) else None), (int(label[hit[0]]) if len(hit) else 0)


def load_events(predictions: Path, name: str) -> list[dict]:
    """优先读按轨迹导出的 events.jsonl，否则从批量文件按 source 过滤。"""
    single = predictions / name / "events.jsonl"
    if single.is_file():
        with single.open(encoding="utf-8-sig") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    batch = predictions / "events.jsonl"
    if not batch.is_file():
        raise ValueError(f"{predictions} 缺少 {name}/events.jsonl 或 events.jsonl")
    rows = []
    with batch.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if name in str(row.get("source", "")):
                rows.append(row)
    if not rows:
        raise ValueError(f"{batch} 中没有 {name} 的报警")
    return rows


def generate_events(folder: Path, cca_model: Path, digest: str) -> list[dict]:
    """用冻结 CCA 直接生成报警，不落逐点 NPZ，避免每条轨迹数百 MB 的中间产物。"""
    from costep_diagnosis.layer1 import cca
    from costep_diagnosis.layer2 import baseline
    runs = baseline.read_physical_runs(folder, cca.FEATURE_COLUMNS)
    if len(runs) != 1:
        raise ValueError(f"{folder} 必须只包含一条轨迹")
    return [alarm for _, alarm in pipeline.generate_alarms(runs[0], cca_model, digest, ())]


def select_events(events: list[dict], onset: float | None, dedup: str,
                  limit: int | None) -> tuple[list[dict], dict]:
    kept, seen = [], set()
    for row in events:
        if onset is not None and float(row.get("onset_time_h", -math.inf)) < onset:
            continue
        key = row.get("variable_id")
        if dedup == "first-per-variable":
            if key in seen:
                continue
            seen.add(key)
        kept.append(row)
    kept.sort(key=lambda row: (row.get("onset_time_h", 0), row.get("variable_id") or ""))
    total = len(kept)
    if limit is not None:
        kept = kept[:limit]
    return kept, {"after_onset": onset, "dedup": dedup, "available": total, "kept": len(kept)}


def rank_rows(diagnoses: Path, trajectory: str, fault_id: int) -> list[dict]:
    rows = []
    for line in diagnoses.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        alarm = row.get("alarm") or {}
        if row.get("template_id") == "TEMPERATURE":
            ranked = [item for item in row.get("candidates", []) if item["status"] == "ready"]
            top = ranked[0] if ranked else None
            top_name = top["candidate"] if top else None
            top_d = top.get("D_j") if top else None
            top_s = top.get("S_j_c") if top else None
            n_ready = len(ranked)
            candidates = [item["candidate"] for item in ranked]
        elif row.get("template_id") == "INGREDIENT":
            exceeded = row.get("exceeded_metrics") or []
            top_name = exceeded[0] if exceeded else None
            top_d = top_s = None
            n_ready = len(exceeded)
            candidates = exceeded
        else:
            top_name = top_d = top_s = None
            n_ready, candidates = 0, []
        rows.append({
            "trajectory": trajectory, "fault_id": fault_id,
            "event_row": row.get("event_row"), "variable_id": alarm.get("variable_id"),
            "object_name": (alarm.get("object_name") or ""),
            "template_id": row.get("template_id"), "scope_id": row.get("scope_id"),
            "plan_id": row.get("plan_id"), "status": row.get("status"),
            "reason": row.get("reason") or "",
            "top_candidate": top_name or "", "top_D_j": top_d, "top_S_j": top_s,
            "n_ranked": n_ready, "candidates": "|".join(candidates),
        })
    return rows


def write_candidates(path: Path, rows: list[dict]) -> None:
    fields = ["trajectory", "fault_id", "event_row", "variable_id", "object_name", "template_id",
              "scope_id", "plan_id", "status", "reason", "top_candidate", "top_D_j", "top_S_j",
              "n_ranked", "candidates"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_mapping(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {}
    for row in rows:
        key = str(row.get("fault_id", "")).strip()
        variable = str(row.get("expected_variable", "")).strip()
        if not key.isdigit() or not variable:
            raise ValueError("对照表需要 fault_id（整数）与 expected_variable 两列")
        mapping[int(key)] = variable
    return mapping


def hit_metrics(rows: list[dict], mapping: dict[int, str]) -> dict:
    scored = [row for row in rows if row["fault_id"] in mapping and row["candidates"]]
    if not scored:
        return {"scored_alarms": 0}
    def split(value):
        return [item for item in (value or "").split("|") if item]
    top1 = sum(1 for row in scored if split(row["candidates"])[:1] == [mapping[row["fault_id"]]])
    any_rank = sum(1 for row in scored if mapping[row["fault_id"]] in split(row["candidates"]))
    return {"scored_alarms": len(scored),
            "hit_at_1": top1 / len(scored), "hit_at_any": any_rank / len(scored)}


def run(args) -> dict:
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"输出目录已存在：{output}")
    predictions = args.predictions.resolve() if args.predictions else None
    root = args.data_root.resolve()
    mapping = load_mapping(args.mapping)
    folders = trajectory_dirs(root)
    if args.only:
        folders = [folder for folder in folders if any(key in folder.name for key in args.only)]
        if not folders:
            raise ValueError(f"--only 没有匹配到轨迹：{args.only}")

    output.mkdir(parents=True)
    marker = output / "INCOMPLETE.txt"
    marker.write_text("溯源测试未完成。\n", encoding="utf-8")
    all_rows, per_trajectory = [], []
    status, templates, reasons = Counter(), Counter(), Counter()
    digest = None
    for number, folder in enumerate(folders, 1):
        onset, fault_id = fault_start(folder)
        events, source = None, "layer1_events"
        try:
            if predictions is not None and not args.from_cca:
                events = load_events(predictions, folder.name)
            if events is None:
                if digest is None:
                    digest = llm.load_graph(
                        llm.DEFAULT_DUMP, llm.DEFAULT_SNAPSHOT)[0]["provenance"]["sha256"]
                events, source = generate_events(folder, args.cca_model, digest), "frozen_cca"
        except (ValueError, OSError, KeyError) as error:
            per_trajectory.append({"trajectory": folder.name, "fault_id": fault_id,
                                   "fault_start_h": onset, "status": "failed", "error": str(error)})
            print(f"[{number}/{len(folders)}] {folder.name}: 取报警失败 {error}", flush=True)
            continue
        kept, audit = select_events(events, onset if args.post_onset_only else None,
                                    args.dedup, args.max_events)
        record = {"trajectory": folder.name, "fault_id": fault_id, "fault_start_h": onset,
                  "event_source": source, "events_total": len(events),
                  "events_selected": len(kept), **audit}
        if not kept:
            record["status"] = "skipped_no_event"
            per_trajectory.append(record)
            print(f"[{number}/{len(folders)}] {folder.name}: 无可用报警，跳过", flush=True)
            continue
        staged = output / "_events" / f"{folder.name}.jsonl"
        staged.parent.mkdir(parents=True, exist_ok=True)
        with staged.open("w", encoding="utf-8") as handle:
            for row in kept:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        destination = output / folder.name
        try:
            summary = pipeline.run_pipeline(
                actual_dir=folder, output=destination, events_path=staged,
                selector_mode=args.selector, max_events=None,
                match_features=args.match_features, exclude_features=args.exclude_features,
                max_distance=args.max_distance, upstream_stages=args.upstream_stages,
                rank_rule=args.rank_rule, composition=not args.no_composition,
            )
        except (ValueError, OSError, KeyError) as error:
            record.update(status="failed", error=str(error))
            per_trajectory.append(record)
            print(f"[{number}/{len(folders)}] {folder.name}: 失败 {error}", flush=True)
            continue
        rows = rank_rows(destination / "diagnoses.jsonl", folder.name, fault_id)
        all_rows.extend(rows)
        status.update(row["status"] for row in rows)
        templates.update(row["template_id"] or "UNSUPPORTED" for row in rows)
        reasons.update(row["reason"] for row in rows if row["reason"])
        ready = sum(1 for row in rows if row["status"] == "ready")
        record.update(status="complete", ready=ready, blocked=len(rows) - ready,
                      by_template=dict(summary.get("by_template", {})))
        per_trajectory.append(record)
        print(f"[{number}/{len(folders)}] {folder.name}: {len(rows)} 条报警，"
              f"ready {ready}，blocked {len(rows) - ready}", flush=True)

    write_candidates(output / "trace_candidates.csv", all_rows)
    result = {
        "complete": True, "data_root": str(root), "predictions": str(predictions),
        "selector": args.selector, "dedup": args.dedup, "max_events": args.max_events,
        "post_onset_only": args.post_onset_only,
        "composition_candidates": not args.no_composition,
        "normal_matching": {"match_features": args.match_features,
                            "exclude_features": args.exclude_features,
                            "max_distance": args.max_distance},
        "trajectory_count": len(folders), "alarm_count": len(all_rows),
        "by_status": dict(status), "by_template": dict(templates), "blocked_reasons": dict(reasons),
        "mapping_rows": len(mapping), "trajectories": per_trajectory,
        "files": {"candidates": "trace_candidates.csv"},
    }
    if mapping:
        result["hits"] = hit_metrics(all_rows, mapping)
    write_json(output / "trace_summary.json", result)
    marker.unlink()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="新建本次溯源测试目录")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_EVENTS,
                        help="第一层事件目录；缺该轨迹时回退到冻结 CCA")
    parser.add_argument("--from-cca", action="store_true", help="忽略第一层产物，直接用冻结 CCA 生成报警")
    parser.add_argument("--cca-model", type=Path, default=pipeline.DEFAULT_CCA_MODEL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="待诊断轨迹根目录")
    parser.add_argument("--only", nargs="+", help="只跑名字包含这些片段的轨迹")
    parser.add_argument("--dedup", choices=("first-per-variable", "none"), default="first-per-variable")
    parser.add_argument("--max-events", type=int, help="每条轨迹最多处理多少条报警（默认全部）")
    parser.add_argument("--all-events", dest="post_onset_only", action="store_false",
                        help="不做故障起始时刻过滤")
    parser.add_argument("--selector", choices=("auto", "rules", "llm"), default="rules")
    parser.add_argument("--match-features", nargs="+", default=())
    parser.add_argument("--exclude-features", nargs="+", default=())
    parser.add_argument("--max-distance", type=float)
    parser.add_argument("--upstream-stages", type=int, default=3,
                        help="第二层候选池沿工艺流反向回溯的设备级跳数；0 表示只用报警点所属设备")
    parser.add_argument("--rank-rule", choices=("alarm-first", "dj"), default="alarm-first")
    parser.add_argument("--no-composition", action="store_true",
                        help="不把上游流股的组成指标并入温度候选池")
    parser.add_argument("--mapping", type=Path, help="CSV：fault_id,expected_variable，可选")
    args = parser.parse_args()
    if args.max_events is not None and args.max_events < 1:
        parser.error("--max-events 必须为正整数")
    try:
        result = run(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"溯源测试失败：{error}\n")
    hits = result.get("hits") or {}
    print(f"完成：{result['trajectory_count']} 条轨迹，{result['alarm_count']} 条报警")
    print(json.dumps(result["by_status"], ensure_ascii=False))
    if hits.get("scored_alarms"):
        print(f"命中率：top1 {hits['hit_at_1']:.2%}，任意排名 {hits['hit_at_any']:.2%}"
              f"（{hits['scored_alarms']} 条可评分报警）")


if __name__ == "__main__":
    main()

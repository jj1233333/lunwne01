"""按故障-根因对照表评估第二层溯源的命中率。

只读已有的溯源结果目录，不重跑诊断，因此改阈值、换候选池或补对照表后
可以秒级复评，不必重跑几分钟的推理。

    set PYTHONPATH=src
    python -m costep_diagnosis.score_trace --result outputs/trace_test_20261116 ^
        --mapping data/kg/COSTEP_fault_mapping.csv

对照表最少需要 fault_id 与 expected_variables 两列，expected_variables 用
分号分隔（温度类写变量 ID，组成类写指标键如 S4.A/C），expected_scope 可选。
命中判定：温度类看候选排名里是否出现期望变量；组成类看 exceeded_metrics
是否与期望指标相交，因为组成诊断只输出超限清单、不输出候选排名。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = PROJECT / "outputs/trace_test_20261116"
DEFAULT_MAPPING = PROJECT / "data/kg/COSTEP_fault_mapping.csv"


def load_mapping(path: Path) -> dict[int, dict]:
    """读取对照表，返回 fault_id → 期望变量集合与方案范围。"""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"对照表为空：{path}")
    mapping = {}
    for row in rows:
        key = str(row.get("fault_id", "")).strip()
        if not key.isdigit():
            continue
        variables = [item.strip() for item in str(row.get("expected_variables", "")).split(";")
                     if item.strip()]
        flag = (row.get("in_test_scope") or "yes").strip().lower()
        mapping[int(key)] = {
            "variables": variables,
            "scope": (row.get("expected_scope") or "").strip(),
            "form": (row.get("form") or "").strip(),
            "name": (row.get("name") or "").strip(),
            "reachable": (row.get("reachable") or "").strip(),
            "in_test_scope": flag not in {"no", "n", "否", "false", "0"},
        }
    if not mapping:
        raise ValueError(f"对照表没有有效的 fault_id 行：{path}")
    return mapping


def fault_id_of(trajectory: str) -> int | None:
    """从 fault_07_20261107_xxx 这类目录名取故障编号。"""
    head = trajectory.split("_")
    if len(head) >= 2 and head[0] == "fault" and head[1].isdigit():
        return int(head[1])
    return None


def ranked_candidates(record: dict) -> list[str]:
    """温度方案的候选顺序：优先用诊断记录里已排好的 rank。"""
    rows = [item for item in record.get("candidates") or [] if item.get("status") == "ready"]
    if any(item.get("rank") is not None for item in rows):
        rows = sorted(rows, key=lambda item: (item.get("rank") is None, item.get("rank") or 0))
    elif any(item.get("D_j") is not None for item in rows):
        rows = sorted(rows, key=lambda item: -(item.get("D_j") or 0.0))
    return [str(item.get("candidate")) for item in rows if item.get("candidate")]


def score_record(record: dict, spec: dict) -> dict:
    """对单条诊断记录判定命中情况。"""
    expected = spec["variables"]
    template = record.get("template_id")
    result = {"expected": "|".join(expected), "rank": 0, "hit_at_1": False,
              "hit_at_any": False, "scope_hit": False}
    if spec["scope"] and record.get("scope_id"):
        result["scope_hit"] = spec["scope"].endswith(f":{record['scope_id']}")
    if template == "INGREDIENT":
        exceeded = [str(item) for item in record.get("exceeded_metrics") or []]
        hit = [name for name in exceeded if name in expected]
        result.update(rank=1 if hit else 0, hit_at_1=bool(hit), hit_at_any=bool(hit))
        result["exceeded"] = "|".join(exceeded)
        return result
    names = ranked_candidates(record)
    result["top"] = names[0] if names else ""
    for position, name in enumerate(names, 1):
        if name in expected:
            result.update(rank=position, hit_at_any=True, hit_at_1=position == 1)
            break
    return result


def collect(result_dir: Path) -> list[dict]:
    """遍历结果目录，拼出可评分的行。"""
    rows = []
    for folder in sorted(result_dir.glob("fault_*")):
        path = folder / "diagnoses.jsonl"
        if not path.is_file():
            continue
        fault = fault_id_of(folder.name)
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                rows.append({"trajectory": folder.name, "fault_id": fault,
                             "record": record})
    return rows


def summarise(rows: list[dict], mapping: dict[int, dict]) -> dict:
    """汇总命中率，并区分“覆盖不足”与“排序错误”。"""
    scored, detail = [], []
    coverage = defaultdict(lambda: {"alarms": 0, "ready": 0})
    for row in rows:
        record, fault = row["record"], row["fault_id"]
        status = record.get("status")
        bucket = coverage[fault]
        bucket["alarms"] += 1
        if status == "ready":
            bucket["ready"] += 1
        spec = mapping.get(fault)
        if status != "ready" or spec is None or not spec["variables"]:
            continue
        outcome = score_record(record, spec)
        base = {"trajectory": row["trajectory"], "fault_id": fault,
                "fault_name": spec["name"], "form": spec["form"],
                "variable_id": (record.get("alarm") or {}).get("variable_id"),
                "event_row": record.get("event_row"),
                "template_id": record.get("template_id"), "scope_id": record.get("scope_id"),
                "reachable": spec["reachable"], "in_test_scope": spec["in_test_scope"]}
        detail.append({**base, **outcome})
        scored.append({**base, **outcome})

    total = len(scored)
    hit1 = sum(1 for item in detail if item["hit_at_1"])
    hit_any = sum(1 for item in detail if item["hit_at_any"])
    by_fault = defaultdict(lambda: {"alarms": 0, "ready": 0, "scored": 0, "hit1": 0, "hit_any": 0,
                                    "scope_hit": 0})
    for fault, bucket in coverage.items():
        by_fault[fault].update(bucket)
    for item in detail:
        bucket = by_fault[item["fault_id"]]
        bucket["scored"] += 1
        bucket["hit1"] += item["hit_at_1"]
        bucket["hit_any"] += item["hit_at_any"]
        bucket["scope_hit"] += item["scope_hit"]

    by_form = defaultdict(lambda: {"scored": 0, "hit1": 0})
    for item in detail:
        by_form[item["form"] or "未知"]["scored"] += 1
        by_form[item["form"] or "未知"]["hit1"] += item["hit_at_1"]

    unreachable = Counter()
    for item in detail:
        if item["reachable"] == "no":
            unreachable[item["fault_id"]] += 1

    by_scope = defaultdict(lambda: {"scored": 0, "hit1": 0, "faults": set()})
    for item in detail:
        bucket = by_scope["范围内" if item["in_test_scope"] else "范围外"]
        bucket["scored"] += 1
        bucket["hit1"] += item["hit_at_1"]
        bucket["faults"].add(item["fault_id"])
    outside = sorted(fault for fault, spec in mapping.items() if not spec["in_test_scope"])
    inside = sorted(fault for fault, spec in mapping.items() if spec["in_test_scope"])

    return {"scored_alarms": total,
            "hit_at_1": hit1 / total if total else 0.0,
            "hit_at_any": hit_any / total if total else 0.0,
            "by_fault": {key: dict(value) for key, value in sorted(by_fault.items())},
            "by_form": {key: dict(value) for key, value in by_form.items()},
            "by_scope": {key: {**value, "faults": sorted(value["faults"])}
                         for key, value in by_scope.items()},
            "faults_in_scope": inside, "faults_outside_scope": outside,
            "unreachable_alarms": dict(unreachable), "detail": detail}


def write_detail(path: Path, detail: list[dict]) -> None:
    fields = ["trajectory", "fault_id", "fault_name", "form", "variable_id", "event_row",
              "template_id", "scope_id", "reachable", "in_test_scope", "expected", "rank",
              "hit_at_1", "hit_at_any", "scope_hit"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(detail)


def report(summary: dict, mapping: dict[int, dict]) -> None:
    inside, outside = summary["faults_in_scope"], summary["faults_outside_scope"]
    print(f"对照表 {len(mapping)} 条故障：测试范围 {len(inside)} 条"
          f"（{','.join(str(item) for item in inside)}）"
          f"，标黄排除 {len(outside)} 条（{','.join(str(item) for item in outside)}）")
    print(f"可评分报警 {summary['scored_alarms']} 条："
          f"top1 命中 {summary['hit_at_1']:.1%}，任意排名命中 {summary['hit_at_any']:.1%}")
    for name, bucket in summary["by_scope"].items():
        rate = bucket["hit1"] / bucket["scored"] if bucket["scored"] else 0.0
        print(f"  {name}：{len(bucket['faults'])} 条故障，可评分 {bucket['scored']} 条，top1 {rate:.0%}")
    if not summary["scored_alarms"]:
        return
    print()
    print(f"{'fault':7}{'名称':26}{'形态':7}{'报警':6}{'可溯源':7}{'可评分':7}{'top1':7}{'命中率'}")
    for fault, bucket in summary["by_fault"].items():
        spec = mapping.get(fault) or {}
        name = spec.get("name") or "未登记"
        rate = bucket["hit1"] / bucket["scored"] if bucket["scored"] else 0.0
        print(f"{fault:<7}{name[:24]:26}{spec.get('form','')[:5]:7}{bucket['alarms']:<6}"
              f"{bucket['ready']:<7}{bucket['scored']:<7}{bucket['hit1']:<7}{rate:.0%}")
    print()
    for form, bucket in summary["by_form"].items():
        rate = bucket["hit1"] / bucket["scored"] if bucket["scored"] else 0.0
        print(f"{form}：可评分 {bucket['scored']} 条，top1 命中率 {rate:.0%}")
    if summary["unreachable_alarms"]:
        detail = "、".join(f"fault_{key:02d}（{count} 条）"
                          for key, count in sorted(summary["unreachable_alarms"].items()))
        print(f"\n结构上无法命中：{detail}，候选池里不存在对应变量。")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按对照表评估溯源命中率")
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT, help="溯源结果目录")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING, help="故障-根因对照表")
    parser.add_argument("--output", type=Path, help="明细 CSV 路径，默认写到结果目录")
    args = parser.parse_args(argv)

    result_dir = args.result.resolve()
    if not result_dir.is_dir():
        raise SystemExit(f"结果目录不存在：{result_dir}")
    mapping = load_mapping(args.mapping.resolve())
    rows = collect(result_dir)
    if not rows:
        raise SystemExit(f"{result_dir} 下没有 diagnoses.jsonl")
    summary = summarise(rows, mapping)
    report(summary, mapping)
    if summary["detail"]:
        target = (args.output or result_dir / "trace_hits.csv").resolve()
        write_detail(target, summary["detail"])
        print(f"\n明细已写出：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Turn CCA alarms into graph-verified temperature or composition plans.

Temperature plans always bind the device that watches the alarm point. With
--upstream-stages above zero the plan also carries `upstream_plans`: devices
reachable by walking against the process flow (管线 -输入-> 设备 -输出-> 管线),
so a fault that travelled downstream can still borrow the upstream ARX model.
Only incoming edges are followed, so downstream vessels never become suspects.

The same window yields `composition_plans`: the trained feed streams (S1-S4)
that reach this device, scored by their frozen ratio thresholds. A temperature
pool holds measured columns only, so without them a root cause named S4.A or
S4.A/C could never appear in the candidate list.

Examples from the project root:
  .venv/Scripts/python.exe -X utf8 src/costep_diagnosis/layer2/llm.py --variable-id SM_019
  .venv/Scripts/python.exe -X utf8 src/costep_diagnosis/layer2/llm.py --events events.jsonl --output plans.jsonl

The optional model selects a plan ID from graph-verified options. Configure
COSTEP_LLM_BASE_URL, COSTEP_LLM_MODEL and COSTEP_LLM_API_KEY, then use
--selector llm for an OpenAI-compatible endpoint. The default auto mode uses
local rules if no endpoint is configured. No fault labels enter the selector.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

PROJECT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT / "src"))

from costep_diagnosis.data_convert.data import FEATURE_COLUMNS
from costep_diagnosis.layer1.cca import CCAError, DEFAULT_DUMP, read_dump
from costep_diagnosis.layer2 import ingredient, temperature


DEFAULT_SNAPSHOT = PROJECT / "outputs/cca_run2/graph_snapshot.json"
DEFAULT_INGREDIENT_MODEL = PROJECT / "outputs/layer2_ingredient/parameters.json"
DEFAULT_TEMPERATURE_MODEL = PROJECT / "outputs/layer2_temperature/parameters.json"
COMPOSITION_QUANTITY = "摩尔百分含量"
TEMPERATURE_QUANTITY = "温度"
MEASUREMENT_ID = re.compile(r"(?:SM_\d{3}|XMEAS_\d{2})")
PHYSICAL_ID = re.compile(r"(?:SM_\d{3}|XMEAS_\d{2}|XMV_\d{2})")
ALIAS_IDS = re.compile(r"复用\s*[:：]\s*((?:XMEAS_\d{2}|SM_\d{3}))(?![\d])")
LABEL_FIELDS = {"检测变量": "变量ID", "操作参数": "参数ID",
                "设备": "设备名称", "管线": "管线名称", "模块": "模块名称"}
NUMERIC_ALARM_FIELDS = ("onset_time_h", "confirmation_time_h", "onset_source_row",
                        "confirmation_source_row", "score", "actual", "prediction",
                        "raw_residual", "centered_residual", "threshold")


class PlanError(ValueError):
    """An alarm, graph, artifact or model selection is invalid."""


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise PlanError(f"需要 JSON 对象：{path}")
    return data


def _unit(unit):
    return "°C" if unit in ("℃", "degC") else unit


def _alarm_fields(event: Mapping) -> dict:
    signal = event.get("variable_id")
    if not isinstance(signal, str) or not MEASUREMENT_ID.fullmatch(signal):
        raise PlanError("报警点需为测点 ID，例如 SM_019 或 XMEAS_09")
    result = {"variable_id": signal}
    for key in NUMERIC_ALARM_FIELDS:
        if key in event:
            value = event[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise PlanError(f"报警字段 {key} 需要有限数值")
            result[key] = value
    for key in ("unit", "direction", "onset_direction"):
        if key in event and isinstance(event[key], str):
            result[key] = _unit(event[key]) if key == "unit" else event[key]
    if result.get("onset_time_h", -math.inf) > result.get("confirmation_time_h", math.inf):
        raise PlanError("确认时刻早于报警起点")
    return result


def load_graph(dump: Path, snapshot: Path | None = DEFAULT_SNAPSHOT, *,
               neo4j_home: Path | None = None, refresh: bool = False) -> tuple[dict, str]:
    """Reuse a matching snapshot of this dump or extract it via the CCA reader."""
    digest = hashlib.sha256(dump.read_bytes()).hexdigest()
    if not refresh and snapshot is not None and snapshot.is_file():
        graph = _read_json(snapshot)
        if graph.get("provenance", {}).get("sha256", "").lower() == digest:
            return graph, "verified_cca_snapshot"
    graph = read_dump(dump, neo4j_home)
    if graph.get("provenance", {}).get("sha256", "").lower() != digest:
        raise PlanError("提取图谱时 dump 文件指纹发生变化")
    return graph, "neo4j_dump"


class GraphIndex:
    def __init__(self, graph: dict):
        self.nodes = {}
        identities = {}
        for node in graph["nodes"]:
            props = node["properties"]
            for label in node["labels"]:
                field = LABEL_FIELDS.get(label)
                if field and field in props:
                    key = props[field]
                    if key in self.nodes:
                        raise PlanError(f"图谱标识重复：{key}")
                    self.nodes[key] = (label, props)
                    identities[node["id"]] = key
                    break
        self.outgoing = defaultdict(set)
        self.incoming = defaultdict(set)
        for edge in graph["edges"]:
            left, right = identities.get(edge["source"]), identities.get(edge["target"])
            if left and right:
                relation = edge["type"]
                self.outgoing[left, relation].add(right)
                self.incoming[right, relation].add(left)

    def signal(self, column: str) -> dict | None:
        entry = self.nodes.get(column)
        if entry is None:
            return None
        kind, props = entry
        if kind not in ("检测变量", "操作参数"):
            return None
        relation = "检测位置" if kind == "检测变量" else "操作位置"
        locations = sorted(self.outgoing[column, relation])
        return {"column": column, "name": props.get("变量名称", props.get("参数名称")),
                "quantity": props.get("物理量类型"), "unit": _unit(props.get("单位", props.get("参数单位"))),
                "component": props.get("检测组分"), "source": props.get("变量来源"),
                "locations": locations, "evidence": [[column, relation, site] for site in locations]}

    def inspect_alarm(self, variable: str) -> dict | None:
        info = self.signal(variable)
        if info is None:
            return None
        explicit, connected, equipment = set(), set(), set()
        evidence = list(info["evidence"])
        for site in info["locations"]:
            kind, props = self.nodes[site]
            modules = self.incoming[site, "包含"]
            explicit.update(modules)
            evidence.extend([module, "包含", site] for module in sorted(modules))
            if kind == "设备":
                equipment.add(site)
                continue
            for endpoint in (props.get("起点"), props.get("终点")):
                if self.nodes.get(endpoint, (None,))[0] == "设备":
                    equipment.add(endpoint)
            for adjacent in sorted(self.outgoing[site, "输入"] | self.incoming[site, "输出"]):
                connected.update(self.incoming[adjacent, "包含"])
                triple = ([site, "输入", adjacent] if adjacent in self.outgoing[site, "输入"]
                          else [adjacent, "输出", site])
                evidence.append(triple)
                evidence.extend([module, "包含", adjacent] for module in sorted(self.incoming[adjacent, "包含"]))
        for device in equipment:
            connected.update(self.incoming[device, "包含"])
        return {**info, "explicit_modules": sorted(explicit), "connected_modules": sorted(connected),
                "equipment": sorted(equipment), "evidence": evidence}

    def upstream_variables(self, locations, *, max_stages: int = 2) -> list[dict]:
        """Walk against the process flow and collect the variables on upstream pipes.

        One stage is a 管线 -> 设备 -> 管线 step, so it costs two reverse edges.
        Only incoming edges are followed, therefore a downstream vessel can never
        become a suspect, and the visited set stops the recycle loop. Every hop is
        returned as [from, relation, to] so the path can be stored as evidence.
        """
        if max_stages < 1:
            return []
        depth_limit = 2 * max_stages
        seen = set(locations)
        queue = deque((site, 0, []) for site in sorted(locations))
        found = []
        while queue:
            site, depth, path = queue.popleft()
            if depth >= depth_limit:
                continue
            for relation in ("输出", "输入"):
                for neighbor in sorted(self.incoming[site, relation]):
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    step = [*path, [site, relation, neighbor]]
                    queue.append((neighbor, depth + 1, step))
                    if self.nodes.get(neighbor, (None,))[0] != "管线":
                        continue
                    found.append({"site": neighbor, "stage": max(1, (depth + 2) // 2), "path": step,
                                  "variables": sorted(self.incoming[neighbor, "检测位置"])})
        found.sort(key=lambda row: (row["stage"], row["site"]))
        return found

    def alias_chain(self, column: str) -> list[list[str]]:
        """Follow only explicit reuse of the same physical quantity and unit."""
        links = []
        seen = {column}
        current = self.signal(column)
        while current:
            match = ALIAS_IDS.search(current["source"] or "")
            if match is None or match[1] in seen:
                break
            source = self.signal(match[1])
            if source is None or any(source[k] != current[k] for k in ("quantity", "unit", "component")):
                break
            links.append([current["column"], "复用", source["column"]])
            seen.add(source["column"])
            current = source
        return links


class DiagnosticPlanner:
    def __init__(self, graph: dict, *, graph_origin: str, template_path: Path,
                 temperature_model: Path, ingredient_model: Path,
                 available_columns: Iterable[str] = FEATURE_COLUMNS,
                 selector: Callable[[dict], str] | None = None,
                 upstream_stages: int = 3):
        self.graph = GraphIndex(graph)
        self.provenance = {"source": graph_origin, **graph.get("provenance", {})}
        template = _read_json(template_path)
        if template.get("source", {}).get("sha256", "").lower() != self.provenance.get("sha256", "").lower():
            raise PlanError("图谱 dump 与工艺模板的指纹不一致")
        self.stream_names = {
            row["graph_name"]: key for key, row in template["streams"].items()
        }
        self.ingredient_bindings = ingredient.load_bindings(template_path)
        self.ingredient_model_path = ingredient_model.resolve()
        self.ingredient_params = _read_json(ingredient_model) if ingredient_model.is_file() else None
        self.temperature_model_path = temperature_model.resolve()
        self.temperature_system = (temperature.TemperatureSystem.load(temperature_model)
                                   if temperature_model.is_file() else None)
        self.devices = (self.temperature_system.devices if self.temperature_system is not None else
                        temperature.build_definitions(template, FEATURE_COLUMNS))
        self.available = {name for name in available_columns if isinstance(name, str) and PHYSICAL_ID.fullmatch(name)}
        self.selector = selector
        self.upstream_stages = upstream_stages
        self._selection_cache: dict[str, tuple[str, str]] = {}

    def _binding(self, column, expected_unit, missing, *, location=None, component=None):
        row = self.graph.signal(column)
        if row is None:
            missing.append(f"graph:{column}")
            return {"column": column, "unit": _unit(expected_unit), "status": "missing_graph_binding"}
        if not row["locations"]:
            missing.append(f"graph_location:{column}")
        if row["unit"] != _unit(expected_unit):
            missing.append(f"unit:{column}")
        if location is not None and location not in row["locations"]:
            missing.append(f"location:{column}")
        if component is not None and (row["component"] != component or row["quantity"] != COMPOSITION_QUANTITY):
            missing.append(f"component:{column}")
        if column not in self.available:
            missing.append(column)
        return {**row, "available": column in self.available}

    @staticmethod
    def _candidate(template, scope, *, missing=(), reason=None, **fields):
        missing = list(dict.fromkeys(missing))
        return {"plan_id": f"{template}:{scope}", "template_id": template, "scope_id": scope,
                "status": "blocked" if missing or reason else "ready", "reason": reason,
                "missing_slots": missing, "slot_bindings": fields.pop("slot_bindings", {}), **fields}

    def _ingredient_candidates(self, info, *, role="component", require_alarm_component=True):
        streams = {self.stream_names[site]: site for site in info["locations"] if site in self.stream_names}
        if not streams:
            return [self._candidate("INGREDIENT", None, reason="composition_stream_not_resolved")]
        plans = []
        for stream, location in sorted(streams.items()):
            columns = self.ingredient_bindings.get(stream)
            if columns is None:
                plans.append(self._candidate("INGREDIENT", stream, reason="ingredient_stream_not_trained"))
                continue
            missing = []
            bound = {part: self._binding(column, "mol%", missing, location=location, component=part)
                     for part, column in columns.items()}
            params = self.ingredient_params or {}
            if params.get("bindings", {}).get(stream) != columns:
                missing.append(f"ingredient_parameters:{stream}")
            if require_alarm_component and info["component"] not in columns:
                missing.append(f"alarm_component:{info['component']}")
            names = [*columns, *(["A/C"] if stream == "S4" else [])]
            metrics = []
            for name in names:
                key = f"{stream}.{name}"
                limit = params.get("metrics", {}).get(key, {})
                values = [limit.get(field) for field in ("lower", "baseline", "upper")]
                unit = "ratio" if name == "A/C" else "%"
                valid = (limit.get("unit") == unit
                         and all(type(v) in (int, float) and math.isfinite(v) for v in values)
                         and values == sorted(values))
                if not valid:
                    missing.append(f"metric_parameters:{key}")
                metrics.append({"metric_id": key, "unit": unit,
                                "operation": "ratio" if name == "A/C" else "percent_of_stream_total",
                                "numerator": columns[name] if name != "A/C" else columns["A"],
                                "denominator_columns": list(columns.values()) if name != "A/C" else [columns["C"]],
                                "parameters": limit if valid else None})
            plans.append(self._candidate(
                "INGREDIENT", stream, missing=missing, slot_bindings={"components": bound},
                stream_name=location, alarm_role=role, metrics=metrics,
                model_path=str(self.ingredient_model_path),
            ))
        return plans

    def _device_plan(self, key, device, *, role, selected, alias_evidence, **extra):
        """Bind one trained ARX device and keep every slot check explicit."""
        missing = []
        target = self._binding(device.target, "°C", missing)
        if target.get("quantity") != TEMPERATURE_QUANTITY:
            missing.append(f"target_quantity:{device.target}")
        inputs = []
        for index, spec in enumerate(device.inputs):
            slot = self._binding(spec.column, spec.unit, missing)
            slot.update(input_index=index, slot_name=spec.label, group=spec.group)
            if spec.reference_column:
                reference = self._binding(spec.reference_column, "mol%", missing)
                if (slot.get("quantity") != COMPOSITION_QUANTITY
                        or reference.get("quantity") != COMPOSITION_QUANTITY
                        or not set(slot.get("locations", [])) & set(reference.get("locations", []))):
                    missing.append(f"composition_reference:{spec.reference_column}")
                slot["composition_reference"] = reference
            inputs.append(slot)
        system = self.temperature_system
        if system is None:
            missing.append(f"temperature_parameters:{key}")
        elif not system.parameters[key].calibration:
            missing.append(f"temperature_calibration:{key}")
        return self._candidate(
            "TEMPERATURE", key, missing=missing, model_key=key, model_path=str(self.temperature_model_path),
            scope_label=device.label, module_id=device.module, alarm_role=role, alarm_bound_column=selected,
            alias_evidence=alias_evidence,
            independent_temperature=device.independent_temperature,
            window_seconds=system.horizon_s if system else None,
            slot_bindings={"target": target, "inputs": inputs}, **extra,
        )

    def _composition_plans(self, sites):
        """Composition models of the trained feed streams inside the upstream window.

        A temperature pool only holds measured columns, so a root cause given as a
        ratio (S4.A, S4.A/C) can never appear there. Each trained feed stream that
        reaches this device is therefore carried as an extra composition plan and
        scored by the frozen ingredient thresholds, which keeps the candidate name
        identical to the one used by the ingredient template.
        """
        plans = []
        for row in sites:
            stream = self.stream_names.get(row["site"])
            if stream is None or stream not in self.ingredient_bindings:
                continue
            for plan in self._ingredient_candidates(
                    {"locations": [row["site"]], "component": None},
                    role="upstream_composition", require_alarm_component=False):
                if plan["status"] != "ready":
                    continue
                plans.append({**plan, "upstream": True, "scope_depth": row["stage"],
                              "upstream_site": row["site"], "graph_path": row["path"]})
        plans.sort(key=lambda plan: (plan["scope_depth"], plan["scope_id"]))
        return plans

    def _upstream_plans(self, info, *, local_keys, stages=None, sites=None):
        """Devices inside the upstream window whose inputs sit on upstream pipes.

        A device enters the pool when at least one of its trained inputs is
        measured on a pipeline reached by walking against the flow. Those inputs
        are then scored with that device's own ARX model, so the fault chain is
        followed across equipment boundaries without retraining anything.
        """
        stages = self.upstream_stages if stages is None else stages
        if stages < 1:
            return []
        if sites is None:
            sites = self.graph.upstream_variables(info["locations"], max_stages=stages)
        if not sites:
            return []
        stage_of, path_of = {}, {}
        for row in sites:
            stage_of[row["site"]] = min(stage_of.get(row["site"], stages + 1), row["stage"])
            path_of.setdefault(row["site"], row["path"])
        planned = []
        for key, device in sorted(self.devices.items()):
            if key in local_keys:
                continue
            hits = []
            for spec in device.inputs:
                row = self.graph.signal(spec.column) or {}
                hits.extend((stage_of[site], site) for site in row.get("locations", ()) if site in stage_of)
            if not hits:
                continue
            stage, site = min(hits)
            if stage > stages:
                continue
            plan = self._device_plan(
                key, device, role="upstream_input", selected=None, alias_evidence=[],
                scope_depth=stage, upstream=True, upstream_site=site, graph_path=path_of[site],
            )
            plan["linked_variables"] = sorted(
                spec.column for spec in device.inputs
                if site in ((self.graph.signal(spec.column) or {}).get("locations") or ())
            )
            planned.append(plan)
        planned.sort(key=lambda plan: (plan["status"] != "ready", plan["scope_depth"], plan["plan_id"]))
        return planned

    def _temperature_candidates(self, info):
        column = info["column"]
        alias_chain = self.graph.alias_chain(column)
        source = alias_chain[-1][2] if alias_chain else column
        options = []
        for key, device in self.devices.items():
            if device.target == column:
                options.append((0, key, device, "target", column))
            elif device.target == source:
                options.append((1, key, device, "target_alias", source))
            elif any(x.column == column for x in device.inputs):
                options.append((2, key, device, "input", column))
            elif source != column and any(x.column == source for x in device.inputs):
                options.append((3, key, device, "input_alias", source))
        sites = (self.graph.upstream_variables(info["locations"], max_stages=self.upstream_stages)
                 if self.upstream_stages >= 1 else [])
        upstream = self._upstream_plans(info, local_keys={item[1] for item in options}, sites=sites)
        composition = self._composition_plans(sites)
        if not options:
            if not upstream:
                scopes = info["explicit_modules"] or info["connected_modules"]
                return [self._candidate("TEMPERATURE", scopes[0] if len(scopes) == 1 else None,
                                        reason="temperature_channel_outside_trained_models",
                                        composition_plans=composition)]
            # No model watches this channel, so the nearest upstream vessel with a
            # trained model becomes the carrier and the rest stay as extra pools.
            primary = {**upstream[0], "alarm_role": "upstream_carrier", "upstream": False,
                       "scope_depth": 0, "upstream_carrier": True}
            return [{**primary, "upstream_plans": upstream[1:], "composition_plans": composition}]
        priority = min(item[0] for item in options)
        plans = []
        for rank, key, device, role, selected in options:
            if rank != priority:
                continue
            plan = self._device_plan(key, device, role=role, selected=selected,
                                     alias_evidence=alias_chain if role.endswith("alias") else [],
                                     scope_depth=0, upstream=False)
            plans.append({**plan, "upstream_plans": [item for item in upstream if item["model_key"] != key],
                          "composition_plans": composition})
        return sorted(plans, key=lambda plan: (plan["status"] != "ready", plan["plan_id"]))

    def _select(self, info, plans):
        valid = [plan for plan in plans if plan["status"] == "ready"]
        if self.selector is None or not valid:
            return plans[0], "rules"
        variable = info["column"]
        if variable not in self._selection_cache:
            # Send only graph descriptions and allowed plan IDs. Keep event values
            # and any file or simulation metadata local.
            keys = ("plan_id", "template_id", "scope_id", "alarm_role", "module_id", "scope_label", "stream_name")
            options = [{key: plan[key] for key in keys if key in plan} for plan in valid]
            chosen = self.selector({"alarm_point": info, "allowed_plans": options})
            if not isinstance(chosen, str) or chosen not in {row["plan_id"] for row in valid}:
                raise PlanError(f"大模型为 {variable} 返回了未验证的诊断方案")
            self._selection_cache[variable] = (chosen, "llm")
        chosen, method = self._selection_cache[variable]
        return next(plan for plan in valid if plan["plan_id"] == chosen), method

    def plan_alarm(self, event: Mapping) -> dict:
        alarm = _alarm_fields(event)
        info = self.graph.inspect_alarm(alarm["variable_id"])
        result = {"alarm": alarm, "graph": self.provenance, "graph_location": info,
                  "graph_evidence": info["evidence"] if info else []}
        if info is None:
            plans = [self._candidate(None, None, reason="alarm_measurement_not_in_graph")]
        elif info["quantity"] == COMPOSITION_QUANTITY:
            plans = self._ingredient_candidates(info)
        elif info["quantity"] == TEMPERATURE_QUANTITY:
            plans = self._temperature_candidates(info)
        else:
            plans = [self._candidate(None, None, reason="measurement_type_not_supported")]
        if info and alarm.get("unit", info["unit"]) != info["unit"]:
            plans = [{**plan, "status": "blocked", "reason": "alarm_unit_disagrees_with_graph"} for plan in plans]
        selected, method = self._select(info, plans)
        return {**result, **selected, "selected_by": method,
                "other_plans": [{"plan_id": plan["plan_id"], "status": plan["status"], "reason": plan["reason"],
                                 "missing_slots": plan["missing_slots"]} for plan in plans if plan is not selected]}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def make_model_selector(mode: str) -> Callable[[dict], str] | None:
    if mode == "rules":
        return None
    names = ("COSTEP_LLM_BASE_URL", "COSTEP_LLM_MODEL", "COSTEP_LLM_API_KEY")
    config = [os.getenv(name, "").strip() for name in names]
    if mode == "auto" and not any(config):
        return None
    if not all(config):
        missing = [name for name, value in zip(names, config) if not value]
        raise PlanError(f"大模型配置缺少：{', '.join(missing)}。可用 --selector rules 本地运行。")
    base, model, token = config
    url = urlsplit(base)
    local_http = url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}
    if (not url.hostname or url.username or url.password or url.query or url.fragment
            or (url.scheme != "https" and not local_http)):
        raise PlanError("COSTEP_LLM_BASE_URL 需要 HTTPS 或本机 HTTP 模型接口")
    endpoint = base.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    opener = build_opener(_NoRedirect())

    def select(context: dict) -> str:
        body = json.dumps({
            "model": model, "temperature": 0,
            "messages": [
                {"role": "system", "content": (
                    "你为 CoSTEP 报警点选择诊断模板与对象。根据 alarm_point 的检测位置、"
                    "物理量和图谱证据，只能从 allowed_plans 选择 plan_id。"
                    "测点、单位与公式由本地模板绑定。只返回 JSON 对象 {\"plan_id\": \"...\"}。"
                )},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False, allow_nan=False)},
            ],
        }, ensure_ascii=False).encode("utf-8")
        request = Request(endpoint, data=body, method="POST", headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        })
        try:
            with opener.open(request, timeout=45) as response:
                message = json.load(response)["choices"][0]["message"]["content"]
            answer = json.loads(message)
            if not isinstance(answer, dict) or not isinstance(answer.get("plan_id"), str):
                raise ValueError("invalid plan response")
            return answer["plan_id"]
        except HTTPError as exc:
            raise PlanError(f"模型接口返回 HTTP {exc.code}") from None
        except (URLError, TimeoutError):
            raise PlanError("无法连接模型接口或请求超时") from None
        except (ValueError, KeyError, IndexError, TypeError):
            raise PlanError("模型响应未包含有效的 plan_id JSON 对象") from None

    return select


def _events(path: Path):
    with path.open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                raise PlanError(f"CCA 事件第 {number} 行不是合法 JSON") from None
            if not isinstance(event, dict):
                raise PlanError(f"CCA 事件第 {number} 行需要 JSON 对象")
            yield number, event


def _write_plans(planner, events, stream):
    for number, alarm in events:
        plan = planner.plan_alarm(alarm)
        plan["event_row"] = number
        stream.write(json.dumps(plan, ensure_ascii=False, allow_nan=False) + "\n")


def _publish_plans(planner, events, output: Path):
    if output.exists():
        raise PlanError(f"输出文件已存在：{output}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=".plans-", suffix=".jsonl",
            mode="w", encoding="utf-8", delete=False
        ) as stream:
            temporary = Path(stream.name)
            _write_plans(planner, events, stream)
        # Both operations publish a complete file without replacing an existing one.
        if os.name == "nt":
            temporary.rename(output)
        else:
            os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    input_args = parser.add_mutually_exclusive_group(required=True)
    input_args.add_argument("--variable-id", help="检查单个测点的模板绑定")
    input_args.add_argument("--events", type=Path, help="第一层输出的 events.jsonl")
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--graph-snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--refresh-graph", action="store_true", help="从 Neo4j dump 重新提取图谱")
    parser.add_argument("--neo4j-home", type=Path)
    parser.add_argument("--template", type=Path, default=ingredient.DEFAULT_GRAPH_TEMPLATE)
    parser.add_argument("--temperature-model", type=Path, default=DEFAULT_TEMPERATURE_MODEL)
    parser.add_argument("--ingredient-model", type=Path, default=DEFAULT_INGREDIENT_MODEL)
    parser.add_argument("--selector", default="auto", choices=("auto", "rules", "llm"))
    parser.add_argument("--upstream-stages", type=int, default=3,
                        help="沿工艺流反向回溯的设备级跳数；0 表示只用报警点所属设备")
    parser.add_argument("--output", type=Path, help="输出 JSON Lines；默认写到标准输出")
    args = parser.parse_args()
    try:
        graph, origin = load_graph(args.dump, args.graph_snapshot, neo4j_home=args.neo4j_home,
                                   refresh=args.refresh_graph)
        planner = DiagnosticPlanner(
            graph, graph_origin=origin, template_path=args.template, temperature_model=args.temperature_model,
            ingredient_model=args.ingredient_model, selector=make_model_selector(args.selector),
            upstream_stages=args.upstream_stages,
        )
        if args.output:
            resolved = args.output.resolve()
            protected = (args.events, args.dump, args.graph_snapshot, args.template,
                         args.temperature_model, args.ingredient_model)
            if any(path is not None and resolved == path.resolve() for path in protected):
                raise PlanError("输出不能覆盖诊断输入文件")
            args.output.parent.mkdir(parents=True, exist_ok=True)
        events = _events(args.events) if args.events else [(1, {"variable_id": args.variable_id})]
        if args.output:
            _publish_plans(planner, events, args.output)
        else:
            _write_plans(planner, events, sys.stdout)
    except (CCAError, ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()

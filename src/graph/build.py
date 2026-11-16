"""按 CoSTeP V3 本体，将 V4 Excel 构建数据导入 Neo4j 的 costep 数据库。

在项目根目录运行（路径也可用 --data / --ontology 覆盖）：
    .venv\\Scripts\\python.exe src\\graph\\build.py --dry-run
    .venv\\Scripts\\python.exe src\\graph\\build.py

连接参数读取 NEO4J_URI、NEO4J_USER、NEO4J_PASSWORD、NEO4J_DATABASE，
默认 bolt://127.0.0.1:7687、neo4j、交互输入密码、costep。密码不写入代码。
默认端口来自用户指定的 Neo4j Desktop 实例；通过 Bolt 连接运行中的服务，
不直接修改 DBMS 目录下的数据库文件，也不会自动创建、清空或删除数据库。

schema.json 决定节点标签、属性和有向关系；工作表名必须与本体对应。
V4 扩展：模块-连接→模块表的“通过管线”列写入连接关系的同名文本属性，
保留多段管线路径的原始顺序。V3 未提供此列时，不改动已有的关系属性。
模块/设备/管线以名称为键，检测变量/操作参数以 ID 为键，同名变量不会合并。
先完整校验，再创建唯一性约束，最后在单个事务中批量 MERGE 并核验。
重复运行会更新属性（空属性会移除），不会删除输入中已经不再出现的旧记录。
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import getpass
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Iterator
from zipfile import BadZipFile, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "kg" / "CoSTeP_V4_图谱构建数据.xlsx"
DEFAULT_ONTOLOGY = PROJECT_ROOT / "scheme" / "v3.zip"
PRIMARY_KEYS = {
    "模块": "模块名称",
    "设备": "设备名称",
    "管线": "管线名称",
    "检测变量": "变量ID",
    "操作参数": "参数ID",
}
# V3 本体导出格式没有关系属性定义，此处声明用户要求的 V4 扩展。
RELATION_PROPERTIES = {
    ("模块", "连接", "模块"): ("通过管线",),
}
LOGGER = logging.getLogger(__name__)


class BuildError(Exception):
    """输入数据、本体或导入结果不符合要求。"""


@dataclass(frozen=True)
class EntitySpec:
    label: str
    key: str
    properties: tuple[str, ...]

    @property
    def sheet(self) -> str:
        return f"实体-{self.label}"


@dataclass(frozen=True)
class RelationSpec:
    source: str
    name: str
    target: str
    properties: tuple[str, ...] = ()

    @property
    def sheet(self) -> str:
        return f"{self.source}-{self.name}→{self.target}"

    def columns(self, entities: dict[str, EntitySpec]) -> tuple[str, str]:
        source_key = entities[self.source].key
        target_key = entities[self.target].key
        if source_key == target_key:
            return f"源{source_key}", f"目标{target_key}"
        return source_key, target_key


@dataclass
class Ontology:
    entities: dict[str, EntitySpec]
    relations: tuple[RelationSpec, ...]


@dataclass
class GraphData:
    ontology: Ontology
    nodes: dict[str, list[dict[str, str | None]]]
    relations: dict[RelationSpec, list[dict[str, Any]]]

    @property
    def node_count(self) -> int:
        return sum(map(len, self.nodes.values()))

    @property
    def relation_count(self) -> int:
        return sum(map(len, self.relations.values()))


def load_ontology(path: Path) -> Ontology:
    """读取 ZIP 中的 schema.json，也接受独立的同格式 JSON 文件。"""
    try:
        if path.suffix.lower() == ".zip":
            with ZipFile(path) as archive:
                raw = archive.read("schema.json").decode("utf-8-sig")
        else:
            raw = path.read_text(encoding="utf-8-sig")
        schema = json.loads(raw)
    except (OSError, BadZipFile, KeyError, UnicodeError, ValueError) as exc:
        raise BuildError(f"无法读取本体 {path}: {exc}") from exc
    if not isinstance(schema, dict) or not schema:
        raise BuildError("本体 schema.json 必须是非空的类定义对象。")

    entities: dict[str, EntitySpec] = {}
    relations: list[RelationSpec] = []
    for label, definition in schema.items():
        if label not in PRIMARY_KEYS:
            raise BuildError(f"本体类 {label!r} 尚未配置主键，请更新 PRIMARY_KEYS。")
        if not isinstance(definition, dict):
            raise BuildError(f"本体类 {label!r} 的定义必须是对象。")
        properties = definition.get("dataProperty", [])
        objects = definition.get("objectProperty", [])
        if not isinstance(properties, list) or not isinstance(objects, list):
            raise BuildError(f"本体类 {label!r} 的属性定义必须是数组。")
        names: list[str] = []
        for prop in properties:
            if not isinstance(prop, dict) or not isinstance(prop.get("name"), str):
                raise BuildError(f"本体类 {label!r} 的数据属性缺少名称。")
            name = prop["name"]
            if not name.strip() or name in names:
                raise BuildError(f"本体类 {label!r} 包含空名称或重复属性 {name!r}。")
            if prop.get("range") != "string":
                raise BuildError(f"{label}.{name} 的类型不是当前支持的 string。")
            names.append(name)
        key = PRIMARY_KEYS[label]
        if key not in names:
            raise BuildError(f"本体类 {label!r} 缺少主键属性 {key!r}。")
        entities[label] = EntitySpec(label, key, tuple(names))
        for prop in objects:
            if not isinstance(prop, dict):
                raise BuildError(f"本体类 {label!r} 的对象属性必须是对象。")
            name, target = prop.get("name"), prop.get("range")
            if (
                not isinstance(name, str) or not name.strip()
                or not isinstance(target, str) or target not in schema
            ):
                raise BuildError(f"本体类 {label!r} 包含无效对象属性 {prop!r}。")
            relation = RelationSpec(
                label, name, target, RELATION_PROPERTIES.get((label, name, target), ())
            )
            if relation in relations:
                raise BuildError(f"本体关系重复: {relation.sheet}")
            relations.append(relation)
    return Ontology(entities, tuple(relations))


def read_sheet(
    sheet: Any, expected: tuple[str, ...], optional: tuple[str, ...] = ()
) -> list[tuple[int, dict]]:
    """按列名读取文本；拒绝公式、Excel 错误及未声明的列，保留空值。"""
    rows = sheet.iter_rows()
    header_cells = next(rows, ())
    headers = [cell.value for cell in header_cells]
    while headers and headers[-1] is None:
        headers.pop()
    if any(not isinstance(h, str) or not h.strip() for h in headers):
        raise BuildError(f"{sheet.title}: 表头存在空白或非文本列名。")
    headers = [h.strip() for h in headers]
    if len(set(headers)) != len(headers):
        raise BuildError(f"{sheet.title}: 存在重复列名。")
    missing = set(expected) - set(headers)
    extra = set(headers) - set(expected) - set(optional)
    if missing or extra:
        raise BuildError(
            f"{sheet.title}: 列名与本体不符，缺少 {sorted(missing)}，多出 {sorted(extra)}。"
        )
    result = []
    for row_number, cells in enumerate(rows, 2):
        if all(cell.value is None for cell in cells):
            continue
        if any(cell.value is not None for cell in cells[len(headers):]):
            raise BuildError(f"{sheet.title} 第 {row_number} 行: 表头范围外存在数据。")
        values: dict[str, str | None] = {}
        for name, cell in zip(headers, cells):
            if cell.data_type in {"f", "e"}:
                raise BuildError(
                    f"{sheet.title}!{cell.coordinate}: 不接受公式或 Excel 错误，请使用原始文本值。"
                )
            value = cell.value
            if value is not None and not isinstance(value, str):
                raise BuildError(
                    f"{sheet.title}!{cell.coordinate}: 本体属性为 string，实际为 {type(value).__name__}。"
                )
            values[name] = (value.strip() or None) if value is not None else None
        if any(value is not None for value in values.values()):
            result.append((row_number, values))
    return result


def load_graph(data_path: Path, ontology: Ontology) -> GraphData:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise BuildError("缺少 openpyxl，请安装 src/graph/requirements.txt 中的依赖。") from exc
    try:
        workbook = load_workbook(data_path, read_only=True, data_only=False)
    except (OSError, BadZipFile, ValueError) as exc:
        raise BuildError(f"无法读取工作簿 {data_path}: {exc}") from exc
    try:
        expected = {entity.sheet for entity in ontology.entities.values()}
        expected.update(relation.sheet for relation in ontology.relations)
        actual = set(workbook.sheetnames)
        if actual != expected:
            raise BuildError(
                f"工作表与本体不符，缺少 {sorted(expected - actual)}，多出 {sorted(actual - expected)}。"
            )
        nodes: dict[str, list[dict[str, str | None]]] = {}
        identities: dict[str, set[str]] = {}
        for label, entity in ontology.entities.items():
            nodes[label], identities[label] = [], set()
            for row_number, row in read_sheet(workbook[entity.sheet], entity.properties):
                key = row[entity.key]
                if key is None:
                    raise BuildError(f"{entity.sheet} 第 {row_number} 行: 主键 {entity.key} 不能为空。")
                if key in identities[label]:
                    raise BuildError(f"{entity.sheet} 第 {row_number} 行: 重复主键 {key!r}。")
                identities[label].add(key)
                nodes[label].append(row)
            if not nodes[label]:
                raise BuildError(f"{entity.sheet}: 实体表没有有效数据。")

        relations: dict[RelationSpec, list[dict[str, Any]]] = {}
        for relation in ontology.relations:
            source_column, target_column = relation.columns(ontology.entities)
            seen = set()
            relations[relation] = []
            for row_number, row in read_sheet(
                workbook[relation.sheet], (source_column, target_column), relation.properties
            ):
                source, target = row[source_column], row[target_column]
                for label, key, column in (
                    (relation.source, source, source_column),
                    (relation.target, target, target_column),
                ):
                    if key not in identities[label]:
                        raise BuildError(
                            f"{relation.sheet} 第 {row_number} 行: {column}={key!r} "
                            f"在实体-{label} 中不存在。"
                        )
                if (source, target) in seen:
                    raise BuildError(f"{relation.sheet} 第 {row_number} 行: 重复关系 {(source, target)!r}。")
                seen.add((source, target))
                relations[relation].append({
                    "source": source,
                    "target": target,
                    "properties": {name: row[name] for name in relation.properties if name in row},
                })
        return GraphData(ontology, nodes, relations)
    finally:
        workbook.close()


def quote_identifier(value: str) -> str:
    """标签/属性名来自本体，仍统一转义；数据值一律使用查询参数。"""
    return "`" + value.replace("`", "``") + "`"


def batches(rows: list, size: int) -> Iterator[list]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def node_query(entity: EntitySpec) -> str:
    label, key = quote_identifier(entity.label), quote_identifier(entity.key)
    return (
        f"UNWIND $rows AS row MERGE (n:{label} {{{key}: row.{key}}}) "
        "SET n += row RETURN count(n) AS matched"
    )


def relation_query(relation: RelationSpec, entities: dict[str, EntitySpec]) -> str:
    source, target = entities[relation.source], entities[relation.target]
    return (
        "UNWIND $rows AS row "
        f"MATCH (s:{quote_identifier(source.label)} "
        f"{{{quote_identifier(source.key)}: row.source}}) "
        f"MATCH (t:{quote_identifier(target.label)} "
        f"{{{quote_identifier(target.key)}: row.target}}) "
        f"MERGE (s)-[r:{quote_identifier(relation.name)}]->(t) "
        "SET r += row.properties "
        "RETURN count(r) AS matched"
    )


def verify_graph(tx: Any, data: GraphData) -> dict[str, dict[str, int]]:
    """核对节点属性，以及每条输入关系的方向、唯一性和输入中提供的属性。"""
    counts: dict[str, dict[str, int]] = {"nodes": {}, "relations": {}}
    for label, entity in data.ontology.entities.items():
        key = quote_identifier(entity.key)
        expected = {row[entity.key]: row for row in data.nodes[label]}
        query = (
            f"MATCH (n:{quote_identifier(label)}) WHERE n.{key} IN $keys "
            f"RETURN n.{key} AS identity, properties(n) AS properties"
        )
        records = list(tx.run(query, keys=list(expected)))
        if len(records) != len(expected):
            raise BuildError(f"{label}: 数据库节点数量与输入不一致。")
        for record in records:
            actual = {name: record["properties"].get(name) for name in entity.properties}
            if actual != expected[record["identity"]]:
                raise BuildError(f"{label} {record['identity']!r}: 数据库属性与输入不一致。")
        counts["nodes"][label] = len(records)

    for relation, rows in data.relations.items():
        source = data.ontology.entities[relation.source]
        target = data.ontology.entities[relation.target]
        query = (
            "UNWIND $rows AS row "
            f"OPTIONAL MATCH (s:{quote_identifier(source.label)} "
            f"{{{quote_identifier(source.key)}: row.source}})"
            f"-[r:{quote_identifier(relation.name)}]->"
            f"(t:{quote_identifier(target.label)} "
            f"{{{quote_identifier(target.key)}: row.target}}) "
            "RETURN row.source AS source, row.target AS target, count(r) AS matched, "
            "collect(properties(r)) AS properties"
        )
        verified = 0
        for chunk in batches(rows, 1000):
            records = list(tx.run(query, rows=chunk))
            expected_pairs = Counter((r["source"], r["target"]) for r in chunk)
            actual_pairs = Counter((r["source"], r["target"]) for r in records)
            if actual_pairs != expected_pairs or any(r["matched"] != 1 for r in records):
                raise BuildError(f"{relation.sheet}: 关系缺失、方向错误或存在重复关系。")
            expected_properties = {
                (row["source"], row["target"]): row["properties"] for row in chunk
            }
            for record in records:
                properties = record["properties"][0]
                expected = expected_properties[(record["source"], record["target"])]
                if any(properties.get(name) != value for name, value in expected.items()):
                    raise BuildError(
                        f"{relation.sheet} {record['source']} → {record['target']}: "
                        "数据库关系属性与输入不一致。"
                    )
            verified += len(records)
        counts["relations"][relation.sheet] = verified
    return counts


def write_graph(tx: Any, data: GraphData, batch_size: int) -> dict:
    """所有数据操作和核验共用一个事务；异常时整体回滚，允许驱动安全重试。"""
    created = {"nodes": 0, "relations": 0}
    queries = [(node_query(data.ontology.entities[label]), rows) for label, rows in data.nodes.items()]
    queries.extend(
        (relation_query(relation, data.ontology.entities), rows)
        for relation, rows in data.relations.items()
    )
    for query, rows in queries:
        for chunk in batches(rows, batch_size):
            result = tx.run(query, rows=chunk)
            record = result.single()
            summary = result.consume()
            if record is None or record["matched"] != len(chunk):
                raise BuildError("批量写入数量不符，可能存在缺失端点或重复关系，数据事务将回滚。")
            created["nodes"] += summary.counters.nodes_created
            created["relations"] += summary.counters.relationships_created
    return {"created": created, "verified": verify_graph(tx, data)}


def import_graph(
    data: GraphData, uri: str, user: str, password: str, database: str, batch_size: int = 1000
) -> dict:
    if batch_size < 1:
        raise BuildError("batch_size 必须为正整数。")
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise BuildError("缺少 neo4j，请安装 src/graph/requirements.txt 中的依赖。") from exc
    with GraphDatabase.driver(uri, auth=(user, password), connection_timeout=10) as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            # 明确检查目标库可访问；不回退到默认数据库或其他数据库。
            session.run("RETURN 1 AS connected").consume()
            for entity in data.ontology.entities.values():
                name = quote_identifier(f"costep_{entity.label}_identity")
                session.run(
                    f"CREATE CONSTRAINT {name} IF NOT EXISTS "
                    f"FOR (n:{quote_identifier(entity.label)}) "
                    f"REQUIRE n.{quote_identifier(entity.key)} IS UNIQUE"
                ).consume()
            return session.execute_write(write_graph, data, batch_size)


def positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须为正整数") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Excel 构建数据路径")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY, help="本体 ZIP / schema.json 路径")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "costep"))
    parser.add_argument("--batch-size", type=positive_integer, default=1000)
    parser.add_argument("--dry-run", action="store_true", help="只读取和校验数据，不连接或修改数据库")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
    try:
        ontology = load_ontology(args.ontology)
        data = load_graph(args.data, ontology)
        LOGGER.info("本体和数据校验通过: %s 类实体，%s 种有向关系组合。", len(ontology.entities), len(ontology.relations))
        for label, rows in data.nodes.items():
            LOGGER.info("实体 %s: %s", label, len(rows))
        for relation, rows in data.relations.items():
            LOGGER.info("关系 %s: %s", relation.sheet, len(rows))
        LOGGER.info("合计 %s 个节点，%s 条关系。", data.node_count, data.relation_count)
        if args.dry_run:
            LOGGER.info("仅校验模式完成，未连接数据库。")
            return 0
        password = os.getenv("NEO4J_PASSWORD")
        if not password:
            if not sys.stdin.isatty():
                raise BuildError("非交互环境请设置 NEO4J_PASSWORD 环境变量。")
            password = getpass.getpass("Neo4j 密码: ")
        if not password:
            raise BuildError("Neo4j 密码不能为空。")
        result = import_graph(data, args.uri, args.user, password, args.database, args.batch_size)
        LOGGER.info(
            "导入 %s 完成: 新增 %s 个节点、%s 条关系；已核验 %s 个节点及 %s 条关系。",
            args.database, result["created"]["nodes"], result["created"]["relations"],
            sum(result["verified"]["nodes"].values()), sum(result["verified"]["relations"].values()),
        )
        return 0
    except (BuildError, OSError, EOFError) as exc:
        LOGGER.error("构建失败: %s", exc)
        return 1
    except Exception as exc:
        # 驱动的错误码便于区分认证失败、目标库不存在、连接失败和约束冲突。
        # 不打印参数或完整异常文本，以免连接错误泄漏凭据。
        if type(exc).__module__.startswith("neo4j"):
            LOGGER.error(
                "Neo4j 构建失败 [%s]，请检查连接、凭据、目标库状态及约束冲突。",
                getattr(exc, "code", type(exc).__name__),
            )
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())

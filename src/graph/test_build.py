r"""离线回归测试：使用原始工作簿的临时副本，不连接 Neo4j，也不改动源文件。

运行：.venv\Scripts\python.exe -m unittest discover -s src/graph -p test_build.py -v
"""

from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import build


NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


class BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ontology = build.load_ontology(build.DEFAULT_ONTOLOGY)
        cls.data = build.load_graph(build.DEFAULT_DATA, cls.ontology)
        with ZipFile(build.DEFAULT_DATA) as archive:
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            cls.sheet_paths = {
                sheet.attrib["name"]: f"xl/worksheets/sheet{index}.xml"
                for index, sheet in enumerate(workbook.find(f"{{{NS}}}sheets"), 1)
            }

    def modified_workbook(self, sheet, changes):
        """只修改临时副本中的指定单元格，保留其余实际数据作为测试输入。"""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name) / "fixture.xlsx"
        target = self.sheet_paths[sheet]
        with ZipFile(build.DEFAULT_DATA) as source, ZipFile(output, "w") as destination:
            for entry in source.infolist():
                content = source.read(entry.filename)
                if entry.filename == target:
                    root = ET.fromstring(content)
                    for coordinate, kind, value in changes:
                        cell = root.find(f".//{{{NS}}}c[@r='{coordinate}']")
                        self.assertIsNotNone(cell, coordinate)
                        for child in list(cell):
                            cell.remove(child)
                        cell.attrib.pop("t", None)
                        if kind == "text":
                            cell.set("t", "inlineStr")
                            text = ET.SubElement(ET.SubElement(cell, f"{{{NS}}}is"), f"{{{NS}}}t")
                            text.text = value
                        elif kind == "formula":
                            ET.SubElement(cell, f"{{{NS}}}f").text = value
                        elif kind == "number":
                            ET.SubElement(cell, f"{{{NS}}}v").text = value
                        elif kind == "error":
                            cell.set("t", "e")
                            ET.SubElement(cell, f"{{{NS}}}v").text = value
                        elif kind != "blank":
                            self.fail(f"Unknown fixture kind {kind}")
                    content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
                destination.writestr(entry, content)
        return output

    def test_real_workbook_counts_and_ontology(self):
        self.assertEqual(self.data.node_count, 223)
        self.assertEqual(self.data.relation_count, 258)
        self.assertEqual(set(self.data.nodes), set(build.PRIMARY_KEYS))
        self.assertEqual(len(self.ontology.relations), 10)

    def test_same_name_variables_remain_distinct_by_id(self):
        variables = self.data.nodes["检测变量"]
        names = Counter(row["变量名称"] for row in variables)
        self.assertGreater(names["过程管线2 总质量流量"], 1)
        self.assertEqual(len({row["变量ID"] for row in variables}), 179)
        self.assertIsNone(variables[0]["检测组分"])

    def test_v4_module_connections_preserve_pipeline_paths(self):
        relation = next(r for r in self.ontology.relations if r.sheet == "模块-连接→模块")
        actual = {
            (row["source"], row["target"]): row["properties"]
            for row in self.data.relations[relation]
        }
        self.assertEqual(actual, {
            ("M1", "M2"): {"通过管线": "过程管线7"},
            ("M2", "M3"): {"通过管线": "过程管线10"},
            ("M2", "M1"): {"通过管线": "过程管线8 → 过程管线6"},
            ("M3", "M1"): {"通过管线": "过程管线5 → 过程管线8 → 过程管线6"},
        })
        for spec, rows in self.data.relations.items():
            if spec != relation:
                self.assertTrue(all(row["properties"] == {} for row in rows))

    def test_v3_without_pipeline_column_does_not_supply_properties(self):
        path = build.DEFAULT_DATA.with_name("CoSTeP_V3_图谱构建数据.xlsx")
        data = build.load_graph(path, self.ontology)
        self.assertEqual((data.node_count, data.relation_count), (223, 258))
        self.assertTrue(all(
            row["properties"] == {} for rows in data.relations.values() for row in rows
        ))

    def test_blank_pipeline_attribute_is_an_explicit_removal(self):
        path = self.modified_workbook("模块-连接→模块", [("C2", "blank", None)])
        data = build.load_graph(path, self.ontology)
        relation = next(r for r in self.ontology.relations if r.sheet == "模块-连接→模块")
        self.assertEqual(data.relations[relation][0]["properties"], {"通过管线": None})

    def test_undeclared_relationship_attribute_is_rejected(self):
        path = self.modified_workbook("模块-连接→模块", [("C1", "text", "本体外属性")])
        with self.assertRaisesRegex(build.BuildError, "列名与本体不符"):
            build.load_graph(path, self.ontology)

    def test_non_text_pipeline_attribute_is_rejected(self):
        path = self.modified_workbook("模块-连接→模块", [("C2", "number", "7")])
        with self.assertRaisesRegex(build.BuildError, "属性为 string"):
            build.load_graph(path, self.ontology)

    def test_verification_rejects_wrong_stored_pipeline_attribute(self):
        entity = self.ontology.entities["模块"]
        relation = next(r for r in self.ontology.relations if r.sheet == "模块-连接→模块")
        data = build.GraphData(
            build.Ontology({"模块": entity}, (relation,)),
            {"模块": [{"模块名称": "M1"}, {"模块名称": "M2"}]},
            {relation: [{"source": "M1", "target": "M2", "properties": {"通过管线": "过程管线7"}}]},
        )
        for properties in ({}, {"通过管线": "过程管线10"}):
            with self.subTest(properties=properties):
                tx = Mock()
                tx.run.side_effect = [
                    [{"identity": "M1", "properties": {"模块名称": "M1"}},
                     {"identity": "M2", "properties": {"模块名称": "M2"}}],
                    [{"source": "M1", "target": "M2", "matched": 1, "properties": [properties]}],
                ]
                with self.assertRaisesRegex(build.BuildError, "关系属性与输入不一致"):
                    build.verify_graph(tx, data)

    def test_duplicate_primary_key_is_rejected(self):
        path = self.modified_workbook("实体-检测变量", [("C3", "text", "SM_001")])
        with self.assertRaisesRegex(build.BuildError, "重复主键"):
            build.load_graph(path, self.ontology)

    def test_blank_primary_key_is_rejected(self):
        path = self.modified_workbook("实体-检测变量", [("C2", "blank", None)])
        with self.assertRaisesRegex(build.BuildError, "主键.*不能为空"):
            build.load_graph(path, self.ontology)

    def test_missing_relation_endpoints_are_rejected(self):
        for coordinate in ("A2", "B2"):
            with self.subTest(coordinate=coordinate):
                path = self.modified_workbook("模块-包含→设备", [(coordinate, "text", "不存在")])
                with self.assertRaisesRegex(build.BuildError, "中不存在"):
                    build.load_graph(path, self.ontology)

    def test_duplicate_relation_is_rejected(self):
        path = self.modified_workbook(
            "模块-连接→模块", [("A3", "text", "M1"), ("B3", "text", "M2")]
        )
        with self.assertRaisesRegex(build.BuildError, "重复关系"):
            build.load_graph(path, self.ontology)

    def test_undeclared_property_is_rejected(self):
        path = self.modified_workbook("实体-模块", [("A1", "text", "本体外属性")])
        with self.assertRaisesRegex(build.BuildError, "列名与本体不符"):
            build.load_graph(path, self.ontology)

    def test_formulas_errors_and_numeric_values_are_rejected(self):
        for kind, value, message in (
            ("formula", '"M1"', "不接受公式"),
            ("error", "#REF!", "Excel 错误"),
            ("number", "123", "属性为 string"),
        ):
            with self.subTest(kind=kind):
                path = self.modified_workbook("实体-模块", [("A2", kind, value)])
                with self.assertRaisesRegex(build.BuildError, message):
                    build.load_graph(path, self.ontology)

    def test_missing_sheet_is_rejected(self):
        reduced = build.Ontology(self.ontology.entities, self.ontology.relations[:-1])
        with self.assertRaisesRegex(build.BuildError, "工作表与本体不符"):
            build.load_graph(build.DEFAULT_DATA, reduced)

    def test_ontology_disallows_undefined_target_and_wrong_type(self):
        with ZipFile(build.DEFAULT_ONTOLOGY) as archive:
            original = json.loads(archive.read("schema.json"))
        for invalid_range in ("不存在的类", None, []):
            with self.subTest(invalid_range=invalid_range), TemporaryDirectory() as temporary:
                original["模块"]["objectProperty"][0]["range"] = invalid_range
                path = Path(temporary) / "schema.json"
                path.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
                with self.assertRaisesRegex(build.BuildError, "无效对象属性"):
                    build.load_ontology(path)
        original["模块"]["dataProperty"][0]["range"] = "integer"
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "schema.json"
            path.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(build.BuildError, "不是当前支持的 string"):
                build.load_ontology(path)

    def test_dry_run_never_imports_or_asks_for_password(self):
        with patch.object(build, "import_graph") as importer, patch.object(build.getpass, "getpass") as prompt:
            self.assertEqual(build.main(["--dry-run"]), 0)
        importer.assert_not_called()
        prompt.assert_not_called()


if __name__ == "__main__":
    unittest.main()

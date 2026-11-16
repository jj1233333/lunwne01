"""第二层候选池的回归检查。

上游回溯验证四件事：只沿逆流方向走、按设备级跳数截断、遇到循环不重复展开、
每一跳可以作为证据回放。组成候选池验证超限指标与报警点同层排序。
图谱结构与 data/kg 下的模板一致：管线 --输入--> 设备 --输出--> 管线。
"""

import unittest

from costep_diagnosis.layer2.difference import _rank_key
from costep_diagnosis.layer2.llm import GraphIndex


def signal(name, site):
    return {"id": f"v:{name}", "labels": ["检测变量"],
            "properties": {"变量ID": name, "变量名称": f"{name} 温度", "物理量类型": "温度",
                           "单位": "℃", "检测组分": None, "变量来源": ""}}


def pipeline(name):
    return {"id": f"p:{name}", "labels": ["管线"], "properties": {"管线名称": name}}


def device(name):
    return {"id": f"d:{name}", "labels": ["设备"], "properties": {"设备名称": name}}


def edge(left, right, relation):
    return {"source": left, "target": right, "type": relation, "properties": {}}


def process_graph():
    """冷却水供给 → 反应器 → 管线2 → 汽提塔 → 管线3，外加一条循环边。"""
    nodes = [
        pipeline("P1"), pipeline("P2"), pipeline("P3"), pipeline("P4"),
        device("反应器"), device("汽提塔"),
        signal("SM_101", "P1"), signal("SM_102", "P2"),
        signal("SM_103", "P3"), signal("SM_104", "P4"),
    ]
    edges = [
        edge("v:SM_101", "p:P1", "检测位置"), edge("v:SM_102", "p:P2", "检测位置"),
        edge("v:SM_103", "p:P3", "检测位置"), edge("v:SM_104", "p:P4", "检测位置"),
        edge("p:P1", "d:反应器", "输入"), edge("d:反应器", "p:P2", "输出"),
        edge("p:P2", "d:汽提塔", "输入"), edge("d:汽提塔", "p:P3", "输出"),
        # 回流线：把管线3接回反应器进料，形成环
        edge("p:P3", "p:P1", "连接"),
    ]
    return {"nodes": nodes, "edges": edges}


class UpstreamTests(unittest.TestCase):
    def setUp(self):
        self.index = GraphIndex(process_graph())

    def test_walks_against_the_flow_and_stops_at_the_stage_budget(self):
        one = self.index.upstream_variables(["P3"], max_stages=1)
        self.assertEqual([row["site"] for row in one], ["P2"])
        two = self.index.upstream_variables(["P3"], max_stages=2)
        self.assertEqual([(row["site"], row["stage"]) for row in two], [("P2", 1), ("P1", 2)])

    def test_never_returns_a_downstream_pipeline(self):
        sites = {row["site"] for row in self.index.upstream_variables(["P1"], max_stages=3)}
        self.assertNotIn("P2", sites)
        self.assertNotIn("P3", sites)
        # P1 is a raw feed, so nothing sits upstream of it.
        self.assertEqual(sites, set())

    def test_lists_the_variables_measured_on_each_upstream_pipe(self):
        rows = {row["site"]: row for row in self.index.upstream_variables(["P3"], max_stages=2)}
        self.assertEqual(rows["P2"]["variables"], ["SM_102"])
        self.assertEqual(rows["P1"]["variables"], ["SM_101"])

    def test_records_the_hop_as_replayable_evidence(self):
        rows = {row["site"]: row for row in self.index.upstream_variables(["P3"], max_stages=2)}
        self.assertEqual(rows["P2"]["path"], [["P3", "输出", "汽提塔"], ["汽提塔", "输入", "P2"]])

    def test_zero_stages_disables_the_search(self):
        self.assertEqual(self.index.upstream_variables(["P3"], max_stages=0), [])

    def test_recycle_edge_does_not_expand_twice(self):
        rows = self.index.upstream_variables(["P3"], max_stages=3)
        self.assertEqual(len(rows), len({row["site"] for row in rows}))

    def test_unknown_location_yields_nothing(self):
        self.assertEqual(self.index.upstream_variables(["不存在的管线"], max_stages=2), [])


def candidate(name, *, status="ready", d_j=0.0, depth=0, exceeded=False, template="TEMPERATURE"):
    return {"candidate": name, "status": status, "D_j": d_j, "scope_depth": depth,
            "exceeded": exceeded, "template_id": template}


class CompositionPoolTests(unittest.TestCase):
    """组成指标并入温度池后的排序规则。"""

    def order(self, rows, alarming=(), rank_rule="alarm-first"):
        return [row["candidate"] for row in sorted(
            rows, key=lambda row: _rank_key(row, set(alarming), rank_rule))]

    def test_exceeded_composition_shares_the_first_tier_with_alarms(self):
        rows = [candidate("SM_030", d_j=0.2),
                candidate("S4.C", d_j=6.1, exceeded=True, template="INGREDIENT"),
                candidate("SM_044", d_j=0.1)]
        self.assertEqual(self.order(rows, alarming={"SM_030"}), ["S4.C", "SM_030", "SM_044"])

    def test_unexceeded_composition_stays_behind_hard_evidence(self):
        rows = [candidate("S4.B", d_j=0.9, template="INGREDIENT"),
                candidate("SM_132", d_j=0.05),
                candidate("S4.A/C", d_j=1.5, exceeded=True, template="INGREDIENT")]
        self.assertEqual(self.order(rows, alarming={"SM_132"}), ["S4.A/C", "SM_132", "S4.B"])

    def test_blocked_composition_metric_is_last(self):
        rows = [candidate("S4.A", status="blocked", d_j=None, template="INGREDIENT"),
                candidate("SM_033", d_j=0.01)]
        self.assertEqual(self.order(rows), ["SM_033", "S4.A"])

    def test_dj_rule_drops_the_alarm_privilege(self):
        rows = [candidate("SM_030", d_j=0.2), candidate("S4.C", d_j=6.1)]
        self.assertEqual(self.order(rows, alarming={"SM_030"}, rank_rule="dj"), ["S4.C", "SM_030"])


if __name__ == "__main__":
    unittest.main()

"""知识图谱模块测试"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from kg.visualizer import KGVisualizer


class TestNodeColor:
    """节点颜色启发性规则测试"""

    @pytest.fixture
    def visualizer(self):
        return KGVisualizer()

    def test_accident_red(self, visualizer):
        assert visualizer._get_node_color("火灾事故") == "#FF4B4B"
        assert visualizer._get_node_color("爆炸事件") == "#FF4B4B"
        assert visualizer._get_node_color("中毒伤亡") == "#FF4B4B"
        assert visualizer._get_node_color("触电事故") == "#FF4B4B"

    def test_measure_green(self, visualizer):
        assert visualizer._get_node_color("应急措施") == "#4BFF4B"
        assert visualizer._get_node_color("安全管理") == "#4BFF4B"
        assert visualizer._get_node_color("隐患排查") == "#4BFF4B"
        assert visualizer._get_node_color("急救方法") == "#4BFF4B"

    def test_org_blue(self, visualizer):
        assert visualizer._get_node_color("企业单位") == "#4B4BFF"
        assert visualizer._get_node_color("监管部门") == "#4B4BFF"
        assert visualizer._get_node_color("消防人员") == "#4B4BFF"

    def test_equipment_purple(self, visualizer):
        assert visualizer._get_node_color("消防设备") == "#FF4BFF"
        assert visualizer._get_node_color("安全设施") == "#FF4BFF"
        assert visualizer._get_node_color("应急工具") == "#FF4BFF"
        assert visualizer._get_node_color("防护装置") == "#FF4BFF"

    def test_other_gold(self, visualizer):
        assert visualizer._get_node_color("安全生产") == "#FFD700"
        assert visualizer._get_node_color("法律法规") == "#FFD700"

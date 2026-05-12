"""知识图谱可视化模块"""
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from pyvis.network import Network

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / "config" / ".env"


def _get_available_relation_types(driver, database):
    with driver.session(database=database) as session:
        rows = session.run(
            """
            MATCH ()-[r]->()
            RETURN DISTINCT coalesce(r.original_relation, type(r)) AS rel_type
            ORDER BY rel_type
            """
        )
        return [row["rel_type"] for row in rows]


def _get_available_sources(driver, database):
    with driver.session(database=database) as session:
        rows = session.run(
            """
            MATCH ()-[r]->()
            WHERE r.source IS NOT NULL
            RETURN DISTINCT r.source AS source
            ORDER BY source
            """
        )
        return [row["source"] for row in rows]


class KGVisualizer:
    """知识图谱可视化器"""

    def __init__(self):
        load_dotenv(ENV_FILE)
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

        from utils.neo4j_driver import create_driver

        self.driver = create_driver()
        self.driver.verify_connectivity()
    
    def close(self):
        """关闭连接"""
        self.driver.close()

    def get_stats(self):
        """获取图谱统计信息"""
        with self.driver.session(database=self.database) as session:
            node_count = session.run(
                "MATCH (n:Entity) RETURN count(n) AS count"
            ).single()["count"]
            rel_count = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS count"
            ).single()["count"]
            doc_count = session.run(
                """
                MATCH ()-[r]->()
                WHERE r.source IS NOT NULL
                RETURN count(DISTINCT r.source) AS count
                """
            ).single()["count"]
            relation_rows = session.run(
                """
                MATCH ()-[r]->()
                RETURN coalesce(r.original_relation, type(r)) AS rel_type,
                       count(*) AS count
                ORDER BY count DESC
                LIMIT 8
                """
            )

            relations = [
                {"type": row["rel_type"], "count": row["count"]}
                for row in relation_rows
            ]

        return {
            "node_count": node_count,
            "rel_count": rel_count,
            "doc_count": doc_count,
            "relations": relations,
        }
    
    def get_filter_options(self):
        """获取可用的筛选选项"""
        return {
            "relations": _get_available_relation_types(self.driver, self.database),
            "sources": _get_available_sources(self.driver, self.database),
        }

    def visualize_all(self, output_file="knowledge_graph.html", limit=100,
                      relation_filter=None, source_filter=None):
        """可视化知识图谱

        Args:
            output_file: 输出HTML文件名
            limit: 限制关系数量
            relation_filter: 可选，按关系类型筛选
            source_filter: 可选，按来源文档筛选
        """
        net = Network(
            height="900px",
            width="100%",
            bgcolor="#1E1E1E",
            font_color="white",
        )

        where_clauses = []
        params = {"limit": limit}
        if relation_filter:
            where_clauses.append(
                "(r.original_relation = $rel_filter OR type(r) = $rel_filter)"
            )
            params["rel_filter"] = relation_filter
        if source_filter:
            where_clauses.append("r.source = $src_filter")
            params["src_filter"] = source_filter

        where_str = " AND ".join(where_clauses) if where_clauses else "TRUE"

        with self.driver.session(database=self.database) as session:
            result = list(session.run(
                f"""
                MATCH (n:Entity)-[r]->(m:Entity)
                WHERE {where_str}
                RETURN n, r, m
                LIMIT $limit
                """,
                **params,
            ))

            degree = defaultdict(int)
            edges = []
            seen_edges = set()

            for record in result:
                source = record['n']
                target = record['m']
                relation = record['r']
                source_name = source.get('name', 'Unknown')
                target_name = target.get('name', 'Unknown')
                rel_label = relation.get('original_relation') or getattr(relation, "type", type(relation).__name__)
                edge_key = (source_name, rel_label, target_name)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edges.append((source_name, target_name, rel_label))
                degree[source_name] += 1
                degree[target_name] += 1

            added_nodes = set()
            for name, deg in degree.items():
                node_color = self._get_node_color(name)
                node_size = min(48, 18 + deg * 2)
                node_label = name[:18] + "..." if len(name) > 18 else name
                net.add_node(
                    name,
                    label=node_label,
                    title=f"{name}<br>关联数: {deg}",
                    color=node_color,
                    size=node_size,
                )
                added_nodes.add(name)

            for source_name, target_name, rel_label in edges:
                edge_label = rel_label[:12] + "..." if len(rel_label) > 12 else rel_label
                net.add_edge(
                    source_name,
                    target_name,
                    label=edge_label,
                    title=rel_label,
                    color="#8B8B8B",
                    arrows="to",
                )
        
        # 设置物理效果
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "stabilization": {
              "iterations": 250
            },
            "barnesHut": {
              "gravitationalConstant": -18000,
              "centralGravity": 0.22,
              "springLength": 170,
              "springConstant": 0.04,
              "damping": 0.2
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true,
            "multiselect": true
          },
          "nodes": {
            "shape": "dot",
            "font": {
              "size": 14
            }
          },
          "edges": {
            "smooth": {
              "type": "dynamic"
            },
            "font": {
              "size": 10,
              "align": "middle"
            }
          }
        }
        """)
        
        # 生成HTML
        net.save_graph(output_file)
        print(f"✅ 图谱已生成：{output_file}")
        print(f"   数据库: {self.database}")
        print(f"   节点数量: {len(added_nodes)}")
        print(f"   可在浏览器中打开查看")
        
        return output_file
    
    def _get_node_color(self, name):
        """根据节点名称推测类型并返回颜色（按优先级匹配）"""
        if any(word in name for word in ['事故', '火灾', '爆炸', '中毒', '触电', '泄漏', '坍塌']):
            return '#FF4B4B'  # 红色 - 事故/风险
        elif any(word in name for word in ['措施', '管理', '排查', '急救', '救援', '处置']):
            return '#4BFF4B'  # 绿色 - 措施/处置
        elif any(word in name for word in ['设备', '设施', '工具', '装置', '器材']):
            return '#FF4BFF'  # 紫色 - 设备/资源（在组织之前检查）
        elif any(word in name for word in ['企业', '部门', '人员', '消防', '单位', '机构']):
            return '#4B4BFF'  # 蓝色 - 组织/人员
        else:
            return '#FFD700'  # 金色 - 其他


if __name__ == "__main__":
 
    print("应急知识图谱可视化")
 
    
    visualizer = KGVisualizer()
    
    print("\n生成知识图谱可视化...")
    visualizer.visualize_all("emergency_kg_full.html", limit=150)
    
    visualizer.close()
    
    print("\n🎉 完成！请打开 emergency_kg_full.html 查看图谱")
 

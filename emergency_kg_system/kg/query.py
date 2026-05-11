"""知识图谱查询辅助"""
import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "config" / ".env")


def get_neighbors(driver, entity_name, database=None, limit=20):
    """查询某个实体的一跳邻居（双向）"""
    db = database or os.getenv("NEO4J_DATABASE", "neo4j")
    with driver.session(database=db) as session:
        rows = session.run(
            """
            MATCH (h:Entity {name: $name})-[r]->(t:Entity)
            RETURN t.name AS target,
                   coalesce(r.original_relation, type(r)) AS relation,
                   r.source AS source
            UNION
            MATCH (h:Entity)-[r]->(t:Entity {name: $name})
            RETURN h.name AS target,
                   coalesce(r.original_relation, type(r)) AS relation,
                   r.source AS source
            LIMIT $limit
            """,
            name=entity_name, limit=limit,
        )
        return [
            {"target": row["target"], "relation": row["relation"], "source": row["source"]}
            for row in rows
        ]


def get_two_hop_paths(driver, source_name, target_name, database=None, limit=10):
    """查找两个实体之间的两跳路径"""
    db = database or os.getenv("NEO4J_DATABASE", "neo4j")
    with driver.session(database=db) as session:
        rows = session.run(
            """
            MATCH path = (a:Entity {name: $src})-[*2]->(b:Entity {name: $tgt})
            RETURN path LIMIT $limit
            """,
            src=source_name, tgt=target_name, limit=limit,
        )
        paths = []
        for record in rows:
            path = record["path"]
            nodes = [node.get("name", "?") for node in path.nodes]
            rels = [
                rel.get("original_relation") or type(rel).__name__
                for rel in path.relationships
            ]
            steps = []
            for i in range(len(nodes) - 1):
                steps.append(f"({nodes[i]})-[{rels[i]}]->({nodes[i + 1]})")
            paths.append(" → ".join(steps))
        return paths


def search_by_entity(driver, keyword, database=None, limit=20):
    """模糊搜索实体名称"""
    db = database or os.getenv("NEO4J_DATABASE", "neo4j")
    with driver.session(database=db) as session:
        rows = session.run(
            "MATCH (n:Entity) WHERE n.name CONTAINS $kw "
            "RETURN n.name AS name LIMIT $limit",
            kw=keyword, limit=limit,
        )
        return [row["name"] for row in rows]


def get_entity_stats(driver, entity_name, database=None):
    """获取单个实体的出入度及来源文档统计"""
    db = database or os.getenv("NEO4J_DATABASE", "neo4j")
    with driver.session(database=db) as session:
        out = session.run(
            "MATCH (:Entity {name: $name})-[r]->() RETURN count(r) AS out_count",
            name=entity_name,
        ).single()["out_count"]
        in_count = session.run(
            "MATCH ()-[r]->(:Entity {name: $name}) RETURN count(r) AS in_count",
            name=entity_name,
        ).single()["in_count"]
        sources = session.run(
            "MATCH (:Entity {name: $name})-[r]->() "
            "WHERE r.source IS NOT NULL "
            "RETURN DISTINCT r.source AS source",
            name=entity_name,
        )
        doc_list = [row["source"] for row in sources]
        return {
            "name": entity_name,
            "outgoing": out_count,
            "incoming": in_count,
            "documents": doc_list,
        }

"""将三元组导入Neo4j知识图谱"""
import json
import os
import re
import sys
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

ENV_FILE = BASE_DIR / "config" / ".env"
ANNOTATIONS_DIR = BASE_DIR / "data" / "annotations"
LEGACY_OUTPUT_FILE = ANNOTATIONS_DIR / "自动提取三元组.json"
IMPORT_TRACKER = ANNOTATIONS_DIR / "import_tracker.json"

load_dotenv(ENV_FILE)

# 噪声实体黑名单（过泛或无效的概念）
ENTITY_BLACKLIST = {
    "事故", "事件", "情况", "问题", "工作", "内容", "要求", "规定",
    "相关", "有关", "相应", "必要", "一定", "可能", "存在", "影响",
    "处理", "进行", "发生", "出现", "采取", "实施", "使用", "落实",
}

# 全角→半角映射
_FULLWIDTH_MAP = str.maketrans(
    "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ（）［］，。！？：；＂＇",
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz()[],.!?:;\"'",
)


def _normalize_entity(name):
    """实体名称标准化：去空白、全角转半角、去引号"""
    name = name.strip()
    name = name.translate(_FULLWIDTH_MAP)
    name = name.strip("""'"''「」『』""")
    name = re.sub(r"\s+", " ", name)
    return name


def _build_triple_batch(triples):
    """预处理三元组，过滤无效数据并清理关系名"""
    valid = []
    for t in triples:
        head = _normalize_entity(t.get("head") or "")
        relation = (t.get("relation") or "").strip()
        tail = _normalize_entity(t.get("tail") or "")
        source = t.get("source") or ""

        if not head or not relation or not tail:
            continue
        if len(head) < 1 or len(tail) < 1:
            continue
        if head in ENTITY_BLACKLIST or tail in ENTITY_BLACKLIST:
            continue
        # 实体名过长通常是错误抽取
        if len(head) > 60 or len(tail) > 60:
            continue

        # 清理关系名：将标点替换为 _，保留中文字符
        clean = re.sub(r"[^\w]+", "_", relation, flags=re.UNICODE).strip("_")
        if clean and clean[0].isdigit():
            clean = f"REL_{clean}"
        if not clean:
            clean = "RELATED_TO"
        # 过滤无意义的关系名（纯下划线或过短）
        if clean in ("_", "__") or len(clean) < 2:
            clean = "RELATED_TO"

        valid.append((head, relation, tail, source, clean))
    return valid


class TripleImporter:
    """三元组导入器"""

    def __init__(self):
        self.database = os.getenv("NEO4J_DATABASE", "dc14aaed")
        from utils.neo4j_driver import create_driver

        self.driver = create_driver()
        self.driver.verify_connectivity()
    
    def close(self):
        """关闭连接"""
        self.driver.close()

    def merge_similar_entities(self, threshold=0.85, dry_run=True):
        """合并名称高度相似的同类型实体节点

        对全部 Entity 节点两两比较名称相似度，超过阈值的合并为一个节点，
        保留度数更高的节点名，将另一节点的所有关系迁移到保留节点。

        Args:
            threshold: 相似度阈值 (0-1)，默认 0.85
            dry_run: True 时只报告不执行
        """
        with self.driver.session(database=self.database) as session:
            names = [
                row["name"]
                for row in session.run("MATCH (n:Entity) RETURN n.name AS name")
            ]
            if not names:
                print(" 数据库中没有实体节点")
                return

            print(f"扫描 {len(names)} 个实体节点，阈值={threshold}...")

            pairs = []
            for i, a in enumerate(names):
                for b in names[i + 1 :]:
                    if min(len(a), len(b)) < 2:
                        continue
                    if abs(len(a) - len(b)) > 3:
                        continue
                    sim = SequenceMatcher(None, a, b).ratio()
                    if sim >= threshold:
                        pairs.append((a, b, sim))

            if not pairs:
                print(" 未发现可合并的相似实体")
                return

            print(f"发现 {len(pairs)} 对相似实体：")
            merged = set()
            for a, b, sim in sorted(pairs, key=lambda x: -x[2]):
                if a in merged or b in merged:
                    continue
                # 查询两端度数，保留高度数节点
                deg_a = session.run(
                    "MATCH (n:Entity {name: $name})-[r]-() RETURN count(r) AS c",
                    name=a,
                ).single()["c"]
                deg_b = session.run(
                    "MATCH (n:Entity {name: $name})-[r]-() RETURN count(r) AS c",
                    name=b,
                ).single()["c"]
                keep, drop = (a, b) if deg_a >= deg_b else (b, a)
                print(f"  [{sim:.3f}] (\"{a}\" / \"{b}\") → 保留 \"{keep}\"，移除 \"{drop}\"")

                if not dry_run:
                    session.run(
                        """
                        MATCH (drop:Entity {name: $drop})
                        MATCH (keep:Entity {name: $keep})
                        CALL apoc.refactor.mergeNodes([keep, drop], {
                          properties: 'combine',
                          mergeRels: true
                        })
                        YIELD node
                        RETURN count(node) AS merged_nodes
                        """,
                        drop=drop,
                        keep=keep,
                    )
                merged.add(a)
                merged.add(b)

            if dry_run:
                print("\n 试运行模式，未实际合并。将 dry_run=False 以执行合并。")
            else:
                print(f"\n 合并完成")

    def cleanup_dangling(self, dry_run=True):
        """清理无任何关系的孤立实体节点"""
        with self.driver.session(database=self.database) as session:
            count = session.run(
                "MATCH (n:Entity) WHERE NOT (n)--() RETURN count(n) AS c"
            ).single()["c"]

            if count == 0:
                print(" 没有孤立节点")
                return 0

            print(f"发现 {count} 个孤立节点（无任何关系）")
            if not dry_run:
                session.run("MATCH (n:Entity) WHERE NOT (n)--() DELETE n")
                print(f" 已删除 {count} 个孤立节点")
            else:
                print(" 试运行模式，未实际删除。将 dry_run=False 以执行删除。")
            return count

    def run_kb_cleanup(self, merge_threshold=0.85, dry_run=True):
        """一键运行知识库清理：去孤点 → 实体合并"""
        print("=" * 60)
        print(" 知识库质量清理")
        print("=" * 60)
        self.cleanup_dangling(dry_run=dry_run)
        print()
        self.merge_similar_entities(threshold=merge_threshold, dry_run=dry_run)
        print("=" * 60)

    def clear_database(self):
        """清空数据库（谨慎使用！）"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        print(" 数据库已清空")

    def _load_triples_from_file(self, json_file):
        """读取单个 JSON 文件中的三元组"""
        with open(json_file, 'r', encoding='utf-8') as f:
            triples = json.load(f)

        if not isinstance(triples, list):
            raise ValueError(f"JSON 内容不是列表: {json_file}")

        return triples

    def _get_json_files(self, json_dir):
        """获取需要导入的 JSON 文件列表"""
        json_files = sorted(Path(json_dir).glob('**/*.json'))

        if not json_files:
            return []

        # 兼容旧版总文件，若已有新的分文件结果则跳过旧总文件，避免重复导入
        if len(json_files) > 1 and LEGACY_OUTPUT_FILE in json_files:
            json_files = [path for path in json_files if path != LEGACY_OUTPUT_FILE]

        # 排除追踪文件和聊天记录等非三元组文件
        json_files = [
            path for path in json_files
            if path.name not in ("import_tracker.json", "chat_history.json")
        ]

        return json_files
    
    def import_triples(self, json_file, batch_size=50):
        """批量导入三元组（每批一个事务）"""
        triples = self._load_triples_from_file(json_file)
        valid = _build_triple_batch(triples)

        total_raw = len(triples)
        total_valid = len(valid)
        total_invalid = total_raw - total_valid

        if not valid:
            print(f" 无有效三元组可导入（原始 {total_raw} 条，全部无效）")
            return total_raw, total_valid, 0

        print(f"准备导入 {total_valid} 个三元组（共 {total_raw} 条原始数据，跳过 {total_invalid} 条无效）...")

        now = datetime.now(timezone.utc).isoformat()

        def _batch_import(tx, batch):
            for head, relation, tail, source, rel_clean in batch:
                tx.run(
                    f"""
                    MERGE (h:Entity {{name: $head}})
                      ON CREATE SET h.source = $source, h.created_at = $now
                      ON MATCH SET h.updated_at = $now
                    MERGE (t:Entity {{name: $tail}})
                      ON CREATE SET t.source = $source, t.created_at = $now
                      ON MATCH SET t.updated_at = $now
                    MERGE (h)-[r:{rel_clean}]->(t)
                      ON CREATE SET r.original_relation = $relation,
                                    r.source = $source,
                                    r.created_at = $now
                      ON MATCH SET r.updated_at = $now
                    """,
                    head=head, tail=tail, relation=relation,
                    source=source, now=now,
                )

        success_count = 0
        batches = [valid[i:i + batch_size] for i in range(0, len(valid), batch_size)]

        with self.driver.session(database=self.database) as session:
            for idx, batch in enumerate(batches):
                try:
                    session.execute_write(_batch_import, batch)
                    success_count += len(batch)
                    print(f"  已导入 {success_count}/{total_valid}...")
                except Exception as e:
                    print(f" 批次 {idx + 1} 导入失败: {e}")

        print(f" 导入完成！原始: {total_raw}, 有效: {total_valid}, 成功导入: {success_count}")
        return total_raw, total_valid, success_count

    def _load_tracker(self):
        """加载已导入文件追踪记录"""
        if IMPORT_TRACKER.exists():
            try:
                return json.loads(IMPORT_TRACKER.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_tracker(self, tracker):
        """保存已导入文件追踪记录"""
        IMPORT_TRACKER.parent.mkdir(parents=True, exist_ok=True)
        IMPORT_TRACKER.write_text(
            json.dumps(tracker, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def import_all_triples(self, json_dir=ANNOTATIONS_DIR):
        """批量导入目录下的所有三元组文件（自动跳过已导入的）"""
        json_files = self._get_json_files(json_dir)

        if not json_files:
            print(f" 没有找到可导入的 JSON 文件！目录: {json_dir}")
            return 0

        tracker = self._load_tracker()
        pending = [
            f for f in json_files
            if str(f.relative_to(ANNOTATIONS_DIR)) not in tracker
        ]
        skipped_count = len(json_files) - len(pending)

        print(f"连接 URI: {os.getenv('NEO4J_URI')}")
        print(f"使用数据库: {self.database}")
        print(f"找到 {len(json_files)} 个 JSON 文件 — 已导入 {skipped_count}, 待导入 {len(pending)}")
        print("=" * 60)

        if not pending:
            print(" 所有文件均已导入，无需操作。")
            return 0

        total_success = 0
        processed_files = 0

        for json_file in pending:
            relative_name = str(json_file.relative_to(ANNOTATIONS_DIR))
            print(f"\n 导入文件: {relative_name}")
            total_raw, total_valid, success_count = self.import_triples(json_file)
            total_success += success_count
            processed_files += 1
            tracker[relative_name] = {
                "raw": total_raw,
                "valid": total_valid,
                "imported": success_count,
                "imported_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_tracker(tracker)

        print("\n" + "=" * 60)
        print(f" 本次导入 {processed_files} 个文件, {total_success} 个三元组")
        print(f" 累计已导入: {len(tracker)} 个文件")
        print("=" * 60)

        return total_success
    
    def show_stats(self):
        """显示图谱统计信息"""
        with self.driver.session(database=self.database) as session:
            # 节点数量
            result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = result.single()['count']
            
            # 关系数量
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()['count']
            
            # 关系类型
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(*) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            
            print("\n" + "=" * 60)
            print(" 知识图谱统计")
            print("=" * 60)
            print(f"数据库: {self.database}")
            print(f"节点数量: {node_count}")
            print(f"关系数量: {rel_count}")
            print("\n常见关系类型:")
            for record in result:
                print(f"  • {record['rel_type']}: {record['count']}条")
            print("=" * 60)
    
    def show_sample_data(self, limit=10):
        """显示示例数据"""
        with self.driver.session(database=self.database) as session:
            result = session.run(f"""
                MATCH (h)-[r]->(t)
                RETURN h.name as head, 
                       r.original_relation as relation,
                       t.name as tail
                LIMIT {limit}
            """)
            
            print("\n" + "=" * 60)
            print(" 示例知识三元组")
            print("=" * 60)
            for i, record in enumerate(result, 1):
                print(f"{i}. ({record['head']}) -[{record['relation']}]-> ({record['tail']})")
            print("=" * 60)


def main():
    """主函数"""
    importer = TripleImporter()
    
    # 询问是否清空数据库
    print(" 是否清空现有数据库？(y/n)")
    choice = input("请选择: ").strip().lower()
    
    if choice == 'y':
        importer.clear_database()
    
    # 导入三元组
    importer.import_all_triples()
    
    # 显示统计信息
    importer.show_stats()
    
    # 显示示例数据
    importer.show_sample_data(15)
    
    importer.close()
    
    print("\n 知识图谱构建完成！")
    print("现在可以：")
    print("1. 在Neo4j Browser中查看图谱: http://localhost:7474")
    print("2. 或在Aura控制台查看")
    print("3. 运行可视化脚本: python kg/visualizer.py")


if __name__ == "__main__":
    main()

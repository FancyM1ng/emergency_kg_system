"""将三元组导入Neo4j知识图谱"""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / "config" / ".env"
ANNOTATIONS_DIR = BASE_DIR / "data" / "annotations"
LEGACY_OUTPUT_FILE = ANNOTATIONS_DIR / "自动提取三元组.json"

load_dotenv(ENV_FILE)


def _build_triple_batch(triples):
    """预处理三元组，过滤无效数据并清理关系名"""
    valid = []
    for t in triples:
        head = t.get("head", "").strip()
        relation = t.get("relation", "").strip()
        tail = t.get("tail", "").strip()
        source = t.get("source", "")
        if not head or not relation or not tail:
            continue
        # 清理关系名（Neo4j 不能有特殊字符）
        clean = re.sub(r"[^\w]+", "_", relation, flags=re.UNICODE).strip("_")
        if clean and clean[0].isdigit():
            clean = f"REL_{clean}"
        if not clean:
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
    
    def clear_database(self):
        """清空数据库（谨慎使用！）"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✅ 数据库已清空")

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

        return json_files
    
    def import_triples(self, json_file, batch_size=50):
        """批量导入三元组（每批一个事务）"""
        triples = self._load_triples_from_file(json_file)
        valid = _build_triple_batch(triples)

        if not valid:
            print(f"⚠️ 无有效三元组可导入")
            return 0

        print(f"准备导入 {len(valid)} 个三元组（共 {len(triples)} 条原始数据）...")
        print("=" * 60)

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
                    print(f"  已导入 {success_count}/{len(valid)}...")
                except Exception as e:
                    print(f"⚠️ 批次 {idx + 1} 导入失败: {e}")

        print(f"✅ 导入完成！成功: {success_count}, 跳过无效: {len(triples) - len(valid)}")
        return success_count

    def import_all_triples(self, json_dir=ANNOTATIONS_DIR):
        """批量导入目录下的所有三元组文件"""
        json_files = self._get_json_files(json_dir)

        if not json_files:
            print(f"⚠️ 没有找到可导入的 JSON 文件！目录: {json_dir}")
            return 0

        print(f"连接 URI: {os.getenv('NEO4J_URI')}")
        print(f"使用数据库: {self.database}")
        print(f"找到 {len(json_files)} 个 JSON 文件，开始导入...")
        print("=" * 60)

        total_success = 0
        processed_files = 0

        for json_file in json_files:
            relative_name = json_file.relative_to(ANNOTATIONS_DIR)
            print(f"\n📥 导入文件: {relative_name}")
            total_success += self.import_triples(json_file)
            processed_files += 1

        print("\n" + "=" * 60)
        print(f"🎉 目录导入完成！共处理 {processed_files} 个文件，导入 {total_success} 个三元组")
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
            print("📊 知识图谱统计")
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
            print("🔍 示例知识三元组")
            print("=" * 60)
            for i, record in enumerate(result, 1):
                print(f"{i}. ({record['head']}) -[{record['relation']}]-> ({record['tail']})")
            print("=" * 60)


def main():
    """主函数"""
    importer = TripleImporter()
    
    # 询问是否清空数据库
    print("⚠️ 是否清空现有数据库？(y/n)")
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
    
    print("\n🎉 知识图谱构建完成！")
    print("现在可以：")
    print("1. 在Neo4j Browser中查看图谱: http://localhost:7474")
    print("2. 或在Aura控制台查看")
    print("3. 运行可视化脚本: python kg/visualizer.py")


if __name__ == "__main__":
    main()

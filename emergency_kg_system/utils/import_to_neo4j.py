"""将三元组导入Neo4j知识图谱"""
import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / "config" / ".env"
ANNOTATIONS_DIR = BASE_DIR / "data" / "annotations"
LEGACY_OUTPUT_FILE = ANNOTATIONS_DIR / "自动提取三元组.json"

load_dotenv(ENV_FILE)

class TripleImporter:
    """三元组导入器"""
    
    def __init__(self):
        """初始化Neo4j连接"""
        self.database = os.getenv('NEO4J_DATABASE', 'dc14aaed')
        self.driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI'),
            auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
        )
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
    
    def import_triples(self, json_file):
        """批量导入三元组"""
        triples = self._load_triples_from_file(json_file)
        
        print(f"准备导入 {len(triples)} 个三元组...")
        print("=" * 60)
        
        success_count = 0
        error_count = 0
        
        with self.driver.session(database=self.database) as session:
            for i, triple in enumerate(triples, 1):
                try:
                    head = triple.get('head', '').strip()
                    relation = triple.get('relation', '').strip()
                    tail = triple.get('tail', '').strip()
                    source = triple.get('source', '')
                    
                    # 跳过空值
                    if not head or not relation or not tail:
                        continue
                    
                    # 清理关系名（Neo4j关系名不能有特殊字符）
                    relation_clean = self._clean_relation(relation)
                    
                    # 创建节点和关系
                    query = f"""
                    MERGE (h {{name: $head}})
                    SET h:Entity
                    MERGE (t {{name: $tail}})
                    SET t:Entity
                    MERGE (h)-[r:{relation_clean}]->(t)
                    SET h.source = $source,
                        t.source = $source,
                        r.original_relation = $relation,
                        r.source = $source
                    """
                    
                    session.run(query, head=head, tail=tail, 
                               relation=relation, source=source)
                    
                    success_count += 1
                    
                    if i % 20 == 0:
                        print(f"  已导入 {i}/{len(triples)} 个...")
                    
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:  # 只显示前5个错误
                        print(f"⚠️ 导入失败: {triple} - {e}")
        
        print(f"✅ 导入完成！")
        print(f"  成功: {success_count}")
        print(f"  失败: {error_count}")
        
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
    
    def _clean_relation(self, relation):
        """清理关系名，使其符合Neo4j规范"""
        # 将非法字符统一替换为下划线，保留中文、字母、数字和下划线
        relation = re.sub(r"[^\w]+", "_", relation, flags=re.UNICODE).strip("_")

        # 未转义的关系类型不能以数字开头
        if relation and relation[0].isdigit():
            relation = f"REL_{relation}"

        # 如果为空，使用默认关系
        if not relation:
            relation = "RELATED_TO"

        return relation
    
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

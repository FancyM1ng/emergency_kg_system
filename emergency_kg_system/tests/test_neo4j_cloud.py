"""测试Neo4j连接"""
from neo4j import GraphDatabase

class Neo4jTest:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def test_connection(self):
        """测试连接"""
        with self.driver.session() as session:
            result = session.run("RETURN 'Hello Neo4j!' as message")
            print(result.single()['message']) # type: ignore
    
    def create_sample_data(self):
        """创建示例数据"""
        with self.driver.session() as session:
            # 创建节点
            session.run("""
                CREATE (e:Event {name: '地震', magnitude: 7.0})
                CREATE (m:Measure {name: '紧急疏散'})
                CREATE (e)-[:REQUIRES]->(m)
            """)
            print("✅ 示例数据创建成功")
    
    def query_data(self):
        """查询数据"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Event)-[:REQUIRES]->(m:Measure)
                RETURN e.name as event, m.name as measure
            """)
            
            print("\n查询结果：")
            for record in result:
                print(f"  {record['event']} 需要 {record['measure']}")

if __name__ == "__main__":
    # 连接配置（改成你的密码）
    neo4j = Neo4jTest(
        uri="neo4j+s://a73a6418.databases.neo4j.io",
        user="neo4j",
        password="qbC6OgOqEyrnVG_JNCvT9nCXLYAw6NYTNicV1XVAk8Y"  # ← 你设置的密码
    )
    
    # 测试连接
    print("测试连接...")
    neo4j.test_connection()
    
    # 创建数据
    print("\n创建示例数据...")
    neo4j.create_sample_data()
    
    # 查询数据
    neo4j.query_data()
    
    # 关闭连接
    neo4j.close()
    print("\n✅ 测试完成！")

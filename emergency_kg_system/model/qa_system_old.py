"""应急知识图谱问答系统"""
import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
import requests
import jieba

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / "config" / ".env"

class EmergencyQASystem:
    """应急问答系统"""
    
    def __init__(self):
        """初始化"""
        load_dotenv(ENV_FILE)
        
        # Neo4j连接
        self.driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI'),
            auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
        )
        
        # API配置
        self.api_key = os.getenv('DEEPSEEK_API_KEY') or os.getenv('ZHIPU_API_KEY')
        self.api_url = os.getenv('DEEPSEEK_API_URL', "https://api.deepseek.com/chat/completions")
        self.api_model = os.getenv('DEEPSEEK_MODEL', "deepseek-chat")
        
        print("✅ 问答系统初始化完成")
    
    def close(self):
        """关闭连接"""
        self.driver.close()
    
    def answer_question(self, question):
        """回答问题的完整流程"""
        
        print("\n" + "=" * 60)
        print(f"❓ 问题: {question}")
        print("=" * 60)
        
        # Step 1: 提取关键词
        print("\n🔍 正在分析问题...")
        keywords = self._extract_keywords(question)
        print(f"   关键词: {', '.join(keywords)}")
        
        # Step 2: 检索知识
        print("\n📚 正在检索知识图谱...")
        knowledge = self._search_knowledge(keywords)
        print(f"   找到 {len(knowledge)} 条相关知识")
        
        # Step 3: 格式化知识
        knowledge_text = self._format_knowledge(knowledge)
        
        # Step 4: 构建Prompt
        prompt = self._build_prompt(question, knowledge_text)
        
        # Step 5: 调用API
        print("\n💭 正在生成答案...")
        answer = self._call_api(prompt)
        
        return {
            'answer': answer,
            'knowledge': knowledge,
            'keywords': keywords
        }
    
    def _extract_keywords(self, text):
        """提取关键词"""
        words = jieba.cut(text)
        stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', 
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '应该',
            '哪些', '什么', '怎么', '如何', '吗', '呢', '能', '可以'
        }
        keywords = [w for w in words if len(w) > 1 and w not in stopwords]
        return keywords[:5]  # 只取前5个关键词
    
    def _search_knowledge(self, keywords):
        """从Neo4j检索知识"""
        if not keywords:
            return []
        
        with self.driver.session() as session:
            # 构建查询条件
            conditions = " OR ".join([f"h.name CONTAINS '{kw}' OR t.name CONTAINS '{kw}'" for kw in keywords])
            
            query = f"""
            MATCH (h)-[r]->(t)
            WHERE {conditions}
            RETURN h.name as head, 
                   type(r) as relation,
                   r.original_relation as original_relation,
                   t.name as tail
            LIMIT 15
            """
            
            try:
                result = session.run(query)
                
                knowledge = []
                for record in result:
                    knowledge.append({
                        'head': record['head'],
                        'relation': record.get('original_relation') or record['relation'],
                        'tail': record['tail']
                    })
                
                return knowledge
            except Exception as e:
                print(f"⚠️ 检索失败: {e}")
                return []
    
    def _format_knowledge(self, knowledge_list):
        """格式化知识"""
        if not knowledge_list:
            return "未找到相关知识。"
        
        formatted = "从应急知识图谱中检索到的相关知识：\n\n"
        for i, item in enumerate(knowledge_list, 1):
            formatted += f"{i}. {item['head']} -[{item['relation']}]-> {item['tail']}\n"
        
        return formatted
    
    def _build_prompt(self, question, knowledge):
        """构建Prompt"""
        return f"""你是一个专业的应急管理助手，擅长提供应急处置建议。

{knowledge}

用户问题：{question}

请基于上述知识图谱中的信息，给出专业、详细的应急处置建议。要求：
1. 结构清晰，分点说明
2. 突出重点措施和注意事项
3. 语言专业但易于理解
4. 如果知识图谱信息不足，可以适当补充你的专业知识
5. 回答要实用、可操作

请开始回答："""
    
    def _call_api(self, prompt):
        """调用智谱API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.api_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"❌ API调用失败: {str(e)}"


def main():
    """主函数 - 测试问答系统"""
    qa = EmergencyQASystem()
    
    # 测试问题列表
    test_questions = [
        "企业在安全生产中有哪些职责？",
        "如何预防触电事故？",
        "违章操作会导致什么后果？",
        "企业需要配备哪些安全设施？"
    ]
    
    print("\n" + "🚨" * 30)
    print("应急知识图谱问答系统测试")
    print("🚨" * 30)
    
    # 让用户选择问题或自己输入
    print("\n预设问题：")
    for i, q in enumerate(test_questions, 1):
        print(f"{i}. {q}")
    print("5. 自己输入问题")
    
    choice = input("\n请选择 (1-5): ").strip()
    
    if choice == '5':
        question = input("请输入您的问题: ").strip()
    elif choice in ['1', '2', '3', '4']:
        question = test_questions[int(choice) - 1]
    else:
        question = test_questions[0]  # 默认第一个
    
    # 回答问题
    result = qa.answer_question(question)
    
    # 显示结果
    print("\n" + "=" * 60)
    print("💡 答案:")
    print("=" * 60)
    print(result['answer'])
    
    if result['knowledge']:
        print("\n" + "=" * 60)
        print("📚 知识来源:")
        print("=" * 60)
        for i, k in enumerate(result['knowledge'][:5], 1):
            print(f"{i}. ({k['head']}) -[{k['relation']}]-> ({k['tail']})")
    
    qa.close()
    print("\n" + "=" * 60)
    print("✅ 测试完成！")


if __name__ == "__main__":
    main()

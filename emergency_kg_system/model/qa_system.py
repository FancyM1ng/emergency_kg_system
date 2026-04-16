"""应急知识图谱问答系统（集成BERT）"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
import requests
import jieba
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
# 导入BERT处理器
from utils.bert_processor import BERTProcessor

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / "config" / ".env"


class EmergencyQASystem:
    """应急问答系统"""
    
    def __init__(self):
        """初始化"""
        load_dotenv(ENV_FILE)
        
        # Neo4j连接
        self.database = os.getenv('NEO4J_DATABASE', 'neo4j')
        self.driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI'),
            auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
        )
        
        # API配置
        self.api_key = os.getenv('DEEPSEEK_API_KEY') or os.getenv('ZHIPU_API_KEY')
        self.api_url = os.getenv('DEEPSEEK_API_URL', "https://api.deepseek.com/chat/completions")
        self.api_model = os.getenv('DEEPSEEK_MODEL', "deepseek-chat")
        
        # 🆕 初始化BERT
        print("正在初始化BERT模型...")
        self.bert = BERTProcessor('bert-base-chinese')
        
        print("✅ 问答系统初始化完成（已启用BERT语义检索）")
    
    def close(self):
        """关闭连接"""
        self.driver.close()
    
    def answer_question(self, question, use_bert=True):
        """
        回答问题的完整流程
        
        Args:
            question: 用户问题
            use_bert: 是否使用BERT语义检索（默认True）
        """
        
        print("\n" + "=" * 60)
        print(f"❓ 问题: {question}")
        print("=" * 60)
        
        # Step 1: 提取关键词
        print("\n🔍 正在分析问题...")
        keywords = self._extract_keywords(question)
        print(f"   关键词: {', '.join(keywords)}")
        
        # Step 2: 检索知识
        if use_bert:
            print("\n📚 正在使用BERT语义检索...")
            knowledge = self._search_knowledge_bert(question, keywords)
        else:
            print("\n📚 正在使用关键词检索...")
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
        return keywords[:5]
    
    def _search_knowledge(self, keywords):
        """传统关键词检索"""
        if not keywords:
            return []
        
        with self.driver.session(database=self.database) as session:
            conditions = " OR ".join([f"h.name CONTAINS '{kw}' OR t.name CONTAINS '{kw}'" for kw in keywords])
            
            query = f"""
            MATCH (h)-[r]->(t)
            WHERE {conditions}
            RETURN h.name as head, 
                   type(r) as relation,
                   r.original_relation as original_relation,
                   t.name as tail,
                   coalesce(r.source, h.source, t.source) as source
            LIMIT 15
            """
            
            try:
                result = session.run(query)
                
                knowledge = []
                for record in result:
                    knowledge.append({
                        'head': record['head'],
                        'relation': record.get('original_relation') or record['relation'],
                        'tail': record['tail'],
                        'source': record.get('source') or '未知来源'
                    })
                
                return knowledge
            except Exception as e:
                print(f"⚠️ 检索失败: {e}")
                return []
    
    def _search_knowledge_bert(self, question, keywords):
        """
        🆕 BERT语义检索（推荐）
        
        流程：
        1. 先用关键词检索获取候选知识（30条）
        2. 用BERT计算每条知识与问题的语义相似度
        3. 按相似度排序，返回最相关的10条
        """
        # Step 1: 获取候选知识（扩大范围到30条）
        candidates = []
        
        with self.driver.session(database=self.database) as session:
            if keywords:
                conditions = " OR ".join([
                    f"h.name CONTAINS '{kw}' OR t.name CONTAINS '{kw}'" 
                    for kw in keywords
                ])
                
                query = f"""
                MATCH (h)-[r]->(t)
                WHERE {conditions}
                RETURN h.name as head,
                       type(r) as relation,
                       r.original_relation as original_relation,
                       t.name as tail,
                       coalesce(r.source, h.source, t.source) as source
                LIMIT 30
                """
            else:
                # 如果没有关键词，随机取一些
                query = """
                MATCH (h)-[r]->(t)
                RETURN h.name as head,
                       type(r) as relation,
                       r.original_relation as original_relation,
                       t.name as tail,
                       coalesce(r.source, h.source, t.source) as source
                LIMIT 30
                """
            
            try:
                result = session.run(query)
                for record in result:
                    candidates.append({
                        'head': record['head'],
                        'relation': record.get('original_relation') or record['relation'],
                        'tail': record['tail'],
                        'source': record.get('source') or '未知来源'
                    })
            except Exception as e:
                print(f"⚠️ 候选检索失败: {e}")
                return []
        
        if not candidates:
            return []
        
        # Step 2: 用BERT计算相似度
        print(f"   → 对 {len(candidates)} 条候选知识进行语义排序...")
        
        scored_knowledge = []
        for k in candidates:
            # 将三元组拼接成句子
            k_text = f"{k['head']} {k['relation']} {k['tail']}"
            
            # 计算相似度
            similarity = self.bert.similarity(question, k_text)
            
            scored_knowledge.append({
                'knowledge': k,
                'similarity': similarity
            })
        
        # Step 3: 按相似度排序
        scored_knowledge.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 返回最相关的10条
        top_knowledge = [item['knowledge'] for item in scored_knowledge[:10]]
        
        # 打印相似度（调试用）
        print("\n   相似度TOP5:")
        for i, item in enumerate(scored_knowledge[:5], 1):
            k = item['knowledge']
            sim = item['similarity']
            print(f"   {i}. [{sim:.3f}] ({k['head']})-[{k['relation']}]->({k['tail']})")
        
        return top_knowledge
    
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
    """主函数 - 测试BERT检索效果"""
    qa = EmergencyQASystem()
    
    print("\n" + "🚨" * 30)
    print("应急知识图谱问答系统测试（BERT增强版）")
    print("🚨" * 30)
    
    # 测试问题
    test_questions = [
        "企业在安全生产中有哪些职责？",
        "如何预防触电事故？",
        "违章操作会导致什么后果？",
        "企业需要配备哪些安全设施？"
    ]
    
    print("\n预设问题：")
    for i, q in enumerate(test_questions, 1):
        print(f"{i}. {q}")
    print("5. 自己输入问题")
    print("6. 对比测试（关键词 vs BERT）")
    
    choice = input("\n请选择 (1-6): ").strip()
    
    if choice == '5':
        question = input("请输入您的问题: ").strip()
        result = qa.answer_question(question, use_bert=True)
        
        print("\n" + "=" * 60)
        print("💡 答案:")
        print("=" * 60)
        print(result['answer'])
        
    elif choice == '6':
        # 对比测试
        question = test_questions[0]
        
        print(f"\n对比测试问题: {question}")
        print("\n" + "▶️" * 30)
        print("方法1: 关键词检索")
        print("▶️" * 30)
        result1 = qa.answer_question(question, use_bert=False)
        
        print("\n" + "▶️" * 30)
        print("方法2: BERT语义检索")
        print("▶️" * 30)
        result2 = qa.answer_question(question, use_bert=True)
        
        print("\n" + "=" * 60)
        print("📊 对比结果")
        print("=" * 60)
        print(f"关键词检索: 找到 {len(result1['knowledge'])} 条知识")
        print(f"BERT检索: 找到 {len(result2['knowledge'])} 条知识")
        
    else:
        idx = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= 4 else 0
        question = test_questions[idx]
        
        result = qa.answer_question(question, use_bert=True)
        
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

"""应急知识图谱问答系统（集成语义检索）"""
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests
import jieba

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from utils.bert_processor import BERTProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / "config" / ".env"


class EmergencyQASystem:
    """应急问答系统"""
    
    def __init__(self):
        """初始化"""
        load_dotenv(ENV_FILE)

        # Neo4j连接
        self.database = os.getenv('NEO4J_DATABASE', 'neo4j')
        from utils.neo4j_driver import create_driver
        self.driver = create_driver()

        # API配置
        self.api_key = os.getenv('DEEPSEEK_API_KEY') or os.getenv('ZHIPU_API_KEY')
        self.api_url = os.getenv('DEEPSEEK_API_URL', "https://api.deepseek.com/v1/chat/completions")
        self.api_model = os.getenv('DEEPSEEK_MODEL', "deepseek-chat")

        # 初始化语义检索模型
        logger.info("正在加载语义检索模型...")
        self.bert = BERTProcessor('BAAI/bge-base-zh-v1.5')

        # 多轮对话历史
        self.conversation_history = []

        logger.info("问答系统初始化完成（已启用语义检索）")

    def reset_conversation(self):
        """重置多轮对话历史"""
        self.conversation_history = []

    def close(self):
        """关闭连接"""
        self.driver.close()

    def answer_question(self, question, use_bert=True, stream=False):
        """
        回答问题的完整流程

        Args:
            question: 用户问题
            use_bert: 是否使用语义检索
            stream: 是否流式返回（返回生成器）
        """
        logger.info("问题: %s", question)

        # Step 1: 提取关键词
        keywords = self._extract_keywords(question)
        logger.info("关键词: %s", ', '.join(keywords))

        # Step 2: 检索知识（一跳 + 多跳融合）
        if use_bert:
            logger.info("正在使用语义检索...")
            single_hop = self._search_knowledge_bert(question, keywords)
            multi_hop = self._search_knowledge_multihop(question, keywords)
            # 融合：一跳取8条，多跳取4条，总计12条
            knowledge = single_hop[:8] + multi_hop[:4]
            logger.info("一跳 %d 条 + 多跳 %d 条 → 融合 %d 条",
                        len(single_hop), len(multi_hop), len(knowledge))
        else:
            logger.info("正在使用关键词检索...")
            knowledge = self._search_knowledge(keywords)

        logger.info("找到 %d 条相关知识", len(knowledge))

        # Step 3: 格式化知识
        knowledge_text = self._format_knowledge(knowledge)

        # Step 4: 构建消息列表（含历史对话）
        messages = self._build_messages(question, knowledge_text)

        # Step 5: 调用API
        logger.info("正在生成答案...")
        if stream:
            answer = self._call_api_stream(messages)
        else:
            answer = self._call_api(messages)

        return {
            'answer': answer,
            'knowledge': knowledge,
            'keywords': keywords,
        }

    def add_to_history(self, question, answer):
        """将一轮对话加入历史"""
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})
        # 保留最近10轮
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def _extract_keywords(self, text):
        """基于 TF-IDF 提取关键词"""
        keywords = jieba.analyse.extract_tags(text, topK=8, allowPOS=())
        return keywords
    
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
                logger.warning("检索失败: %s", e)
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
                logger.warning("候选检索失败: %s", e)
                return []
        
        if not candidates:
            return []

        # Step 2: 去重（同一三元组可能从不同文件导入产生重复）
        seen = set()
        unique_candidates = []
        for k in candidates:
            key = (k['head'], k['relation'], k['tail'])
            if key not in seen:
                seen.add(key)
                unique_candidates.append(k)

        # Step 3: 用语义模型批量计算相似度
        logger.debug("对 %d 条候选知识进行语义排序...", len(unique_candidates))

        k_texts = [f"{k['head']} {k['relation']} {k['tail']}" for k in unique_candidates]
        similarities = self.bert.batch_similarity(question, k_texts)

        scored_knowledge = [
            {'knowledge': k, 'similarity': sim}
            for k, sim in zip(unique_candidates, similarities)
        ]

        # Step 4: 按相似度排序
        scored_knowledge.sort(key=lambda x: x['similarity'], reverse=True)

        # 返回最相关的10条
        top_knowledge = [item['knowledge'] for item in scored_knowledge[:10]]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("相似度TOP5:")
            for i, item in enumerate(scored_knowledge[:5], 1):
                k = item['knowledge']
                logger.debug("  %d. [%.3f] (%s)-[%s]->(%s)",
                             i, item['similarity'], k['head'], k['relation'], k['tail'])

        return top_knowledge

    def _search_knowledge_multihop(self, question, keywords, max_hops=3, limit=25):
        """多跳路径检索 — 从关键词命中实体出发做 BFS 扩展

        返回 2-3 跳路径，发现事故→原因→措施→责任人的深层关联。
        """
        candidates = []

        with self.driver.session(database=self.database) as session:
            if not keywords:
                return []

            # 从关键词命中的实体出发，做 2-3 跳 BFS 扩展
            try:
                result = session.run(
                    """
                    MATCH path = (start:Entity)-[*2..3]->(end:Entity)
                    WHERE any(kw IN $keywords WHERE start.name CONTAINS kw)
                    RETURN
                      [n in nodes(path) | n.name] AS nodes,
                      [r in relationships(path) |
                        coalesce(r.original_relation, type(r))] AS rels,
                      [r in relationships(path) | r.source] AS sources
                    LIMIT $limit
                    """,
                    keywords=keywords, limit=limit,
                )

                seen_paths = set()
                for record in result:
                    nodes = record["nodes"]
                    rels = record["rels"]
                    sources = record["sources"]

                    if len(nodes) < 2:
                        continue

                    # 路径去重（同一组节点序列视为相同路径）
                    path_key = tuple(nodes)
                    if path_key in seen_paths:
                        continue
                    seen_paths.add(path_key)

                    # 格式化为链式文本
                    chain_parts = []
                    for i in range(len(nodes) - 1):
                        chain_parts.append(f"({nodes[i]})-[{rels[i]}]")
                    chain_parts.append(f"->({nodes[-1]})")
                    path_text = " ".join(chain_parts)

                    source = sources[0] if sources else "未知来源"
                    candidates.append({
                        "head": nodes[0],
                        "relation": " → ".join(rels),
                        "tail": nodes[-1],
                        "source": source,
                        "path_text": path_text,
                        "is_multihop": True,
                        "hops": len(nodes) - 1,
                    })

            except Exception as e:
                logger.warning("多跳检索失败: %s", e)
                return []

        if not candidates:
            return []

        # BERT 语义排序
        path_texts = [c["path_text"] for c in candidates]
        similarities = self.bert.batch_similarity(question, path_texts)

        scored = [
            {"knowledge": c, "similarity": sim}
            for c, sim in zip(candidates, similarities)
        ]
        scored.sort(key=lambda x: x["similarity"], reverse=True)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("多跳路径TOP3:")
            for i, item in enumerate(scored[:3], 1):
                c = item["knowledge"]
                logger.debug("  %d. [%.3f] %s", i, item["similarity"], c["path_text"])

        return [item["knowledge"] for item in scored]

    def _format_knowledge(self, knowledge_list):
        """格式化知识（支持一跳三元组和多跳路径）"""
        if not knowledge_list:
            return "（知识库中暂无与该问题直接匹配的记录，请基于专业知识作答。）"

        parts = []
        singles = [k for k in knowledge_list if not k.get("is_multihop")]
        multihop = [k for k in knowledge_list if k.get("is_multihop")]

        if singles:
            parts.append("【直接关联三元组】")
            for i, item in enumerate(singles, 1):
                parts.append(
                    f"{i}. {item['head']} ——{item['relation']}——> {item['tail']}"
                )

        if multihop:
            parts.append("\n【多跳推理链】")
            for i, item in enumerate(multihop, 1):
                display = item.get("path_text",
                    f"{item['head']} ——{item['relation']}——> {item['tail']}")
                parts.append(f"{i}. {display}")

        return "\n".join(parts)
    
    def _build_messages(self, question, knowledge):
        """构建消息列表，包含系统提示、历史对话和当前问题"""
        system_prompt = """你是一名持证安全工程师兼应急处置专家，通过知识图谱辅助为企业提供合规、可落地的安全指导。

## 回答原则
1. 优先引用【知识来源】中的结构化信息，信息不足时用你的专业知识补齐，但需注明哪些是知识库信息、哪些是补充建议
2. 先分析事故机理或风险根源，再给出处置措施，避免罗列无关条目
3. 所有建议必须具体可执行（含检查频次、参照标准、责任主体、时限要求）
4. 按危害紧迫性排序：紧急处置 > 人员防护 > 工程控制 > 管理措施

## 输出格式
- 使用 ### 标题分级，关键动作用 **粗体** 突出
- 每条措施以数字序号列出，包含 做什么 → 谁来做 → 多久做一次 → 参照什么标准
- 如有必要，末尾添加「#注意事项」小节

## 安全底线
- 涉及人员生命安全的建议，须明确标注安全警示
- 涉及化学品、电气、有限空间等特殊作业，须标明需持证上岗"""

        # 用户消息：知识 + 问题（精简，不再放冗长示例）
        current_user = f"""【知识来源】
{knowledge}

【用户问题】
{question}

请基于上述知识给出结构化应急处置方案："""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": current_user})
        return messages

    def _call_api(self, messages):
        """调用大模型 API（非流式）"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.api_model,
            "messages": messages,
            "temperature": 0.5,
        }
        try:
            response = requests.post(
                self.api_url, headers=headers, json=data, timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"API调用失败: {str(e)}"

    def _call_api_stream(self, messages):
        """调用大模型 API（流式），返回生成器"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.api_model,
            "messages": messages,
            "temperature": 0.5,
            "stream": True,
        }
        try:
            response = requests.post(
                self.api_url, headers=headers, json=data, timeout=120, stream=True
            )
            response.raise_for_status()

            def generate():
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    chunk = line[6:]  # 去掉 "data: " 前缀
                    if chunk == "[DONE]":
                        break
                    try:
                        import json as _json
                        delta = _json.loads(chunk)["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except Exception:
                        continue

            return generate()
        except Exception as e:
            def error_gen():
                yield f"API调用失败: {str(e)}"
            return error_gen()


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

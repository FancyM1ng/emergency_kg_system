"""检索效果评估脚本 — 对比三种方法

用法:  python tests/eval_retrieval.py
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.qa_system import EmergencyQASystem

# 测试集 — 多跳相关的正例对也适当标注
TEST_SET = [
    (
        "如何预防KTV火灾事故？",
        [
            ("KTV火灾事故", "电暖器"),
            ("KTV火灾事故", "人员伤亡"),
            ("KTV火灾事故", "财产损失"),
            ("违规使用", "火灾事故"),
            # 多跳路径可能匹配的相关节点
            ("火灾事故应急指挥部", "全力处置火灾事故"),
        ],
    ),
    (
        "尾矿库安全需要注意什么？",
        [
            ("企业", "安全生产管控"),
            ("企业", "安全管理人员"),
            ("企业", "安全监管部门"),
            ("使用不安全设备", "临时使用不牢固的设施"),
            ("安全装置失效", "拆除了安全装置"),
        ],
    ),
    (
        "发生触电事故应该怎么处理？",
        [
            ("触电事故", "应急预案"),
            ("触电事故", "电"),
            ("事故", "应急措施"),
            ("三违", "事故"),
            ("节假日前后", "各种事故"),
        ],
    ),
    (
        "企业在安全生产中有哪些职责？",
        [
            ("企业", "安全生产管控"),
            ("企业", "安全责任承诺"),
            ("企业", "安全管理人员"),
            ("企业", "安全隐患的排查"),
            ("企业", "加强非生产一线的安全管理"),
        ],
    ),
]


def is_relevant(item, positive_pairs):
    """判断知识条目是否与任一正例对匹配"""
    head = item.get("head", "")
    tail = item.get("tail", "")
    path_text = item.get("path_text", "")

    for ph, pt in positive_pairs:
        # 一跳匹配
        if (ph in head and pt in tail) or (ph in tail and pt in head):
            return True
        # 多跳路径匹配：路径文本同时包含两个关键词
        if path_text and ph in path_text and pt in path_text:
            return True
    return False


def precision_at_k(retrieved, positives, k):
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for t in top_k if is_relevant(t, positives)) / k


def recall_at_k(retrieved, positives, k):
    if not positives:
        return 1.0
    top_k = retrieved[:k]
    return sum(1 for t in top_k if is_relevant(t, positives)) / len(positives)


def mrr(retrieved, positives):
    for i, t in enumerate(retrieved, 1):
        if is_relevant(t, positives):
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved, positives, k):
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    dcg = sum(
        (1.0 if is_relevant(t, positives) else 0.0) / math.log2(i + 2)
        for i, t in enumerate(top_k)
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(positives))))
    return dcg / ideal if ideal > 0 else 0.0


K_VALUES = [5, 10, 12]


def eval_method(qa, method_name, retrieve_fn):
    """评估单个检索方法"""
    print(f"\n{'─' * 50}")
    print(f"方法: {method_name}")
    print(f"{'─' * 50}")

    all_metrics = {k: {"precision": [], "recall": [], "ndcg": []} for k in K_VALUES}
    all_mrr = []

    for question, positives in TEST_SET:
        keywords = qa._extract_keywords(question)
        knowledge = retrieve_fn(qa, question, keywords)

        for k in K_VALUES:
            all_metrics[k]["precision"].append(precision_at_k(knowledge, positives, k))
            all_metrics[k]["recall"].append(recall_at_k(knowledge, positives, k))
            all_metrics[k]["ndcg"].append(ndcg_at_k(knowledge, positives, k))
        all_mrr.append(mrr(knowledge, positives))

        p10 = precision_at_k(knowledge, positives, 10)
        r10 = recall_at_k(knowledge, positives, 10)
        print(f"  {question[:35]}... → P@10={p10:.2f} R@10={r10:.2f}")

    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0
    print(f"\n  【{method_name} 平均指标】")
    for k in K_VALUES:
        m = all_metrics[k]
        print(f"  P@{k:<2} = {avg(m['precision']):.4f}  "
              f"R@{k:<2} = {avg(m['recall']):.4f}  "
              f"NDCG@{k:<2} = {avg(m['ndcg']):.4f}")
    print(f"  MRR    = {avg(all_mrr):.4f}")

    return {k: avg(all_metrics[k]["ndcg"]) for k in K_VALUES}, avg(all_mrr)


def run_eval():
    print("=" * 60)
    print("应急知识图谱检索效果评估 — 一跳 vs 多跳融合")
    print("=" * 60)

    qa = EmergencyQASystem()

    # 方法1: 关键词检索（仅一跳）
    eval_method(
        qa, "1. 关键词检索（仅一跳）",
        lambda qa, q, kw: qa._search_knowledge(kw),
    )

    # 方法2: 语义检索（仅一跳）
    eval_method(
        qa, "2. 语义检索 BGE（仅一跳）",
        lambda qa, q, kw: qa._search_knowledge_bert(q, kw),
    )

    # 方法3: 语义检索 + 多跳融合
    def blended(qa, q, kw):
        single = qa._search_knowledge_bert(q, kw)
        multi = qa._search_knowledge_multihop(q, kw)
        return single[:8] + multi[:4]

    eval_method(
        qa, "3. 语义检索 + 多跳融合",
        blended,
    )

    qa.close()
    print(f"\n{'=' * 60}")
    print("评估完成")
    print("=" * 60)


if __name__ == "__main__":
    run_eval()

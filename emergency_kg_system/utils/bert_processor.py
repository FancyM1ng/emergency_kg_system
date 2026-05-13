"""基于 sentence-transformers 的语义向量处理器"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = BASE_DIR / "data" / "model_cache" / "huggingface"

# ── 必须在任何 huggingface_hub 导入前设置镜像 ──
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class BERTProcessor:
    """文本向量化和语义匹配（兼容旧接口名）"""

    def __init__(self, model_name="BAAI/bge-base-zh-v1.5", cache_dir=None):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir or os.getenv("BERT_CACHE_DIR", DEFAULT_CACHE_DIR))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"正在加载模型: {model_name}")
        print(f"缓存目录: {self.cache_dir}")
        print("=" * 60)

        self.model = SentenceTransformer(
            model_name,
            cache_folder=str(self.cache_dir),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        dim = self.model.get_sentence_embedding_dimension()
        print(f"模型加载成功，向量维度: {dim}")
        print(f"最大长度: {self.model.max_seq_length} tokens")

    def get_embedding(self, text):
        """将文本转换为向量（默认 L2 归一化）"""
        if not text or not text.strip():
            return np.zeros(self.model.get_sentence_embedding_dimension())
        return self.model.encode(text, normalize_embeddings=True)

    def similarity(self, text1, text2):
        """两段文本的语义相似度（归一化后点积即 cosine）"""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        if np.linalg.norm(emb1) == 0 or np.linalg.norm(emb2) == 0:
            return 0.0
        return float(np.dot(emb1, emb2))

    def batch_similarity(self, query, texts):
        """批量计算查询与多个文本的相似度"""
        if not texts:
            return []
        query_emb = self.get_embedding(query)
        text_embs = self.model.encode(texts, normalize_embeddings=True)
        sims = np.dot(text_embs, query_emb)
        return sims.tolist()


if __name__ == "__main__":
    print("句子向量处理器测试")
    bert = BERTProcessor("BAAI/bge-base-zh-v1.5")

    print("\n" + "=" * 60)
    print("测试1: 文本向量化")
    print("=" * 60)
    text = "企业需要配备消防设施"
    embedding = bert.get_embedding(text)
    print(f"文本: {text}")
    print(f"向量维度: {embedding.shape}")
    print(f"向量范数: {np.linalg.norm(embedding):.4f}")

    print("\n" + "=" * 60)
    print("测试2: 语义相似度计算")
    print("=" * 60)
    test_cases = [
        ("如何预防火灾？", "火灾预防措施有哪些？"),
        ("如何预防火灾？", "怎样避免发生火灾事故？"),
        ("企业安全职责", "公司的安全生产责任"),
        ("企业安全职责", "化工爆炸怎么处理"),
        ("触电事故急救", "触电后的应急处理方法"),
    ]
    for text1, text2 in test_cases:
        sim = bert.similarity(text1, text2)
        print(f"\n文本1: {text1}")
        print(f"文本2: {text2}")
        print(f"相似度: {sim:.4f}")

    print("\n" + "=" * 60)
    print("测试3: 批量相似度计算")
    print("=" * 60)
    query = "如何预防火灾事故？"
    candidates = [
        "火灾预防措施包括配备灭火器",
        "触电事故的应急处理",
        "企业需要加强安全管理",
        "消防设施的配置要求",
        "化工爆炸的危害",
    ]
    print(f"查询: {query}\n")
    similarities = bert.batch_similarity(query, candidates)
    ranked = sorted(zip(candidates, similarities), key=lambda x: x[1], reverse=True)
    for i, (text, sim) in enumerate(ranked, 1):
        print(f"{i}. [{sim:.4f}] {text}")
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

"""基于 Hugging Face BERT 的文本处理器"""
from pathlib import Path
import os

import numpy as np
import torch
from transformers import BertModel, BertTokenizer


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = BASE_DIR / "data" / "model_cache" / "huggingface"


class BERTProcessor:
    """BERT 文本向量化和语义匹配"""

    def __init__(self, model_name="bert-base-chinese", cache_dir=None):
        """
        初始化 BERT 模型

        Args:
            model_name: 模型名称
            cache_dir: 本地缓存目录，默认保存在项目目录下
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir or os.getenv("BERT_CACHE_DIR", DEFAULT_CACHE_DIR))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("=" * 60)
        print(f"正在加载模型: {model_name}")
        print(f"缓存目录: {self.cache_dir}")
        print("首次运行会自动下载模型文件，后续会直接复用本地缓存")
        print("=" * 60)

        try:
            self.tokenizer = BertTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
            )
            self.model = BertModel.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
            )
            self.model.to(self.device)
            self.model.eval()

            print("模型加载成功")
            print("   模型维度: 768")
            print("   最大长度: 512 tokens")
            print(f"   运行设备: {self.device}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("请检查网络连接，或确认缓存目录是否可读写")
            raise

    def get_embedding(self, text):
        """
        将文本转换为 768 维向量

        Args:
            text: 输入文本

        Returns:
            numpy.ndarray: shape=(768,)
        """
        if not text or not text.strip():
            return np.zeros(768)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().cpu().numpy()
        return embedding

    def similarity(self, text1, text2):
        """
        计算两段文本的语义相似度
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        similarity = (similarity + 1) / 2
        return float(similarity)

    def batch_similarity(self, query, texts):
        """
        批量计算查询与多个文本的相似度
        """
        query_emb = self.get_embedding(query)

        similarities = []
        for text in texts:
            text_emb = self.get_embedding(text)
            dot = np.dot(query_emb, text_emb)
            norm_q = np.linalg.norm(query_emb)
            norm_t = np.linalg.norm(text_emb)

            if norm_q == 0 or norm_t == 0:
                sim = 0.0
            else:
                sim = dot / (norm_q * norm_t)
                sim = (sim + 1) / 2

            similarities.append(sim)

        return similarities


if __name__ == "__main__":
    print("BERT 处理器测试")
    bert = BERTProcessor("bert-base-chinese")

    print("\n" + "=" * 60)
    print("测试1: 文本向量化")
    print("=" * 60)

    text = "企业需要配备消防设施"
    embedding = bert.get_embedding(text)

    print(f"文本: {text}")
    print(f"向量维度: {embedding.shape}")
    print(f"向量前5维: {embedding[:5]}")
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
    print("测试完成，BERT 处理器工作正常")
    print("=" * 60)

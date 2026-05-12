"""问答系统核心模块测试"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from utils.extract_triples import deduplicate_triples, split_text_into_chunks, _parse_triples_response
from utils.import_to_neo4j import _build_triple_batch


class TestKeywordExtraction:
    """关键词提取测试"""

    def test_extract_tags_returns_list(self):
        import jieba.analyse
        keywords = jieba.analyse.extract_tags("企业在安全生产中有哪些职责", topK=8)
        assert isinstance(keywords, list)
        assert len(keywords) > 0

    def test_extract_tags_meaningful(self):
        import jieba.analyse
        keywords = jieba.analyse.extract_tags("如何预防触电事故", topK=8)
        assert any("触电" in kw or "事故" in kw or "预防" in kw for kw in keywords)

    def test_extract_tags_empty(self):
        import jieba.analyse
        keywords = jieba.analyse.extract_tags("", topK=8)
        assert keywords == []


class TestDeduplicateTriples:
    """三元组去重测试"""

    def test_empty_list(self):
        assert deduplicate_triples([]) == []

    def test_no_duplicates(self):
        triples = [
            {"head": "A", "relation": "R1", "tail": "B"},
            {"head": "C", "relation": "R2", "tail": "D"},
        ]
        result = deduplicate_triples(triples)
        assert len(result) == 2

    def test_with_duplicates(self):
        triples = [
            {"head": "A", "relation": "R1", "tail": "B"},
            {"head": "A", "relation": "R1", "tail": "B"},
            {"head": "C", "relation": "R2", "tail": "D"},
        ]
        result = deduplicate_triples(triples)
        assert len(result) == 2

    def test_different_sources_same_triple(self):
        """不同来源文件但三元组相同，应去重"""
        triples = [
            {"head": "A", "relation": "R1", "tail": "B", "source": "doc1"},
            {"head": "A", "relation": "R1", "tail": "B", "source": "doc2"},
        ]
        result = deduplicate_triples(triples)
        assert len(result) == 1

    def test_missing_fields(self):
        triples = [
            {"head": "A", "relation": "R1"},
            {"head": "A", "relation": "R1"},
        ]
        result = deduplicate_triples(triples)
        assert len(result) == 1


class TestParseTriplesResponse:
    """LLM 返回解析测试"""

    def test_plain_json_array(self):
        content = '[{"head":"A","relation":"R","tail":"B"}]'
        result = _parse_triples_response(content)
        assert len(result) == 1
        assert result[0]["head"] == "A"

    def test_markdown_code_block(self):
        content = '```json\n[{"head":"A","relation":"R","tail":"B"}]\n```'
        result = _parse_triples_response(content)
        assert len(result) == 1

    def test_no_markdown_lang(self):
        content = '```\n[{"head":"A","relation":"R","tail":"B"}]\n```'
        result = _parse_triples_response(content)
        assert len(result) == 1

    def test_with_extra_text(self):
        content = '这是提取结果：[{"head":"A","relation":"R","tail":"B"}]'
        result = _parse_triples_response(content)
        assert len(result) == 1

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_triples_response("no json here")

    def test_not_list_raises(self):
        with pytest.raises(ValueError):
            _parse_triples_response('{"head":"A","relation":"R","tail":"B"}')


class TestSplitTextIntoChunks:
    """文本分段测试"""

    def test_short_text_single_chunk(self):
        text = "一段短文本。"
        chunks = split_text_into_chunks(text, chunk_size=500, overlap=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        text = "一段测试文本。" * 200
        chunks = split_text_into_chunks(text, chunk_size=500, overlap=100)
        assert len(chunks) > 1

    def test_empty_text(self):
        chunks = split_text_into_chunks("", chunk_size=500, overlap=100)
        assert chunks == []

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            split_text_into_chunks("test", chunk_size=0, overlap=0)

    def test_overlap_too_large(self):
        with pytest.raises(ValueError):
            split_text_into_chunks("test", chunk_size=100, overlap=100)


class TestBuildTripleBatch:
    """三元组预处理测试"""

    def test_normal_triples(self):
        triples = [
            {"head": "企业", "relation": "负责", "tail": "安全生产", "source": "doc1.txt"},
            {"head": "泄漏事故", "relation": "由...引起", "tail": "违章操作", "source": "doc2.txt"},
        ]
        result = _build_triple_batch(triples)
        assert len(result) == 2
        assert result[1][4] == "由_引起"  # ... 替换为 _

    def test_empty_fields_filtered(self):
        triples = [
            {"head": "", "relation": "负责", "tail": "安全生产", "source": "doc1.txt"},
            {"head": "企业", "relation": "", "tail": "安全生产", "source": "doc1.txt"},
            {"head": "企业", "relation": "负责", "tail": "", "source": "doc1.txt"},
        ]
        result = _build_triple_batch(triples)
        assert len(result) == 0

    def test_numeric_relation_prefixed(self):
        triples = [
            {"head": "检查", "relation": "123排查", "tail": "隐患", "source": "doc1.txt"},
        ]
        result = _build_triple_batch(triples)
        assert result[0][4].startswith("REL_")

    def test_empty_relation_defaults(self):
        triples = [
            {"head": "检查", "relation": "...", "tail": "隐患", "source": "doc1.txt"},
        ]
        result = _build_triple_batch(triples)
        assert result[0][4] == "RELATED_TO"

    def test_blacklist_entity_filtered(self):
        triples = [
            {"head": "事故", "relation": "导致", "tail": "伤亡", "source": "doc1.txt"},
        ]
        result = _build_triple_batch(triples)
        assert len(result) == 0  # "事故" 在黑名单中

    def test_long_entity_filtered(self):
        triples = [
            {"head": "A" * 61, "relation": "测试", "tail": "B", "source": "doc1.txt"},
        ]
        result = _build_triple_batch(triples)
        assert len(result) == 0

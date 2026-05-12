"""应急知识图谱 — 知识三元组抽取"""
import os
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / "config" / ".env"
TEXT_DIR = BASE_DIR / "data" / "processed"
ANNOTATIONS_DIR = BASE_DIR / "data" / "annotations"

load_dotenv(ENV_FILE)

API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("ZHIPU_API_KEY")
API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
API_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.getenv("LLM_RETRY_BASE_DELAY", "2"))
CHUNK_PAUSE_SECONDS = float(os.getenv("LLM_CHUNK_PAUSE_SECONDS", "0.5"))
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "3"))
LLM_CHUNK_SIZE = int(os.getenv("LLM_CHUNK_SIZE", "12000"))
LLM_CHUNK_OVERLAP = max(200, LLM_CHUNK_SIZE // 20)  # 5% of chunk_size, minimum 200

EXTRACTION_SYSTEM = """你是应急管理领域知识图谱构建专家。你的任务是从文本中精确抽取三元组（头实体, 关系, 尾实体）。

## 抽取规则
1. 实体必须是具体、独立的概念（事故类型、措施、设备、部门、法规、风险后果等），避免泛化词如"事故""情况"
2. 关系尽量简短（2-8字），使用原文动词短语表述实体间的直接关联
3. 每个三元组必须包含实质性信息，避免常识性废话
4. 同一实体在不同句中出现不同称呼时，统一使用最常见形式
5. 关系类型覆盖以下维度：
   - 因果链：事故->导致->后果、原因->引发->事故
   - 处置链：事故->需要->措施、措施->使用->设备、措施->执行者->部门
   - 组织链：部门->负责->职责、人员->需具备->资质
   - 场景链：事故->易发生于->场所、设备->存放于->位置
   - 法规链：行为->违反->法规、法规->要求->措施

## 输出要求
严格返回 JSON 数组，每个元素含 head/relation/tail 三个字段。禁止输出任何解释性文字。
**关键：必须穷举文本中所有可抽取的三元组，逐句扫描，不要遗漏。文本较长时请按段落顺序依次抽取，宁可多抽不可漏抽。**

## 示例
输入：化工企业应定期检查防爆电气设备，并配备防毒面具等应急救援器材。一旦发生泄漏事故，立即启动应急预案。
输出：[{"head":"化工企业","relation":"定期检查","tail":"防爆电气设备"},{"head":"化工企业","relation":"配备","tail":"防毒面具"},{"head":"防毒面具","relation":"属于","tail":"应急救援器材"},{"head":"泄漏事故","relation":"需要","tail":"启动应急预案"}]
"""


def extract_triples_from_text(text, source_file, chunk_size=None, overlap=None, stream=False):
    if chunk_size is None:
        chunk_size = LLM_CHUNK_SIZE
    if overlap is None:
        overlap = LLM_CHUNK_OVERLAP
    """从文本中提取三元组（支持长文档分段处理）

    stream=False 用于并发模式，避免多线程流式输出交错混乱；
    单文件顺序运行时设为 True 可实时观察 LLM 输出。
    """

    all_triples = []

    # 如果文本较短，直接处理
    if len(text) <= chunk_size:
        return extract_single_chunk(text, source_file, stream=stream)

    # 长文档分段处理
    chunks = split_text_into_chunks(text, chunk_size, overlap)
    print(f"  .. 文档较长，分成 {len(chunks)} 段处理")

    for i, chunk in enumerate(chunks):
        print(f"    处理第 {i+1}/{len(chunks)} 段...")
        triples = extract_single_chunk(chunk, source_file, stream=stream)
        all_triples.extend(triples)
        if CHUNK_PAUSE_SECONDS > 0 and i < len(chunks) - 1:
            time.sleep(CHUNK_PAUSE_SECONDS)

    # 去重
    all_triples = deduplicate_triples(all_triples)

    return all_triples


def split_text_into_chunks(text, chunk_size=1500, overlap=200):
    """将长文本分成重叠的段落"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # 尝试在句子结尾处切分
        if end < len(text):
            for sep in ['。\n', '。', '\n\n', '\n']:
                pos = text.rfind(sep, start + chunk_size // 2, end)
                if pos != -1:
                    end = pos + len(sep)
                    break

        chunks.append(text[start:end])

        if end >= len(text):
            break

        next_start = end - overlap
        if next_start <= start:
            break
        start = next_start

    return chunks


def _parse_triples_response(content, debug=False):
    """从模型返回内容中提取 JSON 数组"""
    content = content.strip()

    if content.startswith('```'):
        lines = content.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        content = '\n'.join(lines).strip()

    start = content.find('[')
    end = content.rfind(']')
    if start == -1 or end == -1 or end < start:
        if debug:
            print(f"    [DEBUG] Model returned (first 500 chars): {content[:500]}")
        raise ValueError("API 返回中未找到合法的 JSON 数组")

    triples = json.loads(content[start:end + 1])
    if not isinstance(triples, list):
        raise ValueError("API 返回的三元组结果不是列表")

    return triples


def extract_single_chunk(text, source_file, stream=True):
    """处理单个文本段落（stream=True 时实时输出 LLM 返回内容）"""
    if not API_KEY:
        raise ValueError(
            f"未找到 DEEPSEEK_API_KEY，请检查环境变量文件: {ENV_FILE}"
        )

    user_prompt = f"请从以下应急文本中抽取三元组：\n\n{text}"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    if stream:
        # 流式请求，实时打印并累积完整响应
        data = {
            "model": API_MODEL,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "stream": True,
            "max_tokens": 16384,
            "response_format": {"type": "json_object"},
        }
        try:
            response = _post_stream_with_retry(headers, data)
            content = ""
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                chunk = line[6:]
                if chunk == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        print(text, end="", flush=True)
                        content += text
                except Exception:
                    continue
            print()  # 换行
            triples = _parse_triples_response(content, debug=True)
        except Exception as e:
            print(f"\n.. 提取失败: {e}")
            return []
    else:
        data = {
            "model": API_MODEL,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "stream": False,
            "max_tokens": 16384,
            "response_format": {"type": "json_object"},
        }
        try:
            response = _post_with_retry(headers, data)
            result = response.json()
            choice = result['choices'][0]
            content = choice.get('message', {}).get('content', '')
            finish = choice.get('finish_reason', 'unknown')
            if not content:
                # Dump full response for diagnosis
                import pprint
                print(f"    [DEBUG] Empty content! finish_reason={finish}")
                print(f"    [DEBUG] Full response: {pprint.pformat(result, indent=2, width=200)}")
            triples = _parse_triples_response(content, debug=True)
        except Exception as e:
            print(f".. 提取失败: {e}")
            return []

    for triple in triples:
        triple['source'] = source_file

    return triples


def _post_stream_with_retry(headers, data):
    """带退避重试的流式聊天补全请求"""
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=data, timeout=120, stream=True)

            if response.status_code == 429 or 500 <= response.status_code < 600:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else RETRY_BASE_DELAY * (2 ** (attempt - 1))
                if attempt < MAX_RETRIES:
                    print(f"\n.. 接口限流({response.status_code})，{delay:.1f}s 后重试 ({attempt}/{MAX_RETRIES})")
                    time.sleep(delay)
                    continue

            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt >= MAX_RETRIES:
                break
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"\n.. 请求失败，{delay:.1f}s 后重试 ({attempt}/{MAX_RETRIES}): {e}")
            time.sleep(delay)

    raise last_error or RuntimeError("DeepSeek API 流式请求失败")


def _post_with_retry(headers, data):
    """带退避重试的聊天补全请求"""
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=data, timeout=60)

            if response.status_code == 429 or 500 <= response.status_code < 600:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                else:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))

                if attempt < MAX_RETRIES:
                    print(
                        f".. 接口暂时不可用({response.status_code})，"
                        f"{delay:.1f}s 后重试 ({attempt}/{MAX_RETRIES})"
                    )
                    time.sleep(delay)
                    continue

            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt >= MAX_RETRIES:
                break

            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f".. 请求失败，{delay:.1f}s 后重试 ({attempt}/{MAX_RETRIES}): {e}")
            time.sleep(delay)

    raise last_error or RuntimeError("DeepSeek API 请求失败")


def deduplicate_triples(triples):
    """三元组去重"""
    seen = set()
    unique_triples = []

    for triple in triples:
        key = (triple.get('head', ''), triple.get('relation', ''), triple.get('tail', ''))
        if key not in seen:
            seen.add(key)
            unique_triples.append(triple)

    return unique_triples


def get_output_path(txt_path):
    """根据 txt 路径生成对应的 json 输出路径"""
    relative_path = txt_path.relative_to(TEXT_DIR)
    return ANNOTATIONS_DIR / relative_path.with_suffix(".json")


_print_lock = threading.Lock()


def _process_one_file(txt_path):
    """单个文件的提取 + 保存（线程安全）"""
    relative_name = str(txt_path.relative_to(TEXT_DIR))
    output_path = get_output_path(txt_path)

    if output_path.exists():
        with _print_lock:
            print(f"  [skip] {relative_name} (already exists)")
        return relative_name, 0, True

    with _print_lock:
        print(f"  [start] {relative_name}")

    text = txt_path.read_text(encoding="utf-8")
    triples = extract_triples_from_text(text, relative_name, stream=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(triples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    count = len(triples)
    with _print_lock:
        if count:
            print(f"  [done] {relative_name} -> {count} triples")
            for t in triples:
                print(f"    ({t['head']}) -[{t['relation']}]-> ({t['tail']})")
        else:
            print(f"  [done] {relative_name} -> (empty)")

    return relative_name, count, False


def process_all_texts():
    """并发处理所有 txt 文件，worker 数由 LLM_CONCURRENCY 控制"""
    txt_files = sorted(TEXT_DIR.glob("**/*.txt"))
    if not txt_files:
        print(f"  no txt files found: {TEXT_DIR}")
        return

    workers = max(1, LLM_CONCURRENCY)
    print(f"Found {len(txt_files)} txt files, {workers} workers")
    print("-" * 60)

    total_triples = 0
    processed = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one_file, p): p for p in txt_files}
        for future in as_completed(futures):
            name, count, was_skipped = future.result()
            if was_skipped:
                skipped += 1
            else:
                processed += 1
                total_triples += count

    print("-" * 60)
    print(
        f"Done. processed {processed}, skipped {skipped}, "
        f"total triples: {total_triples}"
    )


if __name__ == "__main__":
    process_all_texts()

import os
import time
from pathlib import Path
from dotenv import load_dotenv
import requests
import json

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / "config" / ".env"
TEXT_DIR = BASE_DIR / "data" / "processed"
ANNOTATIONS_DIR = BASE_DIR / "data" / "annotations"

load_dotenv(ENV_FILE)

API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("ZHIPU_API_KEY")
API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
API_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.getenv("LLM_RETRY_BASE_DELAY", "2"))
CHUNK_PAUSE_SECONDS = float(os.getenv("LLM_CHUNK_PAUSE_SECONDS", "0.5"))


def extract_triples_from_text(text, source_file, chunk_size=2000, overlap=300):
    """从文本中提取三元组（支持长文档分段处理）"""
    
    all_triples = []
    
    # 如果文本较短，直接处理
    if len(text) <= chunk_size:
        return extract_single_chunk(text, source_file)

    # 长文档分段处理
    chunks = split_text_into_chunks(text, chunk_size, overlap)
    print(f"  📑 文档较长，分成 {len(chunks)} 段处理")
    
    for i, chunk in enumerate(chunks):
        print(f"    处理第 {i+1}/{len(chunks)} 段...")
        triples = extract_single_chunk(chunk, source_file)
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


def _parse_triples_response(content):
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
        raise ValueError("API 返回中未找到合法的 JSON 数组")

    triples = json.loads(content[start:end + 1])
    if not isinstance(triples, list):
        raise ValueError("API 返回的三元组结果不是列表")

    return triples


def extract_single_chunk(text, source_file):
    """处理单个文本段落"""
    if not API_KEY:
        raise ValueError(
            f"未找到 DEEPSEEK_API_KEY，请检查环境变量文件: {ENV_FILE}"
        )
    
    prompt = f"""你是一个专业的知识抽取专家。请从以下应急管理文本中提取知识三元组。

三元组格式：（头实体，关系，尾实体）

重点提取以下类型的关系：

【事故相关】
- 事故/事件 需要/应采取 应急措施
- 事故/事件 可能导致/引发 危害/后果
- 事故/事件 由...引起 原因

【资源相关】
- 措施 使用/需要配备 工具/设备
- 措施 由...执行 人员/部门
- 措施 应在...内完成 时限

【组织相关】
- 部门 负责 职责
- 人员 应具备 资质

【地点相关】
- 事故 发生于/易发生在 地点/场景
- 设备 存放于 位置

文本内容：
{text}

请以JSON格式返回，每个三元组包含：head（头实体）、relation（关系）、tail（尾实体）。
只返回JSON数组，不要其他文字。"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": API_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "stream": False
    }
    
    try:
        response = _post_with_retry(headers, data)
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        triples = _parse_triples_response(content)
        
        for triple in triples:
            triple['source'] = source_file
        
        return triples
        
    except Exception as e:
        print(f"❌ 提取失败: {e}")
        return []


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
                        f"⚠️ 接口暂时不可用({response.status_code})，"
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
            print(f"⚠️ 请求失败，{delay:.1f}s 后重试 ({attempt}/{MAX_RETRIES}): {e}")
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


def process_all_texts():
    """处理所有txt文件（支持子文件夹）"""
    total_triples = 0
    processed_files = 0
    skipped_files = 0
    
    # 递归搜索所有txt文件
    txt_files = sorted(TEXT_DIR.glob('**/*.txt'))
    
    if not txt_files:
        print(f"⚠️ 没有找到txt文件！目录: {TEXT_DIR}")
        return
    
    print(f"找到 {len(txt_files)} 个文本文件")
    
    for txt_path in txt_files:
        # 获取相对路径作为显示名
        relative_name = str(txt_path.relative_to(TEXT_DIR))
        output_path = get_output_path(txt_path)

        if output_path.exists():
            skipped_files += 1
            print(f"\n⏭️ 跳过: {relative_name}")
            print(f"已存在结果文件: {output_path}")
            print("-" * 60)
            continue

        print(f"\n📄 处理: {relative_name}")
        
        # 读取文本
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 提取三元组
        triples = extract_triples_from_text(text, relative_name)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(triples, f, ensure_ascii=False, indent=2)

        if triples:
            print(f"✅ 提取到 {len(triples)} 个三元组")
            total_triples += len(triples)
        else:
            print(f"⚠️ 未提取到三元组")

        processed_files += 1
        print(f"已保存到: {output_path}")
        
        print("-" * 60)

    print(
        f"\n🎉 完成！新处理 {processed_files} 个文本文件，"
        f"跳过 {skipped_files} 个，提取 {total_triples} 个三元组"
    )


if __name__ == "__main__":
    process_all_texts()

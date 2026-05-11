"""应急智能助手 - Streamlit 应用"""
from datetime import datetime
from pathlib import Path
import json
import os
import sys

import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner import get_script_run_ctx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def ensure_streamlit_context():
    """Avoid noisy warnings when the file is launched as a plain Python script."""
    if get_script_run_ctx(suppress_warning=True) is not None:
        return

    print("This app must be started with Streamlit.")
    print(f"Run: streamlit run {Path(__file__).resolve()}")
    raise SystemExit(0)


ensure_streamlit_context()

from model.qa_system import EmergencyQASystem
from kg.visualizer import KGVisualizer

HISTORY_DIR = PROJECT_ROOT / "data" / "history"
HISTORY_FILE = HISTORY_DIR / "chat_history.json"
LEGACY_HISTORY_FILE = PROJECT_ROOT / "data" / "annotations" / "chat_history.json"
ENV_FILE = PROJECT_ROOT / "config" / ".env"

load_dotenv(ENV_FILE)

st.set_page_config(
    page_title="应急智能助手",
    page_icon="🧯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .hero {
            padding: 1.5rem 1.75rem;
            border-radius: 1.25rem;
            background: linear-gradient(135deg, #14213d 0%, #1f3c88 55%, #0f172a 100%);
            color: white;
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.2rem;
        }
        .hero p {
            margin: 0.35rem 0 0;
            opacity: 0.85;
        }
        .graph-shell {
            margin-top: 1rem;
            border-radius: 1rem;
            overflow: hidden;
            border: 1px solid rgba(15, 23, 42, 0.12);
        }
        .graph-legend {
            margin-top: 1rem;
            padding: 1rem 1.2rem;
            border-radius: 1rem;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 64, 175, 0.88));
            color: #f8fafc;
            border: 1px solid rgba(148, 163, 184, 0.18);
        }
        .graph-legend h3 {
            margin: 0 0 0.75rem;
            font-size: 1rem;
        }
        .graph-legend-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.65rem 1rem;
        }
        .graph-legend-item {
            display: flex;
            align-items: flex-start;
            gap: 0.65rem;
            line-height: 1.45;
            font-size: 0.94rem;
        }
        .graph-legend-dot {
            width: 0.9rem;
            height: 0.9rem;
            border-radius: 999px;
            margin-top: 0.28rem;
            flex: 0 0 auto;
            box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.08);
        }
        .graph-legend-note {
            margin-top: 0.85rem;
            color: rgba(248, 250, 252, 0.82);
            font-size: 0.9rem;
        }
        .stButton > button {
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_history():
    source_file = HISTORY_FILE if HISTORY_FILE.exists() else LEGACY_HISTORY_FILE
    if source_file.exists():
        try:
            history = json.loads(source_file.read_text(encoding="utf-8"))
            if source_file == LEGACY_HISTORY_FILE:
                save_history(history)
            return history
        except Exception:
            return []
    return []


def save_history(history):
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if "history" not in st.session_state:
    st.session_state.history = load_history()
if "graph_html" not in st.session_state:
    st.session_state.graph_html = None
if "graph_stats" not in st.session_state:
    st.session_state.graph_stats = None
if "graph_limit" not in st.session_state:
    st.session_state.graph_limit = 100


def get_qa_system():
    """延迟初始化，避免打开网页就加载大模型。"""
    if "qa_system" not in st.session_state:
        with st.spinner("正在初始化问答系统..."):
            st.session_state.qa_system = EmergencyQASystem()
    return st.session_state.qa_system


def build_graph(limit, relation_filter=None, source_filter=None):
    """生成图谱 HTML，并返回统计信息。"""
    visualizer = KGVisualizer()
    try:
        stats = visualizer.get_stats()
        html_path = visualizer.visualize_all(
            "temp_graph.html", limit=limit,
            relation_filter=relation_filter,
            source_filter=source_filter,
        )
        html_content = Path(html_path).read_text(encoding="utf-8")
        return html_content, stats
    finally:
        visualizer.close()


@st.cache_resource
def get_source_driver():
    from utils.neo4j_driver import create_driver
    return create_driver()


def lookup_triple_source(head, relation, tail):
    driver = get_source_driver()
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    with driver.session(database=database) as session:
        rows = session.run(
            """
            MATCH (h:Entity {name: $head})-[r]->(t:Entity {name: $tail})
            RETURN coalesce(r.original_relation, type(r)) AS relation,
                   r.source AS source
            """,
            head=head,
            tail=tail,
        )

        for row in rows:
            if row["relation"] == relation:
                return row["source"]

    return None


def enrich_history_sources(history):
    changed = False
    for item in history:
        existing_sources = [s for s in item.get("sources", []) if s]
        if existing_sources:
            continue

        resolved_sources = []
        for triple in item.get("knowledge", []):
            source = triple.get("source")
            if not source:
                source = lookup_triple_source(
                    triple.get("head", ""),
                    triple.get("relation", ""),
                    triple.get("tail", ""),
                )
            if source:
                triple["source"] = source
                resolved_sources.append(source)

        if resolved_sources:
            item["sources"] = list(dict.fromkeys(resolved_sources))
            changed = True

    if changed:
        save_history(history)

    return history


@st.cache_data(ttl=600)
def get_cached_stats():
    """获取图谱统计信息（缓存10分钟）。"""
    visualizer = KGVisualizer()
    try:
        return visualizer.get_stats()
    finally:
        visualizer.close()


@st.cache_data(ttl=600)
def get_cached_filter_options():
    """获取图谱筛选选项（缓存10分钟）。"""
    visualizer = KGVisualizer()
    try:
        return visualizer.get_filter_options()
    finally:
        visualizer.close()


def render_graph_legend():
    st.markdown(
        """
        <div class="graph-legend">
            <h3>图谱颜色说明</h3>
            <div class="graph-legend-grid">
                <div class="graph-legend-item"><span class="graph-legend-dot" style="background:#FF4B4B;"></span><span>红色：事故、火灾、爆炸、中毒、触电等风险事件</span></div>
                <div class="graph-legend-item"><span class="graph-legend-dot" style="background:#4BFF4B;"></span><span>绿色：措施、管理、排查、救援等处置内容</span></div>
                <div class="graph-legend-item"><span class="graph-legend-dot" style="background:#4B4BFF;"></span><span>蓝色：企业、部门、人员、消防等主体</span></div>
                <div class="graph-legend-item"><span class="graph-legend-dot" style="background:#FF4BFF;"></span><span>紫红：设备、设施、工具、装置</span></div>
                <div class="graph-legend-item"><span class="graph-legend-dot" style="background:#FFD700;"></span><span>金色：其他通用概念</span></div>
            </div>
            <div class="graph-legend-note">箭头表示关系方向，连线文字表示关系名称；鼠标悬停节点或连线可查看详情。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero">
        <h1>应急智能助手</h1>
        <p>知识图谱检索、问答与可视化一体化展示</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("系统概览")
    st.caption("应急知识图谱问答与可视化")

    if "sidebar_stats" not in st.session_state:
        st.session_state.sidebar_stats = None

    if st.button("🔄 刷新统计", use_container_width=True):
        try:
            st.session_state.sidebar_stats = get_cached_stats()
        except Exception:
            st.session_state.sidebar_stats = None

    if st.session_state.sidebar_stats:
        stats = st.session_state.sidebar_stats
        st.metric("节点", stats.get("node_count", 0))
        st.metric("关系", stats.get("rel_count", 0))
        st.metric("文档", stats.get("doc_count", 0))
    else:
        st.caption("点击上方按钮加载统计信息")

    st.markdown("---")
    st.header("功能模块")
    mode = st.radio(
        "选择功能",
        ["智能问答", "知识图谱", "历史记录"],
        index=1,
    )

    st.markdown("---")
    st.caption("提示：知识图谱页会按需生成图谱，避免页面打开时加载大模型。")
    st.caption("问答模块首次使用时才会初始化模型。")


if mode == "智能问答":
    st.subheader("智能问答系统")
    st.caption("输入问题后，会结合知识图谱和 DeepSeek 生成回答。支持多轮追问。")

    # 多轮对话控制
    ctrl_col1, ctrl_col2 = st.columns([1, 5])
    with ctrl_col1:
        if st.button("🔄 新对话", use_container_width=True):
            qa = get_qa_system()
            qa.reset_conversation()
            st.session_state.question_input = ""
            st.rerun()
    with ctrl_col2:
        if qa := st.session_state.get("qa_system"):
            hist_len = len(qa.conversation_history) // 2
            st.caption(f"当前对话轮次: {hist_len}" if hist_len else "新对话已就绪")

    # 预设问题
    preset_questions = [
        "企业在安全生产中有哪些职责？",
        "如何预防触电事故？",
        "违章操作会导致什么后果？",
        "企业需要配备哪些安全设施？",
        "如何预防 KTV 火灾事故？",
        "尾矿库安全需要注意什么？",
    ]

    cols = st.columns(3)
    for idx, q in enumerate(preset_questions):
        if cols[idx % 3].button(q, key=f"preset_{idx}"):
            st.session_state.question_input = q
            st.rerun()

    question = st.text_area(
        "请输入您的应急问题：",
        placeholder="例如：企业需要配备哪些安全设施？",
        height=120,
        key="question_input",
    )

    submit_col, clear_col = st.columns([1, 5])
    with submit_col:
        submit = st.button("提交查询", type="primary")
    with clear_col:
        if st.button("清空输入"):
            st.session_state.question_input = ""
            st.rerun()

    if submit and question:
        try:
            with st.spinner("正在检索知识..."):
                qa_system = get_qa_system()
                result = qa_system.answer_question(question, stream=True)

            # 流式显示回答
            st.markdown("### 回答")
            answer_placeholder = st.empty()
            full_answer = ""
            for chunk in result["answer"]:
                full_answer += chunk
                answer_placeholder.markdown(full_answer + "▌")
            answer_placeholder.markdown(full_answer)

            # 保存到多轮对话历史
            qa_system.add_to_history(question, full_answer)

            # 保存到持久化历史
            sources = sorted(
                {
                    item.get("source", "未知来源")
                    for item in result.get("knowledge", [])
                    if item.get("source")
                }
            )
            history_item = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question": question,
                "answer": full_answer,
                "knowledge": result["knowledge"],
                "sources": sources,
            }
            st.session_state.history.append(history_item)
            save_history(st.session_state.history)

            if result["knowledge"]:
                with st.expander("查看知识来源"):
                    for i, item in enumerate(result["knowledge"][:12], 1):
                        source = item.get("source", "未知来源")
                        if item.get("is_multihop"):
                            display = item.get("path_text",
                                f"({item['head']})-[{item['relation']}]->({item['tail']})")
                            st.write(f"{i}. 🧬 **多跳路径**: {display}")
                        else:
                            st.write(f"{i}. ({item['head']}) -[{item['relation']}]-> ({item['tail']})")
                        st.caption(f"来源文件: {source}")
        except Exception as e:
            st.error(f"发生错误：{e}")


elif mode == "知识图谱":
    st.subheader("知识图谱可视化")
    st.caption("图谱会按当前数据库内容动态生成，支持拖拽、缩放和悬停查看。")

    # 筛选选项
    filter_opts = None
    try:
        filter_opts = get_cached_filter_options()
    except Exception:
        pass

    try:
        max_edges = max(30, get_cached_stats().get("rel_count", 30))
    except Exception:
        max_edges = 300

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        relation_options = ["（全部）"]
        if filter_opts:
            relation_options += filter_opts.get("relations", [])
        relation_filter = st.selectbox(
            "按关系类型筛选（可选）",
            options=relation_options,
            index=0,
            key="relation_filter",
        )
    with filter_col2:
        source_options = ["（全部）"]
        if filter_opts:
            source_options += filter_opts.get("sources", [])
        source_filter = st.selectbox(
            "按来源文档筛选（可选）",
            options=source_options,
            index=0,
            key="source_filter",
        )

    control_col, info_col = st.columns([1, 2])

    with control_col:
        default_limit = min(st.session_state.graph_limit, max_edges)
        limit = st.slider("显示边数量", 30, max_edges, default_limit, 10)
        generate = st.button("生成 / 刷新图谱", type="primary")
        st.caption("建议先从 80 到 120 条边开始。")

    with info_col:
        if st.session_state.graph_stats:
            stats = st.session_state.graph_stats
            c1, c2, c3 = st.columns(3)
            c1.metric("节点", stats.get("node_count", 0))
            c2.metric("关系", stats.get("rel_count", 0))
            c3.metric("文档", stats.get("doc_count", 0))

            if stats.get("relations"):
                with st.expander("Top 关系类型"):
                    for item in stats["relations"]:
                        st.write(f"- {item['type']}: {item['count']}")
        else:
            st.info("尚未生成图谱。点击左侧按钮后，这里会显示统计信息。")

    if generate:
        with st.spinner("正在生成图谱..."):
            try:
                rel = None if relation_filter == "（全部）" else relation_filter
                src = None if source_filter == "（全部）" else source_filter
                html_content, stats = build_graph(limit,
                                                   relation_filter=rel,
                                                   source_filter=src)
                st.session_state.graph_html = html_content
                st.session_state.graph_stats = stats
                st.session_state.graph_limit = limit
                st.success("图谱生成完成")
            except Exception as e:
                st.error(f"图谱生成失败：{e}")

    st.markdown('<div class="graph-shell">', unsafe_allow_html=True)
    if st.session_state.graph_html:
        st.components.v1.html(st.session_state.graph_html, height=900, scrolling=True)
    else:
        st.info("点击“生成 / 刷新图谱”后，图谱会在这里完整展示。")
    st.markdown("</div>", unsafe_allow_html=True)
    render_graph_legend()


else:
    st.subheader("历史记录")
    st.session_state.history = enrich_history_sources(st.session_state.history)
    if not st.session_state.history:
        st.info("暂无历史记录")
    else:
        for idx, item in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"{idx}. {item.get('timestamp', '')} {item['question'][:60]}"):
                st.markdown("**问题**")
                st.write(item["question"])
                st.markdown("**回答**")
                answer = item["answer"]
                st.write(answer[:600] + "..." if len(answer) > 600 else answer)

                if item.get("sources"):
                    st.markdown("**来源文件**")
                    for source in item["sources"]:
                        st.write(f"- {source}")

        if st.button("清空历史"):
            st.session_state.history = []
            if HISTORY_FILE.exists():
                HISTORY_FILE.unlink()
            if LEGACY_HISTORY_FILE.exists():
                LEGACY_HISTORY_FILE.unlink()
            st.rerun()

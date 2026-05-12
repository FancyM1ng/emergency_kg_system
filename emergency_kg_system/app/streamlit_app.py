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

# ── CSS ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Root tokens ── */
    :root {
        --primary: #1e40af;
        --primary-light: #3b82f6;
        --danger: #dc2626;
        --success: #16a34a;
        --warning: #f59e0b;
        --slate-50: #f8fafc;
        --slate-100: #f1f5f9;
        --slate-200: #e2e8f0;
        --slate-600: #475569;
        --slate-700: #334155;
        --slate-800: #1e293b;
        --slate-900: #0f172a;
        --radius: 0.75rem;
        --shadow: 0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.06);
        --shadow-md: 0 4px 6px rgba(0,0,0,.07), 0 2px 4px rgba(0,0,0,.06);
    }

    /* ── Hero ── */
    .hero {
        padding: 1.8rem 2.2rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 45%, #1e40af 100%);
        color: #fff;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-md);
    }
    .hero h1 { margin: 0; font-size: 1.9rem; letter-spacing: -.02em; }
    .hero p  { margin: .35rem 0 0; opacity: .82; font-size: .95rem; }

    /* ── Stat cards (sidebar) ── */
    .stat-row { display: flex; gap: .5rem; margin: .6rem 0; }
    .stat-card {
        flex: 1; text-align: center; padding: .45rem .25rem;
        background: rgba(255,255,255,.06); border-radius: .55rem;
    }
    .stat-card .val { font-size: 1.25rem; font-weight: 700; color: #e2e8f0; }
    .stat-card .lbl { font-size: .7rem; color: #94a3b8; margin-top: .1rem; }

    /* ── Preset question chips ── */
    .preset-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: .55rem;
        margin: .6rem 0 1rem;
    }
    @media (max-width: 768px) { .preset-grid { grid-template-columns: 1fr; } }

    /* ── Answer panel ── */
    .answer-panel {
        background: var(--slate-50);
        border: 1px solid var(--slate-200);
        border-radius: var(--radius);
        padding: 1.25rem 1.5rem;
        margin-top: .8rem;
        line-height: 1.75;
        font-size: .95rem;
    }

    /* ── Knowledge source item ── */
    .src-item {
        padding: .45rem .75rem;
        margin: .25rem 0;
        border-radius: .45rem;
        background: var(--slate-100);
        font-size: .88rem;
        font-family: "SF Mono", "Fira Code", "Consolas", monospace;
    }
    .src-item.multihop { border-left: 3px solid #8b5cf6; }

    /* ── Graph shell ── */
    .graph-shell {
        margin-top: 1rem;
        border-radius: 1rem;
        overflow: hidden;
        border: 1px solid var(--slate-200);
        box-shadow: var(--shadow);
    }

    /* ── Legend ── */
    .graph-legend {
        margin-top: 1rem;
        padding: 1rem 1.2rem;
        border-radius: var(--radius);
        background: linear-gradient(135deg, #0f172a, #1e3a5f);
        color: #f1f5f9;
        font-size: .9rem;
    }
    .graph-legend h3 { margin: 0 0 .7rem; font-size: .95rem; }
    .legend-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: .5rem 1rem;
    }
    .legend-item { display: flex; align-items: center; gap: .55rem; }
    .legend-dot {
        width: .75rem; height: .75rem; border-radius: 50%;
        flex-shrink: 0; box-shadow: 0 0 0 2px rgba(255,255,255,.12);
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: .5rem !important;
        font-weight: 500 !important;
        transition: all .15s !important;
    }

    /* ── Expander tweaks ── */
    .streamlit-expanderHeader { font-weight: 600; }

    /* ── Metric cards on graph page ── */
    .metric-row { display: flex; gap: 1rem; margin-bottom: .5rem; }
    .metric-card {
        flex: 1; padding: .9rem 1.1rem;
        background: var(--slate-50); border-radius: var(--radius);
        border: 1px solid var(--slate-200); text-align: center;
    }
    .metric-card .num { font-size: 1.65rem; font-weight: 700; color: var(--slate-800); }
    .metric-card .tag { font-size: .78rem; color: var(--slate-600); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "graph_html" not in st.session_state:
    st.session_state.graph_html = None
if "graph_stats" not in st.session_state:
    st.session_state.graph_stats = None
if "graph_limit" not in st.session_state:
    st.session_state.graph_limit = 100
if "sidebar_loaded" not in st.session_state:
    st.session_state.sidebar_loaded = False


# ── Persistence helpers ──────────────────────────────
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


# Load persisted history once
if not st.session_state.history:
    st.session_state.history = load_history()


# ── Lazy init ────────────────────────────────────────
def get_qa_system():
    if "qa_system" not in st.session_state:
        with st.spinner("正在初始化问答系统（加载语义模型...）"):
            st.session_state.qa_system = EmergencyQASystem()
    return st.session_state.qa_system


@st.cache_resource
def get_source_driver():
    from utils.neo4j_driver import create_driver
    return create_driver()


@st.cache_data(ttl=600)
def get_cached_stats():
    visualizer = KGVisualizer()
    try:
        return visualizer.get_stats()
    finally:
        visualizer.close()


@st.cache_data(ttl=600)
def get_cached_filter_options():
    visualizer = KGVisualizer()
    try:
        return visualizer.get_filter_options()
    finally:
        visualizer.close()


def build_graph(limit, relation_filter=None, source_filter=None):
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
            head=head, tail=tail,
        )
        for row in rows:
            if row["relation"] == relation:
                return row["source"]
    return None


def enrich_history_sources(history):
    changed = False
    for item in history:
        existing = [s for s in item.get("sources", []) if s]
        if existing:
            continue
        resolved = []
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
                resolved.append(source)
        if resolved:
            item["sources"] = list(dict.fromkeys(resolved))
            changed = True
    if changed:
        save_history(history)
    return history


# ── Sidebar ──────────────────────────────────────────
with st.sidebar:
    # Auto-load stats on first render
    if not st.session_state.sidebar_loaded:
        try:
            st.session_state.sidebar_stats = get_cached_stats()
        except Exception:
            st.session_state.sidebar_stats = None
        st.session_state.sidebar_loaded = True

    st.markdown("### 系统概览")

    stats = st.session_state.get("sidebar_stats")
    if stats:
        st.markdown(
            f"""<div class="stat-row">
            <div class="stat-card"><div class="val">{stats.get('node_count',0):,}</div><div class="lbl">实体节点</div></div>
            <div class="stat-card"><div class="val">{stats.get('rel_count',0):,}</div><div class="lbl">关系边</div></div>
            <div class="stat-card"><div class="val">{stats.get('doc_count',0)}</div><div class="lbl">来源文档</div></div>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("🔄 刷新统计", use_container_width=True):
            try:
                st.session_state.sidebar_stats = get_cached_stats()
            except Exception:
                st.session_state.sidebar_stats = None
            st.rerun()
    else:
        st.caption("未能加载统计，请检查 Neo4j 连接")
        if st.button("🔄 重试", use_container_width=True):
            try:
                st.session_state.sidebar_stats = get_cached_stats()
            except Exception:
                pass
            st.rerun()

    st.divider()
    st.markdown("### 功能模块")
    mode = st.radio(
        "选择功能",
        ["智能问答", "知识图谱", "历史记录"],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    with st.expander("💡 使用提示"):
        st.caption("• 问答模块首次使用需加载语义模型（约 5-10 秒）")
        st.caption("• 图谱页按需生成，可拖拽缩放")
        st.caption("• 支持多轮追问，点击「新对话」重置")


# ── Hero ─────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>应急智能助手</h1>
        <p>知识图谱 + 语义检索 + 大模型推理 &nbsp;|&nbsp; 应急管理智能问答与可视化平台</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════
# MODE 1: 智能问答
# ══════════════════════════════════════════════════════
if mode == "智能问答":
    tab1, tab2 = st.tabs(["问答", "知识来源"])

    with tab1:
        # ── 对话控制栏 ──
        ctrl_left, ctrl_right = st.columns([1.2, 5])
        with ctrl_left:
            if st.button("🔄 新对话", use_container_width=True):
                qa = get_qa_system()
                qa.reset_conversation()
                st.session_state.question_input = ""
                st.rerun()
        with ctrl_right:
            qa_ref = st.session_state.get("qa_system")
            if qa_ref:
                rounds = len(qa_ref.conversation_history) // 2
                if rounds:
                    st.info(f"当前对话轮次：**{rounds}**（上下文已保留最近 10 轮）")
                else:
                    st.caption("新对话已就绪，输入问题开始查询。")

        # ── 预设问题 ──
        st.caption("快速提问：")
        preset_questions = [
            ("🏭", "企业在安全生产中有哪些职责？"),
            ("⚡", "如何预防触电事故？"),
            ("⚠️", "违章操作会导致什么后果？"),
            ("🧯", "企业需要配备哪些安全设施？"),
            ("🔥", "如何预防KTV火灾事故？"),
            ("⛏️", "尾矿库安全需要注意什么？"),
        ]
        cols = st.columns(3)
        for idx, (icon, q) in enumerate(preset_questions):
            with cols[idx % 3]:
                if st.button(f"{icon} {q}", key=f"preset_{idx}", use_container_width=True):
                    st.session_state.question_input = q
                    st.rerun()

        # ── 输入区 ──
        question = st.text_area(
            "请输入您的应急问题",
            placeholder="例如：化工厂发生氯气泄漏，现场人员应采取哪些紧急措施？",
            height=100,
            key="question_input",
            label_visibility="collapsed",
        )

        col_a, col_b, col_c = st.columns([1.2, 1, 5])
        with col_a:
            submit = st.button("🔍 提交查询", type="primary", use_container_width=True)
        with col_b:
            if st.button("✕ 清空", use_container_width=True):
                st.session_state.question_input = ""
                st.rerun()

        st.divider()

        # ── 回答区 ──
        if submit and question:
            try:
                with st.spinner("正在检索知识图谱..."):
                    qa_system = get_qa_system()
                    result = qa_system.answer_question(question, stream=True)

                st.markdown("### 回答")
                placeholder = st.empty()
                full_answer = ""
                for chunk in result["answer"]:
                    full_answer += chunk
                    placeholder.markdown(
                        f'<div class="answer-panel">{full_answer}▌</div>',
                        unsafe_allow_html=True,
                    )
                placeholder.markdown(
                    f'<div class="answer-panel">{full_answer}</div>',
                    unsafe_allow_html=True,
                )

                # 保存上下文 / 持久化
                qa_system.add_to_history(question, full_answer)

                sources = sorted({
                    item.get("source", "未知来源")
                    for item in result.get("knowledge", [])
                    if item.get("source")
                })
                history_item = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "question": question,
                    "answer": full_answer,
                    "knowledge": result["knowledge"],
                    "sources": sources,
                }
                st.session_state.history.append(history_item)
                save_history(st.session_state.history)

                # 知识来源摘要
                single_count = sum(1 for k in result["knowledge"] if not k.get("is_multihop"))
                multi_count = sum(1 for k in result["knowledge"] if k.get("is_multihop"))
                st.caption(
                    f"已检索 {len(result['knowledge'])} 条知识 "
                    f"（直接关联 {single_count} 条 + 多跳推理 {multi_count} 条）"
                )

            except Exception as e:
                st.error(f"查询失败：{e}")

    with tab2:
        st.caption("最近一次查询的知识来源明细")
        if st.session_state.history:
            last = st.session_state.history[-1]
            knowledge = last.get("knowledge", [])
            if knowledge:
                for i, item in enumerate(knowledge[:15], 1):
                    is_mh = item.get("is_multihop")
                    css = "src-item multihop" if is_mh else "src-item"
                    icon = "🧬" if is_mh else "📌"
                    if is_mh:
                        display = item.get("path_text",
                            f"{item['head']} -[{item['relation']}]-> {item['tail']}")
                    else:
                        display = f"({item['head']}) -[{item['relation']}]-> ({item['tail']})"
                    st.markdown(
                        f'<div class="{css}">{icon} {i}. {display}'
                        f'&nbsp;&nbsp;<small style="color:#94a3b8">{item.get("source","")}</small></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("该查询未匹配到知识库记录")
        else:
            st.info("暂无查询记录，提交问题后这里会展示检索到的三元组")

# ══════════════════════════════════════════════════════
# MODE 2: 知识图谱
# ══════════════════════════════════════════════════════
elif mode == "知识图谱":
    st.subheader("知识图谱可视化")
    st.caption("动态生成交互式知识图谱，支持拖拽缩放、节点悬停查看详情。")

    # ── 筛选 ──
    filter_opts = None
    try:
        filter_opts = get_cached_filter_options()
    except Exception:
        pass

    try:
        max_edges = max(30, get_cached_stats().get("rel_count", 30))
    except Exception:
        max_edges = 300

    filt1, filt2 = st.columns(2)
    with filt1:
        rel_opts = ["（全部）"] + (filter_opts.get("relations", []) if filter_opts else [])
        relation_filter = st.selectbox("按关系类型筛选", rel_opts, key="rel_filt")
    with filt2:
        src_opts = ["（全部）"] + (filter_opts.get("sources", []) if filter_opts else [])
        source_filter = st.selectbox("按来源文档筛选", src_opts, key="src_filt")

    # ── 控制 ──
    ctrl, info = st.columns([1, 2])
    with ctrl:
        limit = st.slider(
            "显示边数量", 30, max_edges,
            min(st.session_state.graph_limit, max_edges), 10,
        )
        generate = st.button("🔍 生成 / 刷新图谱", type="primary", use_container_width=True)
        st.caption("建议先从 80-120 条边开始，边数越多渲染越慢")

    with info:
        if st.session_state.graph_stats:
            s = st.session_state.graph_stats
            st.markdown(
                f"""<div class="metric-row">
                <div class="metric-card"><div class="num">{s.get('node_count',0):,}</div><div class="tag">节点</div></div>
                <div class="metric-card"><div class="num">{s.get('rel_count',0):,}</div><div class="tag">关系</div></div>
                <div class="metric-card"><div class="num">{s.get('doc_count',0)}</div><div class="tag">文档</div></div>
                </div>""",
                unsafe_allow_html=True,
            )
            if s.get("relations"):
                with st.expander("Top 关系类型"):
                    for item in s["relations"]:
                        st.write(f"- **{item['type']}**: {item['count']} 条")
        else:
            st.info("点击左侧按钮生成图谱，统计信息将在此显示。")

    if generate:
        with st.spinner("正在从 Neo4j 拉取数据并渲染图谱..."):
            try:
                rel = None if relation_filter == "（全部）" else relation_filter
                src = None if source_filter == "（全部）" else source_filter
                html_content, stats = build_graph(limit, relation_filter=rel, source_filter=src)
                st.session_state.graph_html = html_content
                st.session_state.graph_stats = stats
                st.session_state.graph_limit = limit
            except Exception as e:
                st.error(f"图谱生成失败：{e}")

    st.markdown('<div class="graph-shell">', unsafe_allow_html=True)
    if st.session_state.graph_html:
        st.components.v1.html(st.session_state.graph_html, height=880, scrolling=True)
    else:
        st.info("点击「生成 / 刷新图谱」后，交互式知识图谱将在此渲染。")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 图例 ──
    st.markdown(
        """
        <div class="graph-legend">
            <h3>节点颜色说明</h3>
            <div class="legend-grid">
                <div class="legend-item"><span class="legend-dot" style="background:#FF4B4B"></span> 红色 — 事故 / 火灾 / 爆炸 / 中毒 / 触电</div>
                <div class="legend-item"><span class="legend-dot" style="background:#4BFF4B"></span> 绿色 — 措施 / 管理 / 排查 / 救援</div>
                <div class="legend-item"><span class="legend-dot" style="background:#4B4BFF"></span> 蓝色 — 企业 / 部门 / 人员 / 消防</div>
                <div class="legend-item"><span class="legend-dot" style="background:#FF4BFF"></span> 紫红 — 设备 / 设施 / 工具 / 装置</div>
                <div class="legend-item"><span class="legend-dot" style="background:#FFD700"></span> 金色 — 其他通用概念</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════
# MODE 3: 历史记录
# ══════════════════════════════════════════════════════
else:
    st.subheader("历史记录")
    st.session_state.history = enrich_history_sources(st.session_state.history)

    if not st.session_state.history:
        st.info("暂无历史记录，在「智能问答」中提交问题后会自动保存。")
    else:
        st.caption(f"共 **{len(st.session_state.history)}** 条记录，最新在前")

        # 搜索过滤
        filter_text = st.text_input(
            "搜索历史问题",
            placeholder="输入关键词过滤...",
            label_visibility="collapsed",
        )

        visible = list(reversed(st.session_state.history))
        if filter_text:
            visible = [
                item for item in visible
                if filter_text.lower() in item.get("question", "").lower()
            ]

        for idx, item in enumerate(visible, 1):
            ts = item.get("timestamp", "")
            q = item.get("question", "")
            answer = item.get("answer", "")
            sources = item.get("sources", [])

            with st.expander(f"{idx}. {ts}  {q[:70]}{'...' if len(q)>70 else ''}"):
                st.markdown("**问题**")
                st.write(q)
                st.markdown("**回答**")
                # Show full answer, not truncated to 600 chars
                st.markdown(
                    f'<div class="answer-panel">{answer}</div>',
                    unsafe_allow_html=True,
                )
                if sources:
                    st.markdown("**来源文件**")
                    for s in sources:
                        st.write(f"- {s}")

        col_del1, col_del2 = st.columns([1, 5])
        with col_del1:
            if st.button("🗑️ 清空全部历史", use_container_width=True):
                st.session_state.history = []
                if HISTORY_FILE.exists():
                    HISTORY_FILE.unlink()
                if LEGACY_HISTORY_FILE.exists():
                    LEGACY_HISTORY_FILE.unlink()
                st.rerun()
        with col_del2:
            st.caption(f"当前显示 {len(visible)} 条记录")

import streamlit as st
import json
import psycopg2
import psycopg2.extras
import pandas as pd
from datetime import datetime
import os
import sys

# Point sys.path directly at app/core so we can import semantic_engine
# WITHOUT going through the 'app' package (which would re-execute this script)
_core_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core"))
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from semantic_engine import generate_mocked_schema
from agent import build_db_uri, stream_agent_response

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SQLmind — Semantic Layer Builder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    h1 { color: #4ade80 !important; }
    h2, h3 { color: #94a3b8 !important; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .rule-card {
        background: #1e293b;
        border-left: 4px solid #4ade80;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# GENERIC AUDIT COLUMNS TO IGNORE IN AUTO-SUGGEST (Fix #5)
# These columns appear in almost every table and are NOT
# true business relationships.
# ============================================================
GENERIC_COLUMNS = {
    "id", "created_at", "updated_at", "deleted_at",
    "created_by", "updated_by", "is_active", "status",
    "modified_at", "modified_by", "timestamp"
}

# ============================================================
# SESSION STATE INIT
# ============================================================
if "db_conn" not in st.session_state:
    st.session_state.db_conn = None
if "tables" not in st.session_state:
    st.session_state.tables = []
if "columns" not in st.session_state:
    st.session_state.columns = {}
if "domain_tables" not in st.session_state:
    st.session_state.domain_tables = []
if "saved_rules" not in st.session_state:
    st.session_state.saved_rules = []
if "suggested_rules" not in st.session_state:
    st.session_state.suggested_rules = []
if "domain_name" not in st.session_state:      # Fix #8: Persist domain name
    st.session_state.domain_name = ""
if "db_uri" not in st.session_state:           # SQLAlchemy URI for agent
    st.session_state.db_uri = None
if "chat_history" not in st.session_state:     # List of {role, content} dicts
    st.session_state.chat_history = []
if "agent_graph" not in st.session_state:      # Cached compiled LangGraph
    st.session_state.agent_graph = None

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_connection(host, port, dbname, user, password):
    return psycopg2.connect(host=host, port=int(port), dbname=dbname, user=user, password=password)

def fetch_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public' ORDER BY tablename;")
    return [row[0] for row in cur.fetchall()]

def fetch_columns(conn, table_name):
    cur = conn.cursor()
    cur.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = %s ORDER BY ordinal_position;",
        (table_name,)
    )
    return cur.fetchall()

def run_test_query(conn, base_table, base_col, target_table, target_col):
    """
    Fix #4: Use psycopg2 cursor directly instead of pd.read_sql()
    to avoid SQLAlchemy engine requirement.
    """
    query = (
        f"SELECT * FROM {base_table} "
        f"INNER JOIN {target_table} "
        f"ON {base_table}.{base_col} = {target_table}.{target_col} "
        f"LIMIT 5"
    )
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(query)
        rows = cur.fetchall()
        df = pd.DataFrame(rows)
        return df, query, None
    except Exception as e:
        conn.rollback()  # Reset connection after failed query
        return None, query, str(e)

def suggest_joins(conn, domain_tables):
    """
    Fix #5: Skip generic audit columns (created_at, id, etc.)
    to avoid polluting the suggestions with meaningless joins.
    """
    suggestions = []
    col_map = {}
    for t in domain_tables:
        cols = fetch_columns(conn, t)
        # Only keep non-generic columns for matching
        col_map[t] = [c[0] for c in cols if c[0].lower() not in GENERIC_COLUMNS]

    for i, table_a in enumerate(domain_tables):
        for table_b in domain_tables[i+1:]:
            shared = set(col_map[table_a]) & set(col_map[table_b])
            for col in shared:
                suggestions.append({
                    "base_table": table_a,
                    "base_col": col,
                    "target_table": table_b,
                    "target_col": col,
                    "description": f"Auto-suggested via shared column: '{col}'"
                })
    return suggestions

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "semantic_config.json")

def load_existing_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {"domain": "", "relations": []}

def save_config(domain_name, rules):
    config = {
        "domain": domain_name,
        "generated_at": datetime.now().isoformat(),
        "relations": rules
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    return config

# ============================================================
# SIDEBAR — DB CONNECTION
# ============================================================
with st.sidebar:
    st.markdown("## 🔌 Database Connection")
    host     = st.text_input("Host",     value="182.18.181.115", key="sb_host")
    port     = st.number_input("Port",   value=5432, step=1)
    dbname   = st.text_input("Database", value="ptax_gadchandur", key="sb_dbname")
    user     = st.text_input("User",     value="postgres", key="sb_user")
    password = st.text_input("Password", value="technowell", type="password", key="sb_password")

    if st.button("🔗 Connect"):
        try:
            conn_obj = get_connection(host, port, dbname, user, password)
            new_uri = build_db_uri(user, password, host, port, dbname)
            st.session_state.db_conn = conn_obj
            st.session_state.tables = fetch_tables(conn_obj)
            # Clear stale state from previous DB
            st.session_state.domain_tables = []
            st.session_state.saved_rules = []
            st.session_state.suggested_rules = []
            st.session_state.columns = {}
            st.session_state.domain_name = ""
            st.session_state.chat_history = []
            # Invalidate cached graph if the database changed
            if st.session_state.db_uri != new_uri:
                st.session_state.agent_graph = None
            st.session_state.db_uri = new_uri
            st.success(f"✅ Connected! Found {len(st.session_state.tables)} tables.")
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")

    st.divider()
    existing = load_existing_config()
    if existing["relations"]:
        st.markdown(f"**📁 Config Loaded:** `{existing['domain']}`")
        st.caption(f"{len(existing['relations'])} rule(s) saved")

# ============================================================
# MAIN CONTENT
# ============================================================
st.markdown("# 🧠 SQLmind — Semantic Layer Builder")
st.caption("Map Primary and Foreign Keys visually so the AI agent understands your database structure.")

if not st.session_state.db_conn:
    st.info("👈 Connect to your PostgreSQL database using the sidebar to begin.")
    st.stop()  # Safe to use at root level — NOT inside tabs

conn = st.session_state.db_conn
all_tables = st.session_state.tables

tab1, tab2, tab3, tab4 = st.tabs([
    "📦 Step 1: Domain Setup",
    "🔗 Step 2: Map Relations",
    "📤 Step 3: Generate & Export",
    "💬 Step 4: Chat with Data",
])

# ============================================================
# TAB 1 — DOMAIN SETUP
# ============================================================
with tab1:
    st.markdown("### 📦 Create a Business Domain")
    col1, col2 = st.columns([1, 2])
    with col1:
        # Fix #8: Bind input to session_state so it survives reruns
        domain_name_input = st.text_input(
            "Domain Name",
            value=st.session_state.domain_name,
            placeholder="e.g. Property Tax Analytics",
            key="domain_name_input"
        )
    with col2:
        # Filter defaults to only include tables that exist in current DB
        valid_defaults = [t for t in st.session_state.domain_tables if t in all_tables]
        selected_tables = st.multiselect(
            "Select Tables for this Domain",
            options=all_tables,
            default=valid_defaults
        )

    if st.button("✅ Confirm Domain"):
        if not domain_name_input or not selected_tables:
            st.warning("Please enter a domain name and select at least 2 tables.")
        else:
            st.session_state.domain_name = domain_name_input   # Fix #8: Persist it
            st.session_state.domain_tables = selected_tables
            for t in selected_tables:
                if t not in st.session_state.columns:
                    st.session_state.columns[t] = fetch_columns(conn, t)
            st.success(f"✅ Domain '{domain_name_input}' set with {len(selected_tables)} tables!")

    if st.session_state.domain_tables:
        st.divider()
        st.markdown("#### 📋 Table Previews")
        for t in st.session_state.domain_tables:
            with st.expander(f"🗂️ `{t}`"):
                cols = st.session_state.columns.get(t, [])
                df_cols = pd.DataFrame(cols, columns=["Column", "Type"])
                st.dataframe(df_cols, hide_index=True)

# ============================================================
# TAB 2 — MAP RELATIONS (STRUCTURAL PK/FK)
# ============================================================
with tab2:
    if not st.session_state.domain_tables:
        st.warning("⚠️ Please complete Step 1 first — set up your domain and select tables.")
    else:
        st.markdown("### 🔗 Define Table Relationships")

        # ----- AUTO-SUGGEST SECTION -----
        col_a, col_b = st.columns([3, 1])
        with col_b:
            if st.button("✨ Auto-Suggest Relations"):
                with st.spinner("Analyzing shared non-generic columns..."):
                    suggestions = suggest_joins(conn, st.session_state.domain_tables)
                if suggestions:
                    # Filter out suggestions that are already saved
                    existing = {
                        (r["base_table"], r["base_col"], r["target_table"], r["target_col"])
                        for r in st.session_state.saved_rules
                    }
                    new_suggestions = [
                        s for s in suggestions
                        if (s["base_table"], s["base_col"], s["target_table"], s["target_col"]) not in existing
                    ]
                    st.session_state.suggested_rules = new_suggestions
                    if new_suggestions:
                        st.success(f"✅ {len(new_suggestions)} suggestion(s) found! Review and approve below.")
                    else:
                        st.info("All matching relations are already saved.")
                else:
                    st.session_state.suggested_rules = []
                    st.info("No meaningful shared columns found. Map manually.")

        # Display pending suggestions for approval
        if st.session_state.suggested_rules:
            st.divider()
            st.markdown(f"#### 💡 Pending Suggestions ({len(st.session_state.suggested_rules)} — review each)")
            st.caption("These are auto-detected. Approve ✅ the correct ones, reject ❌ the wrong ones.")
            for i, sug in enumerate(st.session_state.suggested_rules):
                col_info, col_approve, col_reject = st.columns([8, 1, 1])
                with col_info:
                    st.markdown(
                        f"**{sug['target_table']}**.`{sug['target_col']}` "
                        f"→ **{sug['base_table']}**.`{sug['base_col']}` "
                        f"&nbsp; <small style='color:#94a3b8'>({sug.get('description', '')})</small>",
                        unsafe_allow_html=True
                    )
                with col_approve:
                    if st.button("✅", key=f"approve_{i}", help="Approve this relation"):
                        st.session_state.saved_rules.append(sug)
                        st.session_state.suggested_rules.pop(i)
                        st.rerun()
                with col_reject:
                    if st.button("❌", key=f"reject_{i}", help="Reject this relation"):
                        st.session_state.suggested_rules.pop(i)
                        st.rerun()

            # Bulk actions
            col_all_approve, col_all_reject, _ = st.columns([1, 1, 6])
            with col_all_approve:
                if st.button("✅ Approve All", key="approve_all"):
                    st.session_state.saved_rules.extend(st.session_state.suggested_rules)
                    st.session_state.suggested_rules = []
                    st.rerun()
            with col_all_reject:
                if st.button("❌ Reject All", key="reject_all"):
                    st.session_state.suggested_rules = []
                    st.rerun()

        # ----- MANUAL MAPPING SECTION -----
        st.divider()
        st.markdown("#### ➕ Manually Map Primary Key to Foreign Key")

        container = st.container(border=True)
        with container:
            col_base, col_tgt = st.columns(2)
            with col_base:
                base_table = st.selectbox(
                    "Base Table (holds Primary Key)",
                    options=st.session_state.domain_tables,
                    key="base_t"
                )
                b_cols = [c[0] for c in st.session_state.columns.get(base_table, [])] if base_table else []
                base_col = st.selectbox("Base Column (Primary Key)", options=b_cols, key="base_c")

            with col_tgt:
                target_options = [t for t in st.session_state.domain_tables if t != base_table]
                target_table = st.selectbox(
                    "Target Table (holds Foreign Key)",
                    options=target_options,
                    key="target_t"
                )
                t_cols = [c[0] for c in st.session_state.columns.get(target_table, [])] if target_table else []
                target_col = st.selectbox("Target Column (Foreign Key)", options=t_cols, key="target_c")

            description = st.text_input(
                "Business Rule Description (optional)",
                placeholder="e.g. Links payment records to property master",
                key="relation_description"
            )

            col_submit, col_test = st.columns(2)
            test_clicked = col_test.button("🧪 Test This Relation")
            submitted   = col_submit.button("💾 Save This Relation")

            if test_clicked:
                if not base_col or not target_col or not target_table:
                    st.error("Select both a Primary Key and a Foreign Key column.")
                else:
                    with st.spinner("Running test join query..."):
                        df_result, test_query, error = run_test_query(
                            conn, base_table, base_col, target_table, target_col
                        )
                    if error:
                        st.error(f"❌ Test Failed: {error}")
                        st.code(test_query, language="sql")
                    else:
                        st.success("✅ Mathematical Join Passed! Here are up to 5 sample rows:")
                        st.code(test_query, language="sql")
                        st.dataframe(df_result)

            if submitted:
                if not base_col or not target_col or not target_table:
                    st.error("Select both a Primary Key and a Foreign Key column.")
                else:
                    rule = {
                        "base_table":   base_table,
                        "base_col":     base_col,
                        "target_table": target_table,
                        "target_col":   target_col,
                        "description":  description
                    }
                    st.session_state.saved_rules.append(rule)
                    st.success(f"✅ Relation saved! Total: {len(st.session_state.saved_rules)} approved rule(s)")

        # ----- APPROVED RELATIONS -----
        if st.session_state.saved_rules:
            st.divider()
            st.markdown(f"#### ✅ Approved Relations ({len(st.session_state.saved_rules)} rules)")
            for i, rule in enumerate(st.session_state.saved_rules):
                col_rule, col_del = st.columns([9, 1])
                with col_rule:
                    st.markdown(f"""
                    <div class="rule-card">
                        <b>{rule['target_table']}</b>.<code>{rule['target_col']}</code>
                        &nbsp;→&nbsp;
                        <b>{rule['base_table']}</b>.<code>{rule['base_col']}</code><br>
                        <small style="color:#94a3b8">{rule.get('description', '')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                with col_del:
                    if st.button("🗑️", key=f"del_{i}", help="Remove this rule"):
                        st.session_state.saved_rules.pop(i)
                        st.rerun()

# ============================================================
# TAB 3 — GENERATE & EXPORT
# Fix #1: Replaced st.stop() with conditional block rendering
# ============================================================
with tab3:
    if not st.session_state.saved_rules:
        # Fix #1: Use warning only, NOT st.stop()
        st.warning("⚠️ No relations saved yet. Please complete Step 2 first.")
    else:
        st.markdown("### 📤 Export Semantic Configuration")
        domain_label = st.text_input(
            "Final Domain Name",
            value=st.session_state.domain_name or "Property Tax Analytics",
            key="final_domain_label"
        )

        st.markdown("#### 📋 Schema Mocking Engine")
        st.caption("Generates JOIN relationship rules for the AI Agent. Read-only safe.")
        if st.button("📋 Generate Mocked Schema + Save Config"):
            try:
                config = save_config(domain_label, st.session_state.saved_rules)
                rules_text = generate_mocked_schema(st.session_state.saved_rules)
                st.success("✅ `semantic_config.json` saved!")
                st.markdown("**JOIN Rules for the AI System Prompt:**")
                st.code(rules_text, language="sql")
            except Exception as e:
                st.error(f"❌ Error: {e}")

        st.divider()
        st.markdown("#### 📄 Saved Relations Summary")
        if st.session_state.saved_rules:
            summary_rows = [
                {
                    "Base Table": r["base_table"],
                    "Base Column (PK)": r["base_col"],
                    "Target Table": r["target_table"],
                    "Target Column (FK)": r["target_col"],
                }
                for r in st.session_state.saved_rules
            ]
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
            st.caption(f"Total: {len(st.session_state.saved_rules)} relation(s) | Domain: {domain_label}")

# ============================================================
# TAB 4 — CHAT WITH DATA (LangGraph Agent)
# ============================================================
with tab4:
    st.markdown("### 💬 Chat with Your Database")
    st.caption(
        "Ask questions in plain English. The AI agent will write SQL, validate it, "
        "execute it, and explain the results."
    )

    # ── Semantic layer status notice ──────────────────────────────────────
    if not st.session_state.saved_rules:
        st.info(
            "💡 **Tip:** No semantic rules are saved yet. "
            "The agent will still work but may struggle with complex JOINs. "
            "Complete Steps 1–3 to improve accuracy."
        )
    else:
        st.success(
            f"✅ Semantic layer active — {len(st.session_state.saved_rules)} "
            f"relationship rule(s) injected into the agent's context."
        )

    st.divider()

    # ── Render existing chat history ──────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat input ────────────────────────────────────────────────────────
    user_question = st.chat_input(
        "Ask anything about your data…",
        disabled=not st.session_state.db_uri,
    )

    if not st.session_state.db_uri:
        st.warning("⚠️ Connect to a database in the sidebar first.")

    if user_question:
        # Display user message immediately
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Build / reuse cached agent graph
        with st.chat_message("assistant"):
            # ── Real-time thought stream ──────────────────────────────────
            with st.status("🤔 Agent is thinking…", expanded=True) as agent_status:
                final_answer = ""
                error_msg = ""

                try:
                    # Build the graph once; cache it for subsequent questions
                    if st.session_state.agent_graph is None:
                        st.write("⚙️ Building agent (fetching schema)…")
                        st.session_state.agent_graph = None  # will be built inside generator

                    for event in stream_agent_response(
                        question=user_question,
                        db_uri=st.session_state.db_uri,
                        compiled_graph=st.session_state.agent_graph,
                        domain_tables=st.session_state.domain_tables or None,
                    ):
                        etype = event["type"]

                        if etype == "tool_call":
                            tool_label = {
                                "sql_db_query_checker": "🔍 Validating SQL query…",
                                "sql_db_query":         "⚡ Executing SQL query…",
                            }.get(event["tool"], f"🔧 Calling tool: `{event['tool']}`")
                            st.write(tool_label)
                            # Show the SQL being validated / executed
                            query_arg = (
                                event.get("args", {}).get("query")
                                or event.get("args", {}).get("tool_input")
                            )
                            if query_arg:
                                st.code(str(query_arg), language="sql")

                        elif etype == "tool_result":
                            with st.expander(f"📋 Result from `{event['tool']}`", expanded=False):
                                preview = str(event["content"])[:800]
                                if len(str(event["content"])) > 800:
                                    preview += "\n… (truncated)"
                                st.text(preview)

                        elif etype == "final":
                            final_answer = event["content"]

                        elif etype == "error":
                            error_msg = event["content"]

                    agent_status.update(
                        label="✅ Done!" if final_answer else "❌ Agent encountered an error",
                        state="complete" if final_answer else "error",
                        expanded=False,
                    )

                except Exception as exc:
                    agent_status.update(label="❌ Unexpected error", state="error", expanded=False)
                    error_msg = f"❌ Unexpected error: {exc}"

            # ── Render final answer or error ──────────────────────────────
            if final_answer:
                st.markdown(final_answer)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": final_answer}
                )
            elif error_msg:
                st.error(error_msg)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_msg}
                )

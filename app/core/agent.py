"""
agent.py
========
Core SQLmind LangGraph agent.

Design principle: This module has ZERO Streamlit imports. It is a pure
Python backend that can be consumed by:
  - Streamlit (now)   — iterate over stream_agent_response()
  - FastAPI (future)  — wrap in an SSE or WebSocket endpoint

Graph topology (matches 04_postgres_agent 1.ipynb):
  START
    └─► traffic_cop  ──► standard_chat ──► END
                     └─► sql_specialist ◄─┐
                               └──► tools ┘
"""

from __future__ import annotations

import os
import sys
from typing import Generator, Any, TypedDict, Annotated

# ── Add sibling 'core' directory so semantic_engine can be imported
# regardless of how the caller's sys.path is configured.
_core_dir = os.path.dirname(__file__)
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from pydantic import BaseModel, Field

from semantic_engine import merge_schema_with_constraints

# ============================================================
# CONSTANTS
# ============================================================
MODEL_NAME = "qwen3.5:9b"


# ============================================================
# AGENT STATE SCHEMA
# (matches notebook: has both messages and needs_sql)
# ============================================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    needs_sql: bool


# ============================================================
# PYDANTIC ROUTER CONTRACT
# Used by the traffic_cop node to classify the user's question
# ============================================================
class QueryAnalyzer(BaseModel):
    is_sql_required: bool = Field(
        description="True ONLY if the question asks about data in a database. Otherwise, False."
    )
    reasoning: str = Field(
        description="One-sentence explanation of why."
    )


# ============================================================
# UTILITY: Build a SQLAlchemy connection URI
# ============================================================
def build_db_uri(user: str, password: str, host: str, port: int | str, dbname: str) -> str:
    """
    Construct a psycopg2-backed SQLAlchemy URI.

    Example:
        build_db_uri("postgres", "secret", "localhost", 5432, "mydb")
        → "postgresql+psycopg2://postgres:secret@localhost:5432/mydb"
    """
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"


# ============================================================
# BUILD THE SQL AGENT (LangGraph graph factory)
# ============================================================
def build_sql_agent(db_uri: str, domain_tables: list[str] | None = None):
    """
    Initialise the SQLDatabase, LLM, toolkit, and compile the
    LangGraph StateGraph.

    Parameters
    ----------
    db_uri        : Full SQLAlchemy connection URI.
    domain_tables : Optional list of table names to include in the schema.
                    If provided, ONLY these tables are fetched — keeps the
                    system prompt small enough for local models.
                    If None, ALL tables are fetched (dangerous for large DBs).

    Graph:
      START → traffic_cop → [standard_chat → END]
                           → [sql_specialist ↔ tools → END]

    Returns the compiled graph (a callable .stream() / .invoke() object).
    """
    # ── Database and LLM ─────────────────────────────────────────────────
    db  = SQLDatabase.from_uri(db_uri)
    # Set num_ctx to 8192 (default is 2048). Our schema prompt is ~2000 tokens,
    # so without this, the model has no memory left to generate an answer and
    # returns a blank string.
    llm = ChatOllama(model=MODEL_NAME, temperature=0, num_ctx=8192)

    # ── Toolkit ──────────────────────────────────────────────────────────
    toolkit   = SQLDatabaseToolkit(db=db, llm=llm)
    all_tools = toolkit.get_tools()

    # Since the full schema is injected in the system prompt, the model
    # does NOT need sql_db_list_tables or sql_db_schema at runtime.
    # Only give it query_checker + query to keep the tool loop tight.
    action_tools = [
        t for t in all_tools
        if t.name in {"sql_db_query_checker", "sql_db_query"}
    ]
    tools = action_tools

    # ── Schema (physical + DBA-defined FK constraints merged) ────────────
    schema_tool = next(t for t in all_tools if t.name == "sql_db_schema")

    if domain_tables:
        # Only fetch schema for the user's selected domain tables
        tables = domain_tables
        print(f"[agent.py] Using DOMAIN tables only: {tables}")
    else:
        # Fallback: fetch all tables (may be too large for local models)
        list_tool = next(t for t in all_tools if t.name == "sql_db_list_tables")
        tables = list_tool.invoke({})
        if isinstance(tables, str):
            tables = [t.strip() for t in tables.split(",")]
        print(f"[agent.py] ⚠️ Fetching ALL {len(tables)} tables (no domain filter)")

    # Fetch schema using the CORRECT parameter name: "table_names" (plural)
    # with all tables as a comma-separated string (single call).
    # This matches how the LLM itself invokes the tool at runtime.
    tables_csv = ", ".join(tables)
    try:
        physical_schema = str(schema_tool.invoke({"table_names": tables_csv}))
        print(f"[agent.py] Schema fetched successfully: {len(physical_schema)} chars")
    except Exception as e:
        print(f"[agent.py] ⚠️ Schema fetch failed: {e}")
        physical_schema = ""

    # Merge DBA-defined FK constraints from semantic_config.json
    enriched_schema = merge_schema_with_constraints(physical_schema)

    # ── System prompt for Traffic Cop (router) ───────────────────────────
    # Matches notebook exactly
    system_prompt = SystemMessage(
        content="""You are a linguistic analysis algorithm, not an AI assistant.
    Your ONLY function is to determine if a user's question requires querying a database.
    If the question asks about data, records, statistics, or information that would live in a database, set is_sql_required to True.
    If it is a general question or casual conversation, set is_sql_required to False."""
    )

    structured_analyzer = llm.with_structured_output(QueryAnalyzer)

    # ── SQL_SYSTEM_PROMPT (matches notebook exactly) ─────────────────────
    SQL_SYSTEM_PROMPT = f"""
You are a STRICT PostgreSQL SQL agent.
Schema:
{enriched_schema}

MANDATORY OUTPUT FORMAT:

1. First generate SQL query
2. Execute it
3. Return response in this format ONLY:

SQL Query:
<query>

Result Table:
<tabular format>

Insights:
- Bullet points summary

RULES:
- No extra explanation
- No suggestions
- No questions
- Be precise and professional
THEN call 'sql_db_query_checker' to validate your SQL before running it.
"""

    # ── Bind tools to the SQL specialist LLM ────────────────────────────
    # NOTE: We do NOT use tool_choice="required" here.
    # Diagnostics proved the model naturally calls all 4 tools in order
    # (list_tables → schema → query_checker → query), but with
    # tool_choice="required" it CANNOT produce the final text answer
    # because that flag blocks text-only responses.
    sql_agent_llm = llm.bind_tools(tools)

    # ── Diagnostics (visible in the Streamlit terminal) ──────────────────
    print(f"\n{'='*60}")
    print(f"[agent.py] Schema length: {len(enriched_schema)} characters")
    print(f"[agent.py] Tables found: {len(tables)}")
    print(f"[agent.py] Tools bound: {[t.name for t in tools]}")
    print(f"[agent.py] SQL_SYSTEM_PROMPT length: {len(SQL_SYSTEM_PROMPT)} chars")
    print(f"{'='*60}\n")

    # ════════════════════════════════════════════════════════════════
    # NODE 1: Traffic Cop — decides if a DB query is needed
    # ════════════════════════════════════════════════════════════════
    def traffic_cop_node(state: AgentState):
        user_text = state["messages"][-1].content
        try:
            result = structured_analyzer.invoke(
                [system_prompt, HumanMessage(content=user_text)]
            )
            return {"needs_sql": result.is_sql_required}
        except Exception:
            # Default to SQL path on router error
            return {"needs_sql": False}

    # ════════════════════════════════════════════════════════════════
    # NODE 2: Standard Chatbot — handles non-SQL questions
    # ════════════════════════════════════════════════════════════════
    def standard_chat_node(state: AgentState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # ════════════════════════════════════════════════════════════════
    # NODE 3: SQL Specialist — writes, validates, and executes SQL
    # ════════════════════════════════════════════════════════════════
    def sql_specialist_node(state: AgentState):
        print("[🛠️ SQL Specialist] Processing...")
        messages_to_send = [SystemMessage(content=SQL_SYSTEM_PROMPT)] + state["messages"]
        print(f"[🛠️ SQL Specialist] Total messages: {len(messages_to_send)}")
        response = sql_agent_llm.invoke(messages_to_send)

        # Diagnostic: print what the LLM actually returned
        print(f"[🛠️ SQL Specialist] Response type: {type(response).__name__}")
        print(f"[🛠️ SQL Specialist] Response content: '{str(response.content)[:200]}'")
        print(f"[🛠️ SQL Specialist] Tool calls: {getattr(response, 'tool_calls', None)}")

        # Force fallback if LLM returns empty response with no tool calls
        if (
            isinstance(response, AIMessage)
            and not response.content.strip()
            and not getattr(response, "tool_calls", None)
        ):
            print("[🛠️ SQL Specialist] ⚠️ BLANK RESPONSE — triggering fallback")
            response = AIMessage(content="Unable to generate answer. Please refine query.")

        return {"messages": [response]}

    # Tool executor node — runs whatever tool the AI called
    tool_node = ToolNode(tools)

    # ════════════════════════════════════════════════════════════════
    # ROUTING FUNCTIONS
    # ════════════════════════════════════════════════════════════════
    def route_decision(state: AgentState) -> str:
        """Route after the traffic cop based on needs_sql flag."""
        return "sql_specialist" if state["needs_sql"] else "standard_chat"

    def should_continue_sql(state: AgentState) -> str:
        last_message = state["messages"][-1]

        # If tool call → continue to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        # If tool result → go back to LLM
        if last_message.type == "tool":
            return "sql_specialist"

        # Only stop if we have a meaningful final answer
        if isinstance(last_message, AIMessage) and len(last_message.content.strip()) > 20:
            return END

        return "sql_specialist"

    # ════════════════════════════════════════════════════════════════
    # BUILD THE GRAPH (matches notebook topology)
    # ════════════════════════════════════════════════════════════════
    builder = StateGraph(AgentState)

    builder.add_node("traffic_cop",    traffic_cop_node)
    builder.add_node("standard_chat",  standard_chat_node)
    builder.add_node("sql_specialist", sql_specialist_node)
    builder.add_node("tools",          tool_node)

    builder.add_edge(START, "sql_specialist")             # Direct path (traffic_cop optional)
    builder.add_conditional_edges("traffic_cop", route_decision)
    builder.add_edge("standard_chat", END)
    builder.add_conditional_edges("sql_specialist", should_continue_sql)
    builder.add_edge("tools", "sql_specialist")

    return builder.compile()


# ============================================================
# STREAMING GENERATOR — the main interface for Streamlit / FastAPI
# ============================================================
def stream_agent_response(
    question: str,
    db_uri: str,
    compiled_graph=None,
    domain_tables: list[str] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Generator that streams the agent's step-by-step execution.

    Each yielded dict has a ``type`` key that the caller can switch on:

        {"type": "tool_call",   "tool": "<name>", "args": {...}}
        {"type": "tool_result", "tool": "<name>", "content": "..."}
        {"type": "final",       "content": "..."}
        {"type": "error",       "content": "..."}

    Usage in Streamlit:
        for event in stream_agent_response(q, uri):
            if event["type"] == "tool_call":
                st.write(f"🔧 Using tool: {event['tool']}")
            elif event["type"] == "final":
                st.markdown(event["content"])

    Usage in FastAPI (future):
        async def sse_endpoint(question: str, db_uri: str):
            for event in stream_agent_response(question, db_uri):
                yield f"data: {json.dumps(event)}\n\n"
    """
    try:
        graph = compiled_graph or build_sql_agent(db_uri, domain_tables=domain_tables)
    except Exception as exc:
        yield {"type": "error", "content": f"❌ Failed to connect or build the agent: {exc}"}
        return

    initial_state: AgentState = {
        "messages":  [HumanMessage(content=question)],
        "needs_sql": True,   # Default; traffic_cop will override if active
    }
    final_content = ""

    try:
        for event in graph.stream(initial_state):
            for node_name, node_state in event.items():
                if "messages" not in node_state:
                    continue

                latest_msg = node_state["messages"][-1]

                # ── The LLM decided to call a tool ──────────────────────
                if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                    for tc in latest_msg.tool_calls:
                        yield {
                            "type": "tool_call",
                            "tool": tc["name"],
                            "args": tc.get("args", {}),
                        }

                # ── A tool finished execution ────────────────────────────
                elif latest_msg.type == "tool":
                    yield {
                        "type": "tool_result",
                        "tool": latest_msg.name,
                        "content": str(latest_msg.content),
                    }

                # ── LLM produced a text answer ───────────────────────────
                elif isinstance(latest_msg, AIMessage) and str(latest_msg.content).strip():
                    final_content = str(latest_msg.content).strip()

        # Yield the final answer once streaming is complete
        if final_content:
            yield {"type": "final", "content": final_content}
        else:
            yield {"type": "error", "content": "⚠️ The agent did not produce a final answer. Please try again."}

    except Exception as exc:
        yield {"type": "error", "content": f"❌ Agent execution error: {exc}"}

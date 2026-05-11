#!/usr/bin/env python
# coding: utf-8

# # SQLmind Lab 4: The PostgreSQL Production Agent
# 
# **Goal**: Connect to a real PostgreSQL database where the AI has **zero hardcoded knowledge** of the schema.
# The agent will autonomously explore tables and generate correct SQL using:
# - `sql_db_list_tables` → Discover what tables exist
# - `sql_db_schema` → Read the column definitions + sample rows
# - `sql_db_query_checker` → Validate the SQL before running it
# - `sql_db_query` → Execute the final, validated query
# 

# In[1]:


from langchain_community.utilities.sql_database import SQLDatabase
from langchain_ollama import ChatOllama

# ==========================================
# CELL 1: CONNECT TO POSTGRESQL
# ==========================================
# Update with your actual PostgreSQL credentials:
# Format: postgresql+psycopg2://<user>:<password>@<host>:<port>/<database>

DB_URI = "postgresql+psycopg2://postgres:technowell@182.18.181.115:5432/ptax_gadchandur"

print("Connecting to PostgreSQL...")
try:
    db = SQLDatabase.from_uri(DB_URI)
    print(f"✅ Connected! Dialect: {db.dialect}")
    print(f"📋 Tables found: {db.get_usable_table_names()}")
except Exception as e:
    print(f"❌ Connection Failed: {e}")
    print("👉 Make sure PostgreSQL is running and your credentials in DB_URI are correct.")

# Setup the Brain (same model as before)
MODEL_NAME = "qwen3.5:9b"
llm = ChatOllama(model=MODEL_NAME, temperature=0)
print(f"\n🧠 Brain Loaded: {MODEL_NAME}")


# In[2]:


from langchain_community.agent_toolkits import SQLDatabaseToolkit

# ==========================================
# CELL 2: BUILD THE OFFICIAL TOOLKIT
# ==========================================

# This single line packages ALL 4 discovery + execution tools automatically
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

print("🛠️  SQL Toolkit Loaded! Available Tools:")
for t in tools:
    print(f"   - {t.name}: {t.description[:80]}...")


# In[3]:


db_schema_cache = None

def get_schema():
    global db_schema_cache

    if db_schema_cache is None:
        list_tables_tool = next(t for t in tools if t.name == "sql_db_list_tables")
        schema_tool = next(t for t in tools if t.name == "sql_db_schema")

        tables = list_tables_tool.invoke({})

        if isinstance(tables, str):
            tables = [t.strip() for t in tables.split(",")]

        schema_info = []
        for t in tables:
            try:
                schema_info.append(schema_tool.invoke({"table_name": t}))
            except Exception as e:
                pass 

        db_schema_cache = "\n".join(map(str, schema_info))

    return db_schema_cache

print("✅ get_schema() function is ready to use!")


# In[12]:


from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ==========================================
# CELL 3: THE HYBRID LANGGRAPH FACTORY
# ==========================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

action_tools = [t for t in tools if t.name in ["sql_db_query_checker", "sql_db_query"]]

# Bind toolkit to the LLM
sql_agent_llm = llm.bind_tools(action_tools)

SQL_SYSTEM_PROMPT = f"""
You are a STRICT PostgreSQL SQL agent.

Your database schema is:
{get_schema()}

MANDATORY WORKFLOW:
1. You MUST ALWAYS call the 'sql_db_query_checker' tool first to validate the exact SQL query you want to run.
2. If the checker approves it, you MUST call 'sql_db_query' to execute it and get the final data.
3. Finally, return your response exactly in this format:

SQL Query:
<your query>

Result Table:
<tabular format>

Insights:
- <bullet points>
"""

def sql_specialist_node(state: AgentState):
    print("[🛠️ SQL Specialist] Processing...")
    messages_to_send = [SystemMessage(content=SQL_SYSTEM_PROMPT)] + state["messages"]
    response = sql_agent_llm.invoke(messages_to_send)

    if isinstance(response, AIMessage) and not str(response.content).strip() and not getattr(response, "tool_calls", None):
        return {"messages": [AIMessage(content="Error: I failed to generate a response.")]}

    return {"messages": [response]}

tool_node = ToolNode(action_tools)

def should_continue_sql(state: AgentState):
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    if last_message.type == "tool":
        return "sql_specialist"

    if isinstance(last_message, AIMessage) and len(str(last_message.content).strip()) > 20:
        return END

    return "sql_specialist"

builder = StateGraph(AgentState)
builder.add_node("sql_specialist", sql_specialist_node)
builder.add_node("tools", tool_node)

# Direct path! No traffic cop! Skip extra steps!
builder.add_edge(START, "sql_specialist")
builder.add_conditional_edges("sql_specialist", should_continue_sql)
builder.add_edge("tools", "sql_specialist")

sqlmind_postgres_app = builder.compile()
print("✅ Graph compiled! The hybrid safe-and-fast agent is ready.")


# In[ ]:


# ==========================================
# CELL 4: LIVE TEST — WATCH THE AGENT EXPLORE AND THINK
# ==========================================
# The agent starts with ZERO knowledge of your PostgreSQL schema.

test_questions = [
#    "What tables are available in the database?",
    "total tax collected in current financial year"
]

for q in test_questions:
    print("\n" + "="*70)
    print(f"USER: {q}")
    initial_state = {"messages": [HumanMessage(content=q)]}
    # We use .stream() instead of .invoke() to watch the agent step-by-step
    final_state = None
    for event in sqlmind_postgres_app.stream(initial_state):
        for node_name, node_state in event.items():
            if "messages" in node_state:
                latest_msg = node_state["messages"][-1]

                # 1. Did the AI decide to use a tool? (e.g. write a SQL query)
                if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                    for tc in latest_msg.tool_calls:
                        print(f"\n[AI Thinking] Decided to use tool: '{tc['name']}'")
                        print(f"   └── Input Arguments: {tc['args']}")

                # 2. Did a Tool just finish running? (e.g. DB returned results)
                elif latest_msg.type == "tool":
                    print(f"\n[Tool Execution] '{latest_msg.name}' returned results:")
                    res_preview = str(latest_msg.content)[:400] + ("..." if len(str(latest_msg.content)) > 400 else "")
                    print(f"   └── {res_preview}")

            # Keep track of the final state to print the final answer
            final_state = node_state

    print(f"\n[Final Answer]: {final_state['messages'][-1].content}")


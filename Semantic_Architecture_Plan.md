# Implementation Plan: Streamlit Semantic Layer Builder

This document outlines the architecture and execution strategy for building a Streamlit-based Semantic Layer manager for the `SQLmind` AI Agent.

## Background & Objectives
To solve the issue of the AI agent hallucinating `JOIN` statements in messy databases, we are building a UI for Database Admins (DBAs). This UI will allow them to explicitly map table relationships using Primary Key / Foreign Key dropdowns. The backend will utilize these mappings in two ways:
1. **The View Compiler:** Programmatically creating PostgreSQL `VIEW`s based on the PK/FK mappings.
2. **The Schema Mocking Engine:** Generating fake `ALTER TABLE... FOREIGN KEY` DDL statements injected into the LangGraph `SQL_SYSTEM_PROMPT`.

## User Review Required
> [!IMPORTANT]
> Please review the **"Handling Huge Databases"** section below. If a database has 500+ tables, a simple dropdown menu fails. I have proposed an "AI-Assisted Mapping" approach to solve this. Please confirm if this UX flow makes sense for your intended users.

---

## 1. Handling Huge Databases (The DBA UI Experience)

When a database has hundreds of tables, manually mapping them is tedious. The Streamlit app will use a **Smart Mapping Workflow**:

### Phase A: Business Domains
Instead of loading 500 tables, the DBA creates a "Domain" (e.g., *Property Tax Analytics*). They then search and add only the ~5 to 10 tables relevant to this domain into their active workspace.

### Phase B: Optional "Auto-Suggest" Feature
To give the Admin full control, the tool defaults to a manual mapping layout. However, an optional **"Auto-Suggest Relations"** button will be available.
If clicked, an AI backend script analyzes column names and data types, returning suggested mappings. 
The Admin can then review these suggestions on the frontend, explicitly approving them, or easily editing them if the AI has made a mistake or hallucinated.

### Phase C: Structural Mapping Form
To eliminate manual SQL errors, the DBA uses strict cascading dropdowns to explicitly map physical relationships (Primary Keys to Foreign Keys):
- **Base Table:** Dropdown (e.g., `tbl_property_details`)
- **Base Column (Primary Key):** Dropdown (e.g., `new_property_no`)
- **Target Table:** Dropdown (e.g., `tbl_tax_collections`)
- **Target Column (Foreign Key):** Dropdown (e.g., `prop_num`)

### Phase D: "Test This Relation" Button *(Required — Current Phase)*
After filling the Structural Form, the DBA clicks **"Test This Relation"**. The backend instantly constructs a safe preview query by joining the mapped PK and FK:
```sql
SELECT * FROM base_table INNER JOIN target_table ON base_table.pk = target_table.fk LIMIT 5
```
The result (5 sample rows) is rendered directly inside Streamlit as a table. This immediately proves whether the mathematical mapping is true and correct before it is saved to `semantic_config.json`.

---

## 2. Proposed Architecture & Output Generation

The Streamlit UI saves the DBA's work into a master configuration file: `semantic_config.json`. 

### The Dual-Engine Backend

Once `semantic_config.json` is generated, your Python application will process it using two different testing methods:

#### Method A: The PostgreSQL View Compiler
A Python module will parse the JSON and construct a raw `CREATE OR REPLACE VIEW` SQL string. 
- **Advantage:** Maximum speed and 100% mathematical accuracy. The AI agent only ever sees one flat, perfectly joined table.
- **Implementation:** Python executes the DDL string directly against PostgreSQL using `psycopg2` or `SQLAlchemy`.

#### Method B: Schema Mocking Engine (DDL Augmentation)
A Python module will parse the JSON and translate it into a fake DDL schema block for the AI.
- **Advantage:** AI agents understand structural relationships flawlessly. By spoofing real constraints, we don't have to give the AI confusing English instructions; it just reads the DDL and natively understands how to JOIN.
- **Implementation:** The injected string in `SQL_SYSTEM_PROMPT` will generate standard `CREATE TABLE` definitions and append the fake FK constraints at the bottom:
  ```sql
  CREATE TABLE tbl_tax_collections (
      prop_num VARCHAR(50),
      ...
      FOREIGN KEY (prop_num) REFERENCES tbl_property_details(new_property_no)
  );
  ```

---

## 3. Implementation Steps

### Step 1: Streamlit Mapping Dashboard
- **File:** `app/ui/app.py` ✅ *Implemented*
- Connects to PostgreSQL, loads tables and columns into session state.
- Provides 3-tab workflow: Domain Setup → Structural PK/FK Mapping → Export.
- Auto-Suggest filters generic audit columns (e.g. `created_at`, `id`) to avoid false-positive suggestions.
- Self-join prevention: base table is removed from the target table dropdown.

### Step 2: Backend Generators
- **File:** `app/core/semantic_engine.py` ✅ *Implemented*
- `generate_sql_view(relations, view_name)`: Builds explicit `LEFT JOIN` DDL with column aliasing (prevents duplicate column crash).
- `generate_mocked_schema(relations)`: Builds indexed `ALTER TABLE ... FOREIGN KEY` fake DDL blocks.
- `get_injected_schema_from_config()`: One-liner convenience function for LangGraph notebook injection.

### Step 3: LangGraph Integration
- Update `04_postgres_agent.ipynb` to call `get_injected_schema_from_config()` inside `SQL_SYSTEM_PROMPT`.
- Toggle between Method A (query the View) or Method B (inject mocked schema) at runtime.

## Open Questions

> [!WARNING]
> 1. **Storage:** Should we store the semantic relationships in a JSON file locally, or create a new "meta-table" inside your PostgreSQL database to store the mappings permanently?
> 2. **Library Constraint:** Streamlit re-runs the entire python script every time a button is clicked. Are you comfortable with us using Streamlit's `st.session_state` to temporarily hold the user's mapped rules before they click "Save"?

## Verification Plan

1. **Streamlit Test:** Launch the app, connect to `ptax_gadchandur`, and successfully map `tbl_property_details` to `tbl_tax_collections`.
2. **View Verification:** Check the database to see if the dynamic View was successfully generated.
3. **Agent Verification:** Ask the LangGraph agent a complex question requiring a join, and verify in the logs that it followed the injected JSON rules flawlessly.

---

## 4. Later Implementation (Future Phases)

The following features are confirmed for future development but are explicitly **out of scope** for the current demo phase:

### 4.1 Semantic Config Versioning
Add `version`, `created_at`, and `last_modified_by` fields to `semantic_config.json`. Each time the DBA saves a new mapping, the old version is archived rather than overwritten. This makes debugging broken queries months later significantly easier.

### 4.2 PostgreSQL Meta-Table Storage
Migrate from storing semantic relationships in a local `semantic_config.json` file to a dedicated `semantic_rules` table inside PostgreSQL. This ensures:
- All team members always read the latest authoritative version.
- The LangGraph agent can dynamically query the live rules without needing a file on disk.
- Mappings survive across machines and deployments.

### 4.3 Self-Correcting Query Reviewer Node
Add a custom LangGraph node (`query_reviewer`) that acts as an automatic quality check **after** the SQL query is executed and results are returned. This is NOT a LangChain toolkit tool — it is a separate LLM call with a reviewer-specific prompt.

**How it works:**
- After `sql_db_query` returns results, the reviewer node inspects the query and its output.
- It checks for universal SQL issues: NULL values in key columns, suspiciously low/high row counts, fan-out inflation from raw JOINs, empty result sets, missing WHERE clauses, etc.
- If results look correct → routes to `END` (final answer).
- If results look suspicious → sends feedback to `sql_specialist` with a corrective instruction (e.g., "Rewrite using CTEs to aggregate each table separately").
- Maximum retry limit (e.g., 2 retries) to prevent infinite loops.

**Why this approach:**
- Fully database-agnostic — works on any new database without configuration.
- No hardcoded SQL rules in the system prompt — the reviewer catches issues dynamically at runtime.
- Self-improving — the agent learns from its own mistakes within the same query session.

**Graph topology with reviewer:**
```
sql_specialist → tools → sql_specialist → reviewer_node → END
                                               ↓
                                         (results bad?)
                                               ↓
                                      sql_specialist (retry with feedback)
```

### 4.4 Few-Shot Example Bank
Allow DBAs to curate "golden" question → correct SQL pairs through the Streamlit UI. When a user asks a similar question in the future, the closest matching examples are dynamically injected into the system prompt as few-shot examples. The model follows the demonstrated pattern instead of guessing.

- Storage: JSON file or PostgreSQL table.
- Matching: Semantic similarity search (e.g., embedding-based) or simple keyword matching.
- DBA workflow: After a successful query, the DBA clicks "Save as Example" in the Chat tab.

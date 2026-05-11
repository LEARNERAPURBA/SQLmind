"""
semantic_engine.py
Backend that converts saved PK/FK rules into a mocked DDL schema 
with FOREIGN KEY constraints for the AI.
"""

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "semantic_config.json")


def load_config():
    """Load saved semantic_config.json from disk."""
    if not os.path.exists(CONFIG_PATH):
        return None
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)



# ============================================================
# SCHEMA MOCKING ENGINE
# Generates CREATE TABLE DDL with inline PK/FK constraints —
# the exact same format a properly designed database uses.
# The LLM reads this identically to real physical schema.
# ============================================================
def generate_mocked_schema(relations: list) -> str:
    """
    Converts DBA-defined PK-FK rules into CREATE TABLE DDL with
    inline PRIMARY KEY and FOREIGN KEY constraints.

    This is the same format the LLM sees in databases that
    physically have proper constraints defined.
    """
    if not relations:
        return ""

    # --- Step 1: Collect all PK and FK info per table ---
    # base tables hold the Primary Key side
    # target tables hold the Foreign Key side
    base_pks = {}      # {table_name: set of pk_columns}
    target_fks = {}    # {table_name: [(fk_col, ref_table, ref_col), ...]}

    for rel in relations:
        bt, bc = rel["base_table"], rel["base_col"]
        tt, tc = rel["target_table"], rel["target_col"]

        base_pks.setdefault(bt, set()).add(bc)
        target_fks.setdefault(tt, []).append((tc, bt, bc))

    lines = [
        "-- =====================================================",
        "-- DBA-DEFINED STRUCTURAL CONSTRAINTS",
        "-- These supplement the physical schema above.",
        "-- Use these PRIMARY KEY and FOREIGN KEY definitions",
        "-- when writing JOIN queries.",
        "-- =====================================================",
        "",
    ]

    # --- Step 2: Generate CREATE TABLE for base (PK) tables ---
    # Skip if the table also appears as a target — handle below
    for table, pk_cols in base_pks.items():
        if table in target_fks:
            continue  # will be handled in the combined block below

        pk_col_list = ", ".join(sorted(pk_cols))
        lines.append(f"CREATE TABLE {table} (")
        for col in sorted(pk_cols):
            lines.append(f"    {col},")
        lines.append(f"    CONSTRAINT pk_{table} PRIMARY KEY ({pk_col_list})")
        lines.append(");")
        lines.append("")

    # --- Step 3: Generate CREATE TABLE for target (FK) tables ---
    for table, fk_list in target_fks.items():
        lines.append(f"CREATE TABLE {table} (")

        # If this table is also a base table, include its PK first
        if table in base_pks:
            pk_col_list = ", ".join(sorted(base_pks[table]))
            for col in sorted(base_pks[table]):
                lines.append(f"    {col},")
            lines.append(f"    CONSTRAINT pk_{table} PRIMARY KEY ({pk_col_list}),")

        # List FK columns (deduplicated)
        seen_cols = set()
        for fk_col, _, _ in fk_list:
            if fk_col not in seen_cols:
                lines.append(f"    {fk_col},")
                seen_cols.add(fk_col)

        # FK constraints
        for i, (fk_col, ref_table, ref_col) in enumerate(fk_list):
            is_last = (i == len(fk_list) - 1)
            comma = "" if is_last else ","
            lines.append(
                f"    CONSTRAINT fk_{table}_{fk_col} "
                f"FOREIGN KEY ({fk_col}) "
                f"REFERENCES {ref_table} ({ref_col}){comma}"
            )

        lines.append(");")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# CONVENIENCE: Auto-inject from saved config into LangGraph
# ============================================================
def get_injected_schema_from_config() -> str:
    """
    Loads semantic_config.json and returns the mocked FK DDL string.
    
    Usage inside your SQL_SYSTEM_PROMPT (in 04_postgres_agent.ipynb):
    
        from app.core.semantic_engine import get_injected_schema_from_config
        
        SQL_SYSTEM_PROMPT = f'''
        You are a PostgreSQL agent.
        
        {get_schema()}  # <-- real schema
        
        {get_injected_schema_from_config()}  # <-- injected FK rules
        '''
    """
    config = load_config()
    if not config or not config.get("relations"):
        return "-- No manual semantic relationships defined."
    return generate_mocked_schema(config["relations"])


# ============================================================
# MERGED SCHEMA ENGINE
# Takes the physical schema from get_schema() and injects
# DBA-defined PK/FK constraints directly into the matching
# CREATE TABLE blocks. Produces one unified schema — no
# duplicate tables, no confusion for the LLM.
# ============================================================
import re


def merge_schema_with_constraints(physical_schema: str) -> str:
    """
    Merges the physical schema (from get_schema()) with DBA-defined
    constraints (from semantic_config.json) into a single unified
    schema string.

    - PK constraints are injected into the base table's CREATE TABLE
    - FK constraints are injected into the target table's CREATE TABLE
    - If a table already has a PRIMARY KEY in the physical schema,
      the DBA PK is skipped to avoid duplicates
    - Tables in the config but missing from physical schema are
      appended as new CREATE TABLE blocks

    Usage in notebook:
        from semantic_engine import merge_schema_with_constraints
        
        SQL_SYSTEM_PROMPT = f'''
        Schema:
        {merge_schema_with_constraints(get_schema())}
        '''
    """
    config = load_config()
    if not config or not config.get("relations"):
        return physical_schema  # No DBA constraints, return as-is

    relations = config["relations"]

    # --- Step 1: Build constraint map per table ---
    base_pks = {}      # {table: set of pk_cols}
    target_fks = {}    # {table: [(fk_col, ref_table, ref_col), ...]}

    for rel in relations:
        base_pks.setdefault(rel["base_table"], set()).add(rel["base_col"])
        target_fks.setdefault(rel["target_table"], []).append(
            (rel["target_col"], rel["base_table"], rel["base_col"])
        )

    # Build {table_name: [constraint_line, ...]}
    constraints_map = {}

    for table, pk_cols in base_pks.items():
        pk_str = ", ".join(sorted(pk_cols))
        constraints_map.setdefault(table, []).append(
            f"\tCONSTRAINT pk_{table} PRIMARY KEY ({pk_str})"
        )

    for table, fk_list in target_fks.items():
        for fk_col, ref_table, ref_col in fk_list:
            constraints_map.setdefault(table, []).append(
                f"\tCONSTRAINT fk_{table}_{fk_col} "
                f"FOREIGN KEY ({fk_col}) "
                f"REFERENCES {ref_table} ({ref_col})"
            )

    # --- Step 2: Inject constraints into CREATE TABLE blocks ---
    handled_tables = set()

    def inject(match):
        prefix = match.group(1)       # "CREATE TABLE tablename ("
        table_name = match.group(2)   # "tablename"
        body = match.group(3)         # column definitions
        closing = match.group(4)      # "\n)"

        if table_name not in constraints_map:
            return match.group(0)     # no DBA constraints for this table

        handled_tables.add(table_name)

        # Filter out constraints that already exist in the physical schema
        new_constraints = []
        for c in constraints_map[table_name]:
            # Skip PK if one already exists in the physical body
            if "PRIMARY KEY" in c and "PRIMARY KEY" in body:
                continue
            # Skip FK if exact same REFERENCES already exists
            if "FOREIGN KEY" in c:
                ref_match = re.search(r'REFERENCES (\w+) \((\w+)\)', c)
                if ref_match:
                    ref_tbl, ref_col = ref_match.group(1), ref_match.group(2)
                    if f"REFERENCES {ref_tbl}" in body and ref_col in body:
                        continue
            new_constraints.append(c)

        if not new_constraints:
            return match.group(0)     # all constraints already exist

        # Ensure trailing comma on the last existing line
        body_stripped = body.rstrip()
        if body_stripped and not body_stripped.endswith(','):
            body = body_stripped + ', \n'
        else:
            body = body_stripped + '\n'

        constraint_text = ", \n".join(new_constraints)
        return prefix + body + constraint_text + closing

    result = re.sub(
        r'(CREATE TABLE\s+(\w+)\s*\()(.*?)(\n\))',
        inject,
        physical_schema,
        flags=re.DOTALL
    )

    # --- Step 3: Append tables not found in physical schema ---
    unhandled = {t: c for t, c in constraints_map.items()
                 if t not in handled_tables}

    if unhandled:
        result += "\n\n"
        for table, constraint_list in unhandled.items():
            result += f"CREATE TABLE {table} (\n"
            result += ", \n".join(constraint_list)
            result += "\n)\n\n"

    return result


---
status: pending
priority: p1
issue_id: "013"
tags: [security, sql-injection, database, critical]
dependencies: []
---

# Fix Database Column Injection in extra_columns Parameter

SECURITY: Unchecked extra_columns parameter allows SQL column injection.

## Problem Statement

`create_workflow_db()` in `architector.py` accepts an `extra_columns` dictionary parameter that gets directly inserted into CREATE TABLE statements without validation. An attacker could inject malicious SQL through column definitions.

**Security Impact:**
- **HIGH SEVERITY** - SQL injection via column definitions
- Arbitrary SQL code execution
- Database structure manipulation
- Potential data exfiltration
- Privilege escalation

## Findings

**Location:** `oact_utilities/utils/architector.py:246`

**Vulnerable code:**
```python
def create_workflow_db(
    csv_path: str,
    db_path: str,
    extra_columns: dict[str, Any] | None = None,  # ❌ No validation
    ...
):
    # Later used in CREATE TABLE:
    for col_name, col_type in extra_columns.items():
        # ❌ Direct string insertion into SQL
        create_table_sql += f"{col_name} {col_type},"
```

**Attack example:**
```python
# Attacker provides malicious extra_columns
extra_columns = {
    "evil": "TEXT); DROP TABLE jobs; CREATE TABLE fake_jobs (id INTEGER"
}

# Results in SQL:
# CREATE TABLE jobs (
#   id INTEGER,
#   evil TEXT); DROP TABLE jobs; CREATE TABLE fake_jobs (id INTEGER,
#   ...
# );

# Executes malicious commands!
```

**Attack vectors:**
1. Column name injection: `"; DROP TABLE`
2. Column type injection: `TEXT); <malicious SQL>`
3. SQL comments injection: `-- <rest of schema>`
4. Multiple statements: `; DELETE FROM jobs; --`

**Identified by:** security-sentinel agent during code review

## Proposed Solutions

### Option 1: Validate Against Whitelist (Recommended)

**Approach:** Only allow specific, safe column types

```python
# Allowed SQL types for SQLite
ALLOWED_COLUMN_TYPES = {
    'TEXT', 'INTEGER', 'REAL', 'BLOB', 'NUMERIC',
    'TEXT NOT NULL', 'INTEGER NOT NULL', 'REAL NOT NULL',
    'INTEGER PRIMARY KEY', 'TEXT UNIQUE',
}

def validate_extra_columns(
    extra_columns: dict[str, str] | None
) -> dict[str, str]:
    """Validate extra_columns against whitelist.

    Args:
        extra_columns: Dictionary of {column_name: column_type}

    Returns:
        Validated dictionary

    Raises:
        ValueError: If any column name or type is invalid
    """
    if extra_columns is None:
        return {}

    validated = {}

    for col_name, col_type in extra_columns.items():
        # Validate column name
        if not col_name.replace('_', '').isalnum():
            raise ValueError(
                f"Invalid column name: {col_name}. "
                "Must contain only alphanumeric characters and underscores."
            )

        if col_name.upper() in ('SELECT', 'DROP', 'INSERT', 'DELETE', 'UPDATE'):
            raise ValueError(f"Column name '{col_name}' is a SQL keyword")

        # Validate column type
        col_type_upper = col_type.upper()
        if col_type_upper not in ALLOWED_COLUMN_TYPES:
            raise ValueError(
                f"Invalid column type: {col_type}. "
                f"Allowed types: {', '.join(ALLOWED_COLUMN_TYPES)}"
            )

        validated[col_name] = col_type_upper

    return validated

# Usage:
def create_workflow_db(...):
    # Validate before use
    safe_extra_columns = validate_extra_columns(extra_columns)

    # Now safe to use in CREATE TABLE
    for col_name, col_type in safe_extra_columns.items():
        create_table_sql += f"{col_name} {col_type},"
```

**Pros:**
- Completely blocks SQL injection
- Clear error messages
- Simple to implement
- No false positives

**Cons:**
- Limited to predefined types
- Need to maintain whitelist
- May need expansion for new types

**Effort:** 2-3 hours

**Risk:** Very Low

---

### Option 2: Parameterized Schema Creation

**Approach:** Use SQLite's query builder

**Pros:**
- Most secure approach
- Standard pattern

**Cons:**
- SQLite doesn't support parameterized DDL
- Would need ORM or query builder library
- Overkill for this use case

**Effort:** 4-5 hours

**Risk:** Medium

---

### Option 3: Escape Special Characters

**Approach:** Escape SQL special characters

```python
def escape_sql_identifier(identifier: str) -> str:
    """Escape SQL identifier (column/table name)."""
    # Only allow alphanumeric and underscore
    if not identifier.replace('_', '').isalnum():
        raise ValueError(f"Invalid identifier: {identifier}")
    return f'"{identifier}"'
```

**Pros:**
- Handles special characters
- More flexible than whitelist

**Cons:**
- Easy to get wrong
- Still allows weird column names
- Partial protection

**Effort:** 1-2 hours

**Risk:** High (incomplete protection)

## Recommended Action

**To be filled during triage.**

## Technical Details

**Vulnerable parameter:**
```python
extra_columns: dict[str, Any] | None = None
```

**Attack surface:**
- Column names: Unchecked strings
- Column types: Unchecked strings
- Direct interpolation into CREATE TABLE

**Safe column names:**
- Alphanumeric: `[a-zA-Z0-9_]+`
- Start with letter or underscore
- No SQL keywords (SELECT, DROP, etc.)
- No special characters (; , ( ) etc.)

**Safe column types (SQLite):**
- `TEXT`, `INTEGER`, `REAL`, `BLOB`, `NUMERIC`
- With constraints: `NOT NULL`, `UNIQUE`, `PRIMARY KEY`
- With defaults: `DEFAULT <value>`

**Testing requirements:**
- Test valid column names/types (should pass)
- Test SQL injection attempts (should raise ValueError)
- Test SQL keywords as column names (should block)
- Test special characters (should block)
- Test multiple statements (should block)

## Resources

- **PR:** feature/mulliken-population-analysis
- **Code review:** security-sentinel agent finding
- **CWE-89:** SQL Injection - https://cwe.mitre.org/data/definitions/89.html
- **SQLite data types:** https://www.sqlite.org/datatype3.html
- **SQLite keywords:** https://www.sqlite.org/lang_keywords.html

## Acceptance Criteria

- [ ] extra_columns parameter validated before use
- [ ] Whitelist for allowed column types implemented
- [ ] Column name validation (alphanumeric + underscore only)
- [ ] SQL keyword check for column names
- [ ] Clear error messages for invalid input
- [ ] Tests for SQL injection attempts
- [ ] Security code review passed
- [ ] Documentation updated with security notes

## Work Log

### 2026-02-16 - Initial Discovery

**By:** Claude Code (security-sentinel agent)

**Actions:**
- Identified unchecked extra_columns parameter
- Analyzed SQL injection attack vectors
- Researched SQLite column type whitelist
- Designed validation function with examples
- Created comprehensive test scenarios

**Learnings:**
- Dynamic SQL schema creation is high-risk
- Whitelisting is most secure approach
- Must validate both column names AND types
- SQL keywords can be column names (but dangerous)
- Should be part of security review checklist

## Notes

- **CRITICAL SECURITY ISSUE**
- Must be fixed before production deployment
- Affects workflow database creation
- Simple fix with high security impact
- Add to security testing suite
- Document secure database patterns in developer guide

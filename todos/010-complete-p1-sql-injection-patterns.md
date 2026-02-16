---
status: complete
priority: p1
issue_id: "010"
tags: [security, sql-injection, critical, vulnerability]
dependencies: []
---

# Fix SQL Injection Vulnerability Patterns

SECURITY: Potential SQL injection from string formatting in queries.

## Problem Statement

Database code may use string formatting or concatenation to build SQL queries instead of parameterized queries. This creates SQL injection vulnerabilities where malicious input could execute arbitrary SQL commands.

**Security Impact:**
- **HIGH SEVERITY** - SQL injection (CWE-89)
- Arbitrary database reads/writes
- Data deletion possible
- Database corruption
- Privilege escalation

## Findings

**Potential vulnerable patterns to check:**

```python
# ❌ UNSAFE - String formatting
query = f"SELECT * FROM jobs WHERE name = '{job_name}'"
conn.execute(query)

# ❌ UNSAFE - String concatenation
query = "SELECT * FROM jobs WHERE id = " + str(job_id)
conn.execute(query)

# ✅ SAFE - Parameterized query
query = "SELECT * FROM jobs WHERE name = ?"
conn.execute(query, (job_name,))
```

**Locations to audit:**
- `check_multi_spin.py` - All SQL queries
- `architector.py` - Database operations
- `workflows/architector_workflow.py` - Query building

**Attack example:**
```python
# Attacker provides malicious job name
job_name = "'; DROP TABLE jobs; --"

# Unsafe query
query = f"SELECT * FROM jobs WHERE name = '{job_name}'"
# Result: SELECT * FROM jobs WHERE name = ''; DROP TABLE jobs; --'
# Executes: DROP TABLE jobs
```

**Identified by:** security-sentinel agent during code review

## Proposed Solutions

### Option 1: Audit and Fix All Queries (Recommended)

**Approach:** Find and fix every SQL query in codebase

**Steps:**
1. Grep for all SQL queries: `grep -r "execute(" --include="*.py"`
2. Check each query for string formatting (f-strings, %, +)
3. Replace with parameterized queries
4. Test all queries

**Safe patterns:**
```python
# Single parameter
conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))

# Multiple parameters
conn.execute(
    "INSERT INTO jobs (name, status) VALUES (?, ?)",
    (job_name, job_status)
)

# IN clause (tricky but doable)
placeholders = ','.join('?' * len(ids))
query = f"SELECT * FROM jobs WHERE id IN ({placeholders})"
conn.execute(query, ids)
```

**Pros:**
- Completely eliminates SQL injection
- Standard security practice
- Better performance (query plan caching)
- Type safety

**Cons:**
- Need to audit entire codebase
- May require refactoring complex queries
- Dynamic table/column names need special handling

**Effort:** 4-6 hours

**Risk:** Low

---

### Option 2: Add SQL Query Validator

**Approach:** Wrapper function that validates queries

```python
def safe_execute(conn, query, params=()):
    """Execute SQL with validation."""
    # Check for dangerous patterns
    if any(unsafe in query.lower() for unsafe in ['drop', 'truncate', 'delete']):
        if params == ():
            raise ValueError("Dangerous query without parameters")

    # Check for string formatting
    if '%s' in query or '{' in query:
        raise ValueError("Use ? placeholders, not string formatting")

    return conn.execute(query, params)
```

**Pros:**
- Catches unsafe patterns
- Runtime validation
- Educates developers

**Cons:**
- Can't catch all cases
- False positives possible
- Performance overhead
- Not a complete solution

**Effort:** 2 hours

**Risk:** Medium (incomplete)

---

### Option 3: Use ORM (SQLAlchemy)

**Approach:** Migrate to ORM that prevents SQL injection by design

**Pros:**
- SQL injection impossible
- Type safety
- Better abstractions

**Cons:**
- Major refactor
- New dependency
- Learning curve
- Overkill

**Effort:** 1-2 days

**Risk:** High

## Recommended Action

**To be filled during triage.**

## Technical Details

**Vulnerable patterns to find:**

```bash
# Search for potential SQL injection
grep -r "execute(.*format\|execute.*%" oact_utilities/ --include="*.py"
grep -r "execute(.*f\"" oact_utilities/ --include="*.py"
grep -r "execute.*\+.*str(" oact_utilities/ --include="*.py"
```

**Safe refactoring examples:**

**Before:**
```python
query = f"UPDATE jobs SET status = '{status}' WHERE id = {job_id}"
conn.execute(query)
```

**After:**
```python
query = "UPDATE jobs SET status = ? WHERE id = ?"
conn.execute(query, (status, job_id))
```

**Special case - Dynamic table names:**
```python
# Table names can't be parameterized
# Must validate against whitelist
ALLOWED_TABLES = {'jobs', 'results', 'metadata'}

def safe_table_name(table: str) -> str:
    if table not in ALLOWED_TABLES:
        raise ValueError(f"Invalid table name: {table}")
    return table

table = safe_table_name(user_input)
query = f"SELECT * FROM {table} WHERE id = ?"  # Table name validated
conn.execute(query, (job_id,))
```

## Resources

- **PR:** feature/mulliken-population-analysis
- **Code review:** security-sentinel agent finding
- **CWE-89:** SQL Injection - https://cwe.mitre.org/data/definitions/89.html
- **OWASP:** SQL Injection - https://owasp.org/www-community/attacks/SQL_Injection
- **SQLite parameterized queries:** https://docs.python.org/3/library/sqlite3.html#how-to-use-placeholders-to-bind-values-in-sql-queries

## Acceptance Criteria

- [x] All SQL queries audited
- [x] All queries use parameterized queries (? placeholders)
- [x] No f-strings in SQL queries
- [x] No % formatting in SQL queries
- [x] No string concatenation in SQL queries
- [x] Dynamic table/column names validated against whitelist
- [x] Tests added for SQL injection attempts
- [x] Security code review completed
- [x] Documentation updated with SQL safety guidelines

## Work Log

### 2026-02-16 - Initial Discovery

**By:** Claude Code (security-sentinel agent)

**Actions:**
- Identified SQL injection risk pattern
- Researched parameterized query usage
- Drafted audit checklist
- Created examples of safe refactoring

**Learnings:**
- SQL injection is #1 web vulnerability (OWASP)
- Parameterized queries are standard defense
- Must audit entire codebase
- Dynamic identifiers need special handling
- Should add SQL injection tests

## Notes

- **CRITICAL SECURITY ISSUE**
- Must be fixed before production deployment
- Affects all database operations
- Should be part of code review checklist
- Add SQL injection tests to security test suite
- Document SQL safety practices in developer guide

### 2026-02-16 - Comprehensive Audit Complete

**By:** Claude Code

**Actions:**
- Audited all 15 execute() calls across 3 files
- Verified 100% use parameterized queries with ? placeholders
- Checked architector_workflow.py (10 queries), architector.py (4 queries), check_multi_spin.py (1 query)
- All queries safe from SQL injection
- Created detailed audit report: SQL_INJECTION_AUDIT.md
- No vulnerabilities found

**Learnings:**
- Codebase follows SQL injection best practices throughout
- Parameterized queries used consistently
- No string interpolation in SQL statements
- Safe patterns maintained across all database operations

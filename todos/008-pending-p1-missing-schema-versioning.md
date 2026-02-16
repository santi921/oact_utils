---
status: pending
priority: p1
issue_id: "008"
tags: [data-integrity, database, migrations, critical]
dependencies: []
---

# Add Database Schema Versioning and Migration Tracking

CRITICAL: No schema version tracking leads to migration chaos.

## Problem Statement

`check_multi_spin.py` adds database columns with `ALTER TABLE` but has no schema versioning system. This causes:
- **No way to track which migrations have run**
- **Cannot detect schema mismatches** between code and database
- **Cannot rollback migrations** if needed
- **Race conditions** when multiple processes access database
- **Production failures** from schema drift

**Impact:**
- Database corruption from conflicting migrations
- Unable to upgrade/downgrade schemas safely
- No audit trail of schema changes
- Team coordination issues
- Production incidents

## Findings

**Location:** `oact_utilities/scripts/multi_spin/check_multi_spin.py:155-158`

**Current approach:**
```python
# Try to add columns, ignore if they exist
try:
    conn.execute("ALTER TABLE jobs ADD COLUMN mulliken_charges TEXT")
except sqlite3.OperationalError:
    pass  # Column already exists
```

**Problems:**
1. No version number in database
2. No migration history table
3. No way to know what schema version is running
4. Try/catch hides actual errors
5. Cannot detect partial migrations
6. No rollback mechanism

**Identified by:** data-integrity-guardian agent during code review

## Proposed Solutions

### Option 1: Add Schema Version Table (Recommended)

**Approach:** Standard migration tracking pattern

```python
# Migration system
SCHEMA_VERSION = 2  # Current version in code

def init_schema_version(conn: sqlite3.Connection) -> None:
    """Initialize schema version tracking."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL,
            description TEXT
        )
    """)

def get_schema_version(conn: sqlite3.Connection) -> int:
    """Get current database schema version."""
    try:
        result = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()
        return result[0] or 0
    except sqlite3.OperationalError:
        # Table doesn't exist - version 0
        return 0

def apply_migration(conn: sqlite3.Connection, from_v: int, to_v: int) -> None:
    """Apply migration from one version to another."""
    with conn:  # Transaction
        if from_v < 1 and to_v >= 1:
            # Migration 1: Add original columns
            conn.execute("ALTER TABLE jobs ADD COLUMN some_column TEXT")
            conn.execute(
                "INSERT INTO schema_version VALUES (1, datetime('now'), 'Initial schema')"
            )

        if from_v < 2 and to_v >= 2:
            # Migration 2: Add Mulliken columns
            conn.execute("ALTER TABLE jobs ADD COLUMN mulliken_charges TEXT")
            conn.execute("ALTER TABLE jobs ADD COLUMN mulliken_spins TEXT")
            conn.execute("ALTER TABLE jobs ADD COLUMN loewdin_charges TEXT")
            conn.execute("ALTER TABLE jobs ADD COLUMN loewdin_spins TEXT")
            conn.execute(
                "INSERT INTO schema_version VALUES (2, datetime('now'), 'Add Mulliken analysis')"
            )

def migrate_database(conn: sqlite3.Connection) -> None:
    """Migrate database to current schema version."""
    init_schema_version(conn)
    current_version = get_schema_version(conn)

    if current_version < SCHEMA_VERSION:
        print(f"Migrating database from v{current_version} to v{SCHEMA_VERSION}")
        apply_migration(conn, current_version, SCHEMA_VERSION)
        print("Migration complete")
    elif current_version > SCHEMA_VERSION:
        raise ValueError(
            f"Database schema v{current_version} is newer than code v{SCHEMA_VERSION}. "
            "Please update the code."
        )
```

**Pros:**
- Industry standard pattern
- Clear migration history
- Detects version mismatches
- Supports upgrades and checks
- Auditable changes
- Safe for concurrent access (with locking)

**Cons:**
- Need to write migrations explicitly
- More code to maintain
- Backward compatibility considerations

**Effort:** 4-5 hours

**Risk:** Low

---

### Option 2: Use Alembic (ORM Migration Tool)

**Approach:** Adopt professional migration framework

**Pros:**
- Auto-generate migrations
- Rollback support
- Battle-tested
- Rich feature set

**Cons:**
- Heavy dependency
- Requires SQLAlchemy
- Overkill for simple needs
- Learning curve

**Effort:** 1-2 days

**Risk:** High (major refactor)

---

### Option 3: Column Check Before ALTER

**Approach:** Query sqlite_master to check columns

```python
def column_exists(conn, table, column):
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns

if not column_exists(conn, 'jobs', 'mulliken_charges'):
    conn.execute("ALTER TABLE jobs ADD COLUMN mulliken_charges TEXT")
```

**Pros:**
- Simple to implement
- No new tables needed

**Cons:**
- No version tracking
- No migration history
- Still fragile
- Doesn't solve core problem

**Effort:** 1 hour

**Risk:** Medium (incomplete solution)

## Recommended Action

**To be filled during triage.**

## Technical Details

**Schema version table:**
```sql
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,     -- ISO timestamp
    description TEXT               -- Human-readable description
);
```

**Migration workflow:**
1. Code defines target schema version (e.g., v2)
2. On database open, check current version
3. If behind, apply migrations sequentially
4. If ahead, raise error (code too old)
5. Record each migration in schema_version table

**Example migrations:**
- v0 → v1: Initial schema
- v1 → v2: Add Mulliken columns
- v2 → v3: Add indexes (future)

**Concurrent access:**
- Use file locking for schema changes
- Check version before every migration
- Use transactions to ensure atomicity

## Resources

- **PR:** feature/mulliken-population-analysis
- **Code review:** data-integrity-guardian agent finding
- **Django migrations:** https://docs.djangoproject.com/en/stable/topics/migrations/
- **Alembic:** https://alembic.sqlalchemy.org/
- **SQLite ALTER TABLE:** https://www.sqlite.org/lang_altertable.html

## Acceptance Criteria

- [ ] schema_version table created
- [ ] Current version queryable
- [ ] Migrations applied automatically on database open
- [ ] Version mismatch detected (code too old)
- [ ] Each migration recorded with timestamp
- [ ] Migrations are idempotent
- [ ] Transaction safety for migrations
- [ ] Documentation for adding new migrations
- [ ] Tests for migration scenarios

## Work Log

### 2026-02-16 - Initial Discovery

**By:** Claude Code (data-integrity-guardian agent)

**Actions:**
- Identified missing schema versioning
- Analyzed current ALTER TABLE approach
- Researched migration best practices
- Designed lightweight migration system
- Drafted implementation with examples

**Learnings:**
- Schema versioning is critical for production databases
- Try/except on ALTER TABLE hides real errors
- Standard pattern: version table + sequential migrations
- Must be transactional for safety
- Should detect version mismatches

## Notes

- **CRITICAL** for production deployment
- Prevents schema drift disasters
- Standard practice in all professional database applications
- Document migration process in developer guide
- Consider adding migration tests to CI

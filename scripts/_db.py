"""
Database connection helper for scripts.

ABE P0: Never hardcode credentials. Load from environment.

Usage:
    from _db import get_database_url, get_db_connection

    # Option 1: Get URL for SQLAlchemy
    url = get_database_url()

    # Option 2: Get psycopg2 connection directly
    conn = get_db_connection()
"""

import os
import sys


def get_database_url() -> str:
    """
    Get DATABASE_URL from environment.

    Raises SystemExit if not configured.
    """
    url = os.environ.get("DATABASE_URL")

    if not url:
        print("ERROR: DATABASE_URL not set in environment.", file=sys.stderr)
        print("Set it with: export DATABASE_URL='postgresql://...'", file=sys.stderr)
        sys.exit(1)

    return url


def get_db_connection():
    """
    Get a psycopg2 connection using DATABASE_URL from environment.

    Returns:
        psycopg2 connection object

    Raises:
        SystemExit if DATABASE_URL not set or psycopg2 not installed
    """
    url = get_database_url()

    try:
        import psycopg2
        return psycopg2.connect(url)
    except ImportError:
        print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary", file=sys.stderr)
        sys.exit(1)

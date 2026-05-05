from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Tuple

import psycopg2
import psycopg2.extensions

from .select_config import DatabaseLogin


@contextmanager
def get_connection(
    login: DatabaseLogin,
) -> Generator[
    Tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor], None, None
]:
    """Context manager that yields (connection, cursor) and closes both on exit."""
    conn = psycopg2.connect(
        host=login.host,
        port=login.port,
        dbname=login.dbname,
        user=login.user,
        password=login.password,
    )
    cur = conn.cursor()
    try:
        yield conn, cur
    finally:
        cur.close()
        conn.close()

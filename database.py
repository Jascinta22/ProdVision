"""
database.py
Initializes the SQLite database connection and creates tables if they
do not already exist.
"""

import sqlite3
import os
from contextlib import contextmanager

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db")
DB_PATH = os.path.join(DB_DIR, "prodvision.db")

os.makedirs(DB_DIR, exist_ok=True)


def init_db():
    """Create the database file and required tables if they don't exist. Migrate old schema if needed."""
    with get_connection() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                image_side TEXT,
                uploaded_at TEXT DEFAULT (datetime('now')),
                raw_text TEXT,
                ocr_confidence REAL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id INTEGER NOT NULL,
                brand TEXT,
                product_name TEXT,
                weight TEXT,
                mrp TEXT,
                mfg_date TEXT,
                exp_date TEXT,
                confidence_score REAL,
                json_path TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (upload_id) REFERENCES uploads (id)
            )
            """
        )

        # Migration: add batch_id and image_side columns if they don't exist
        try:
            cur.execute("PRAGMA table_info(uploads)")
            columns = [col[1] for col in cur.fetchall()]
            
            if "batch_id" not in columns:
                cur.execute("ALTER TABLE uploads ADD COLUMN batch_id TEXT")
                print("[DB] Added batch_id column to uploads table")
            
            if "image_side" not in columns:
                cur.execute("ALTER TABLE uploads ADD COLUMN image_side TEXT")
                print("[DB] Added image_side column to uploads table")
        except Exception as e:
            print(f"[DB] Migration warning: {e}")

        conn.commit()


@contextmanager
def get_connection():
    """Context manager yielding a sqlite3 connection with row factory set."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

"""
models.py
Defines operations to insert uploads, insert extracted product attributes,
retrieve history, and fetch details for a single product/upload.
"""

from database import get_connection


def insert_upload(batch_id: str, filename: str, filepath: str, image_side: str, raw_text: str, ocr_confidence: float) -> int:
    """Insert a new upload record and return its id."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO uploads (batch_id, filename, filepath, image_side, raw_text, ocr_confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (batch_id, filename, filepath, image_side, raw_text, ocr_confidence),
        )
        conn.commit()
        return cur.lastrowid


def insert_product(upload_id: int, attributes: dict, confidence_score: float, json_path: str) -> int:
    """Insert a new extracted-product record and return its id."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO products
                (upload_id, brand, product_name, weight, mrp, mfg_date, exp_date,
                 confidence_score, json_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                upload_id,
                attributes.get("brand"),
                attributes.get("product_name"),
                attributes.get("weight"),
                attributes.get("mrp"),
                attributes.get("mfg_date"),
                attributes.get("exp_date"),
                confidence_score,
                json_path,
            ),
        )
        conn.commit()
        return cur.lastrowid


def get_all_products() -> list:
    """Return all products joined with their upload info, most recent first."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT p.*, u.filename, u.filepath, u.uploaded_at
            FROM products p
            JOIN uploads u ON p.upload_id = u.id
            ORDER BY p.id DESC
            """
        )
        rows = cur.fetchall()
        return [dict(row) for row in rows]


def get_product_by_id(product_id: int):
    """Return a single product (joined with upload info) by its id, or None."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT p.*, u.filename, u.filepath, u.uploaded_at, u.raw_text
            FROM products p
            JOIN uploads u ON p.upload_id = u.id
            WHERE p.id = ?
            """,
            (product_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def delete_product(product_id: int) -> bool:
    """Delete a product record and its associated upload. Returns True if successful."""
    with get_connection() as conn:
        cur = conn.cursor()
        
        # First get the upload_id
        cur.execute("SELECT upload_id FROM products WHERE id = ?", (product_id,))
        row = cur.fetchone()
        if not row:
            return False
        
        upload_id = row[0]
        
        # Delete product record
        cur.execute("DELETE FROM products WHERE id = ?", (product_id,))
        
        # Delete upload record
        cur.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
        
        conn.commit()
        return True


def delete_all_products() -> int:
    """Delete all product and upload records. Returns count of deleted products."""
    with get_connection() as conn:
        cur = conn.cursor()
        
        # Get count before deletion
        cur.execute("SELECT COUNT(*) FROM products")
        count = cur.fetchone()[0]
        
        # Delete all products and uploads
        cur.execute("DELETE FROM products")
        cur.execute("DELETE FROM uploads")
        
        conn.commit()
        return count

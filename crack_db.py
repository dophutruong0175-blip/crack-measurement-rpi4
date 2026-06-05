# crack_db.py
"""SQLite history manager cho hệ thống đo vết nứt.

Public API:
    init_db(db_path)
    save_measurement(payload, overlay_img, db_path) -> int
    get_history(limit, db_path) -> list[dict]
    get_overlay_jpg(record_id, db_path) -> bytes | None
    export_csv(db_path) -> str
"""
import csv
import io
import sqlite3
import threading
from typing import Optional

import cv2
import numpy as np

MAX_RECORDS: int = 50
_LOCK = threading.Lock()

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS measurements (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at   TEXT    NOT NULL,
    image_name    TEXT    NOT NULL,
    length_mm     REAL    NOT NULL,
    max_width_mm  REAL    NOT NULL,
    area_mm2      REAL    NOT NULL,
    segment_count INTEGER NOT NULL,
    pixels_per_mm REAL    NOT NULL,
    overlay_jpg   BLOB
);
"""

_CSV_COLUMNS = [
    "id", "captured_at", "image_name",
    "length_mm", "max_width_mm", "area_mm2",
    "segment_count", "pixels_per_mm",
]


def _connect(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path, check_same_thread=False)


def init_db(db_path: str = "measurements.db") -> None:
    """Tạo bảng measurements nếu chưa có."""
    with _LOCK:
        conn = _connect(db_path)
        try:
            conn.execute(_CREATE_SQL)
            conn.commit()
        finally:
            conn.close()


def save_measurement(
    payload: dict,
    overlay_img: Optional[np.ndarray],
    db_path: str = "measurements.db",
) -> int:
    """Lưu 1 lần đo. Auto-purge nếu vượt MAX_RECORDS (50).

    Args:
        payload: dict chứa các trường đo
        overlay_img: numpy BGR image để lưu dưới dạng JPEG BLOB
        db_path: đường dẫn file SQLite

    Returns:
        id của record vừa lưu
    """
    if overlay_img is not None and isinstance(overlay_img, np.ndarray) and overlay_img.size > 0:
        ok, buf = cv2.imencode(".jpg", overlay_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        overlay_bytes = buf.tobytes() if ok else None
    else:
        overlay_bytes = None

    with _LOCK:
        conn = _connect(db_path)
        try:
            cursor = conn.execute(
                """INSERT INTO measurements
                   (captured_at, image_name, length_mm, max_width_mm,
                    area_mm2, segment_count, pixels_per_mm, overlay_jpg)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(payload.get("processed_at", "")),
                    str(payload.get("image_name", "")),
                    float(payload.get("total_length_mm", 0.0)),
                    float(payload.get("max_width_mm", 0.0)),
                    float(payload.get("area_mm2", 0.0)),
                    int(payload.get("segment_count", 0)),
                    float(payload.get("pixels_per_mm", 0.0)),
                    overlay_bytes,
                ),
            )
            record_id: int = cursor.lastrowid  # type: ignore[assignment]
            # Auto-purge: xóa record cũ nhất nếu vượt giới hạn
            conn.execute(
                """DELETE FROM measurements
                   WHERE id = (SELECT MIN(id) FROM measurements)
                     AND (SELECT COUNT(*) FROM measurements) > ?""",
                (MAX_RECORDS,),
            )
            conn.commit()
        finally:
            conn.close()
    return record_id


def get_history(
    limit: int = 50,
    db_path: str = "measurements.db",
) -> list:
    """Trả về list dict các lần đo gần nhất (không kèm BLOB ảnh).

    Sắp xếp: mới nhất trước (ORDER BY id DESC).
    """
    with _LOCK:
        conn = _connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                f"""SELECT {', '.join(_CSV_COLUMNS)}
                    FROM measurements
                    ORDER BY id DESC
                    LIMIT ?""",
                (limit,),
            )
            rows = [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    return rows


def get_overlay_jpg(
    record_id: int,
    db_path: str = "measurements.db",
) -> Optional[bytes]:
    """Trả về JPEG bytes của overlay image theo record_id, hoặc None."""
    with _LOCK:
        conn = _connect(db_path)
        try:
            cursor = conn.execute(
                "SELECT overlay_jpg FROM measurements WHERE id = ?",
                (record_id,),
            )
            row = cursor.fetchone()
        finally:
            conn.close()
    if row is None or row[0] is None:
        return None
    return bytes(row[0])


def export_csv(db_path: str = "measurements.db") -> str:
    """Trả về CSV string của toàn bộ history (không kèm ảnh).

    Columns: id, captured_at, image_name, length_mm, max_width_mm,
             area_mm2, segment_count, pixels_per_mm
    """
    with _LOCK:
        conn = _connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                f"""SELECT {', '.join(_CSV_COLUMNS)}
                    FROM measurements
                    ORDER BY id ASC"""
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(_CSV_COLUMNS)
    for row in rows:
        writer.writerow(list(row))
    return output.getvalue()

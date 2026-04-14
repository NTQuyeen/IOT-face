from __future__ import annotations

from datetime import datetime, date as date_type
from utils.db import get_db


def init_db():
    db = get_db()
    cur = db.cursor()

    # NEW: attendance sessions (multi checkin/checkout per day)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance_sessions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            date DATE NOT NULL,
            checkin DATETIME NULL,
            checkout DATETIME NULL,
            source VARCHAR(16) NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_date_name (date, name),
            INDEX idx_name_date (name, date)
        )
    """)

    # rfid users
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rfid_users (
            uid VARCHAR(32) PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
    """)

    db.commit()
    cur.close()
    db.close()
    print("✅ Tables ready: attendance_sessions, rfid_users")


def mark_attendance(name: str, action: str, when: datetime | None = None, source: str | None = None):
    """
    action: "checkin" | "checkout"
    - checkin: luôn tạo 1 session mới
    - checkout: đóng session checkin gần nhất chưa checkout; nếu không có session mở thì vẫn lưu 1 dòng checkout-only
    """
    if when is None:
        when = datetime.now()

    today: date_type = when.date()

    db = get_db()
    cur = db.cursor()

    if action == "checkin":
        cur.execute(
            """
            INSERT INTO attendance_sessions (name, date, checkin, checkout, source)
            VALUES (%s, %s, %s, NULL, %s)
            """,
            (name, today, when, source),
        )

    elif action == "checkout":
        cur.execute(
            """
            SELECT id
            FROM attendance_sessions
            WHERE name=%s AND date=%s AND checkout IS NULL AND checkin IS NOT NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (name, today),
        )
        row = cur.fetchone()

        if row:
            session_id = row[0]
            cur.execute(
                "UPDATE attendance_sessions SET checkout=%s, source=COALESCE(source, %s) WHERE id=%s",
                (when, source, session_id),
            )
        else:
            # nếu checkout mà không có checkin mở, vẫn lưu lịch sử (giống hành vi cũ)
            cur.execute(
                """
                INSERT INTO attendance_sessions (name, date, checkin, checkout, source)
                VALUES (%s, %s, NULL, %s, %s)
                """,
                (name, today, when, source),
            )
    else:
        cur.close()
        db.close()
        raise ValueError("action must be 'checkin' or 'checkout'")

    db.commit()
    cur.close()
    db.close()


def get_sessions_by_date(date_str: str):
    """
    Return rows: (id, name, date, checkin, checkout, source)
    """
    db = get_db()
    cur = db.cursor()

    cur.execute(
        """
        SELECT id, name, date, checkin, checkout, source
        FROM attendance_sessions
        WHERE date=%s
        ORDER BY COALESCE(checkin, checkout) DESC
        """,
        (date_str,),
    )
    records = cur.fetchall()
    cur.close()
    db.close()
    return records


def get_totals_by_date(date_str: str):
    """
    Return rows: (name, total_seconds, sessions_done)
    total_seconds = sum(checkout - checkin) cho các session đủ cặp
    """
    db = get_db()
    cur = db.cursor()

    cur.execute(
        """
        SELECT
          name,
          COALESCE(SUM(TIMESTAMPDIFF(SECOND, checkin, checkout)), 0) AS total_seconds,
          SUM(CASE WHEN checkin IS NOT NULL AND checkout IS NOT NULL THEN 1 ELSE 0 END) AS sessions_done
        FROM attendance_sessions
        WHERE date=%s
        GROUP BY name
        ORDER BY total_seconds DESC
        """,
        (date_str,),
    )
    rows = cur.fetchall()
    cur.close()
    db.close()
    return rows


def format_seconds(total_seconds: int) -> str:
    total_seconds = int(total_seconds or 0)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
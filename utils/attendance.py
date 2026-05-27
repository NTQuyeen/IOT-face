from __future__ import annotations

from datetime import datetime, date as date_type
from utils.db import get_db
from datetime import time as time_type


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

def _t(hms: str) -> time_type:
    hh, mm, *rest = [int(x) for x in hms.split(":")]
    ss = rest[0] if rest else 0
    return time_type(hh, mm, ss)

def _overlap_seconds(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> int:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if end <= start:
        return 0
    return int((end - start).total_seconds())

def split_shift_overtime_seconds(
    checkin: datetime,
    checkout: datetime,
    shift_start: str = "08:00:00",
    shift_end: str = "17:00:00",
) -> tuple[int, int, int]:
    """
    Return: (shift_seconds, overtime_seconds, total_seconds)
    """
    if not checkin or not checkout or checkout <= checkin:
        return 0, 0, 0

    total = int((checkout - checkin).total_seconds())
    d = checkin.date()

    ss = datetime.combine(d, _t(shift_start))
    se = datetime.combine(d, _t(shift_end))

    # phần nằm trong ca
    shift_sec = _overlap_seconds(checkin, checkout, ss, se)
    # phần ngoài ca = tổng - trong ca
    overtime_sec = max(0, total - shift_sec)
    return shift_sec, overtime_sec, total

def get_shift_overtime_totals_by_date(
    date_str: str,
    shift_start: str = "08:00:00",
    shift_end: str = "17:00:00",
):
    """
    Return rows: (name, shift_seconds, overtime_seconds, total_seconds, sessions_done)
    Chỉ tính các session đủ cặp checkin+checkout.
    """
    db = get_db()
    cur = db.cursor()
    cur.execute(
        """
        SELECT name, checkin, checkout
        FROM attendance_sessions
        WHERE date=%s AND checkin IS NOT NULL AND checkout IS NOT NULL
        """,
        (date_str,),
    )
    rows = cur.fetchall()
    cur.close()
    db.close()

    agg = {}  # name -> [shift, ot, total, sessions]
    for (name, checkin, checkout) in rows:
        shift_sec, ot_sec, total_sec = split_shift_overtime_seconds(
            checkin, checkout, shift_start=shift_start, shift_end=shift_end
        )
        if name not in agg:
            agg[name] = [0, 0, 0, 0]
        agg[name][0] += shift_sec
        agg[name][1] += ot_sec
        agg[name][2] += total_sec
        agg[name][3] += 1

    result = [(n, v[0], v[1], v[2], v[3]) for (n, v) in agg.items()]
    result.sort(key=lambda x: x[3], reverse=True)
    return result

def get_shift_overtime_totals_by_range(
    start_date: str,  # "YYYY-MM-DD" (inclusive)
    end_date: str,    # "YYYY-MM-DD" (exclusive)
    shift_start: str = "08:00:00",
    shift_end: str = "17:00:00",
):
    """
    Return rows: (date, name, shift_seconds, overtime_seconds, total_seconds, sessions_done)
    """
    db = get_db()
    cur = db.cursor()
    cur.execute(
        """
        SELECT date, name, checkin, checkout
        FROM attendance_sessions
        WHERE date >= %s AND date < %s
          AND checkin IS NOT NULL AND checkout IS NOT NULL
        """,
        (start_date, end_date),
    )
    rows = cur.fetchall()
    cur.close()
    db.close()

    agg = {}  # (date,name) -> [shift, ot, total, sessions]
    for (d, name, checkin, checkout) in rows:
        shift_sec, ot_sec, total_sec = split_shift_overtime_seconds(
            checkin, checkout, shift_start=shift_start, shift_end=shift_end
        )
        key = (str(d), name)
        if key not in agg:
            agg[key] = [0, 0, 0, 0]
        agg[key][0] += shift_sec
        agg[key][1] += ot_sec
        agg[key][2] += total_sec
        agg[key][3] += 1

    result = [(d, n, v[0], v[1], v[2], v[3]) for ((d, n), v) in agg.items()]
    result.sort(key=lambda x: (x[0], x[1]))
    return result

def _month_range(month: str) -> tuple[str, str]:
    # "YYYY-MM" -> [start, end_exclusive]
    y, m = month.split("-")
    y = int(y); m = int(m)
    start = datetime(y, m, 1).date()
    if m == 12:
        end = datetime(y + 1, 1, 1).date()
    else:
        end = datetime(y, m + 1, 1).date()
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
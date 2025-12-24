# utils/attendance.py
from datetime import datetime
from utils.db import get_db


def init_db():
    """
    Tạo bảng attendance nếu chưa tồn tại
    Gọi 1 lần khi server khởi động
    """
    db = get_db()
    cur = db.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100),
            time DATETIME,
            date DATE
        )
    """)

    db.commit()
    cur.close()
    db.close()
    print("✅ Attendance table ready")


def mark_attendance(name):
    db = get_db()
    cur = db.cursor()

    now = datetime.now()
    today = now.date()

    # ❗ chống điểm danh trùng trong cùng ngày
    cur.execute(
        "SELECT id FROM attendance WHERE name=%s AND date=%s",
        (name, today)
    )

    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO attendance (name, time, date) VALUES (%s, %s, %s)",
            (name, now, today)
        )
        db.commit()
        print(f"✅ Đã lưu điểm danh: {name}")

    cur.close()
    db.close()


def get_all_attendance():
    db = get_db()
    cur = db.cursor()

    cur.execute(
        "SELECT id, name, time, date FROM attendance ORDER BY time DESC"
    )
    records = cur.fetchall()

    cur.close()
    db.close()
    return records


def get_attendance_by_date(date):
    db = get_db()
    cur = db.cursor()

    cur.execute(
        "SELECT id, name, time, date FROM attendance WHERE date=%s ORDER BY time DESC",
        (date,)
    )

    records = cur.fetchall()
    cur.close()
    db.close()
    return records

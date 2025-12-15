# utils/attendance.py
import os
import pandas as pd
from datetime import datetime

ATTENDANCE_FILE = "diem_danh.xlsx"

def mark_attendance(name: str):
    """
    Ghi điểm danh cho một người vào file Excel.
    Nếu file chưa tồn tại, tạo mới với các cột: Tên, Thời gian, Ngày
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Đọc file hiện tại (nếu có)
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_excel(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Tên", "Ngày", "Thời gian"])

    # Kiểm tra xem hôm nay người này đã điểm danh chưa
    today_records = df[(df["Tên"] == name) & (df["Ngày"] == date_str)]
    if not today_records.empty:
        # Đã điểm danh hôm nay → bỏ qua
        return

    # Thêm bản ghi mới
    new_row = pd.DataFrame({
        "Tên": [name],
        "Ngày": [date_str],
        "Thời gian": [time_str]
    })

    df = pd.concat([df, new_row], ignore_index=True)

    # Lưu lại file
    df.to_excel(ATTENDANCE_FILE, index=False)
    print(f"Đã điểm danh: {name} - {date_str} {time_str}")
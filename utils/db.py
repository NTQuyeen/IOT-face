import mysql.connector

def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="123456",   # nếu root không có mật khẩu
        database="face_attendance"
    )

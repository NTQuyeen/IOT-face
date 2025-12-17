print("🚀 Creating admin...")

from utils.db import get_db

db = get_db()
cur = db.cursor()

cur.execute(
    "INSERT INTO admin (username, password) VALUES (%s,%s)",
    ("admin", "123456")
)

db.commit()
cur.close()
db.close()

print("✅ Admin created")

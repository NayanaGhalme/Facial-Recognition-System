import sqlite3

def init_db():
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        phone TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        is_verified INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Lost persons table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS lost_persons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        name TEXT,
        age INTEGER,
        photo_path TEXT,
        embedding BLOB,
        is_active INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # Found persons table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS found_persons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lost_person_id INTEGER,
        confidence REAL,
        image_path TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        alert_sent INTEGER DEFAULT 0,
        FOREIGN KEY(lost_person_id) REFERENCES lost_persons(id)
    )
    """)

    conn.commit()
    conn.close()
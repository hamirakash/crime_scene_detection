
import sqlite3
from pathlib import Path

# Database will be saved in the project folder
DATABASE_PATH = Path(__file__).resolve().parent / "cases.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    with get_connection() as connection:

        # Table 1: Store investigation cases
        connection.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_name TEXT NOT NULL,
                image_filename TEXT NOT NULL,
                reviewed_image_filename TEXT,
                total_objects INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'Pending',
                created_at TEXT NOT NULL
                    DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table 2: Store detected evidence
        connection.execute("""
            CREATE TABLE IF NOT EXISTS evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER NOT NULL,
                evidence_id TEXT NOT NULL,
                object_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                bounding_box TEXT NOT NULL,
                selected INTEGER NOT NULL DEFAULT 0,

                FOREIGN KEY (case_id)
                    REFERENCES cases(id)
            )
        """)

        connection.commit()


if __name__ == "__main__":
    initialize_database()
    print("Case database created successfully!")

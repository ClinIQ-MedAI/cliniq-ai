import sqlite3
import re
from config import Config

def _get_chat_db_connection():
    """Create a SQLite connection for chat history operations."""
    conn = sqlite3.connect(Config.CHAT_DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_chat_db():
    """Initialize chat history database if missing."""
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _get_chat_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_messages_patient_id_id
            ON chat_messages (patient_id, id)
            """
        )
        conn.commit()

def sanitize_patient_id(patient_id: str) -> str:
    """Sanitize patient ID to prevent any injection or strange chars."""
    if not patient_id:
        return "anonymous"
    return re.sub(r'[^a-zA-Z0-9_-]', '', str(patient_id))[:100] or "anonymous"

def save_chat_message(patient_id: str, role: str, content: str):
    """Persist one chat message for a patient."""
    patient_id = sanitize_patient_id(patient_id)
    if not content:
        return

    with _get_chat_db_connection() as conn:
        conn.execute(
            "INSERT INTO chat_messages (patient_id, role, content) VALUES (?, ?, ?)",
            (patient_id, role, content),
        )
        conn.commit()

def get_chat_messages(patient_id: str, limit: int = None):
    """Load chat history for a patient in chronological order."""
    patient_id = sanitize_patient_id(patient_id)

    with _get_chat_db_connection() as conn:
        if isinstance(limit, int) and limit > 0:
            rows = conn.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE patient_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (patient_id, limit),
            ).fetchall()
            rows = list(reversed(rows))
        else:
            rows = conn.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE patient_id = ?
                ORDER BY id ASC
                """,
                (patient_id,),
            ).fetchall()

    return [{"role": row["role"], "content": row["content"]} for row in rows]

def clear_chat_messages(patient_id: str):
    """Delete all messages for one patient chat."""
    patient_id = sanitize_patient_id(patient_id)

    with _get_chat_db_connection() as conn:
        conn.execute("DELETE FROM chat_messages WHERE patient_id = ?", (patient_id,))
        conn.commit()

def trim_chat_messages(patient_id: str, max_messages: int = Config.MAX_HISTORY_MESSAGES):
    """Keep only the latest N messages for a patient."""
    patient_id = sanitize_patient_id(patient_id)
    if not isinstance(max_messages, int) or max_messages <= 0:
        return

    with _get_chat_db_connection() as conn:
        conn.execute(
            """
            DELETE FROM chat_messages
            WHERE patient_id = ?
              AND id NOT IN (
                SELECT id
                FROM chat_messages
                WHERE patient_id = ?
                ORDER BY id DESC
                LIMIT ?
              )
            """,
            (patient_id, patient_id, max_messages),
        )
        conn.commit()

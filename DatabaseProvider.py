import sqlite3
import hashlib

class DatabaseProvider:
    
    def __init__(self, database_path: str = "memory.db"):
        self.database_path = database_path
        self._initialize_database()
        
    def _get_conn(self) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        return (conn, cursor)
        
    
    def _initialize_database(self):
        
        conn, cursor = self._get_conn()
        cursor.execute("""
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY,
    question_text TEXT UNIQUE,
    question_type TEXT
)
                       """)
        
        cursor.execute("""
CREATE TABLE IF NOT EXISTS answers (
    id INTEGER PRIMARY KEY,
    question_id INTEGER,
    answer_text TEXT,
    answer_reputation REAL DEFAULT 1,
    FOREIGN KEY(question_id) REFERENCES questions(id)
)
                       """)
        
        conn.commit()
        conn.close()
    
    def compute_question_hash(self, question_text: str, question_answers: list[str]):
        question_hashtext = f"{question_text}+{"//".join(question_answers)}"
        print(f"hashtext: {question_hashtext}")
        question_hash = hashlib.sha256(question_hashtext.encode()).hexdigest()
        return question_hash
    
    def store_question_answer(self, answer_text: str, question_type: str, question_hash: str, new: bool):
        print(f"""
store question:
  answer_text = {answer_text}
  question_type = {question_type}
  question_hash = {question_hash}
              """)
        conn, cursor = self._get_conn()
        
        cursor.execute(
            "INSERT OR IGNORE INTO questions (question_text, question_type) VALUES (?, ?)",
            (question_hash, question_type)
        )
        # Get question ID
        cursor.execute("""
            SELECT id FROM questions WHERE question_text = ?
                       """, (question_hash,))
        
        question_id = cursor.fetchone()[0]
        
        # Check if the answer already exists
        cursor.execute("""
SELECT id, answer_reputation FROM answers WHERE answer_text = ?
                       """, (answer_text,))
        
        row = cursor.fetchone()
        if row:
            print(f"answer already exists")
            id, reputation = row
            cursor.execute(
                """
                UPDATE answers SET answer_reputation = ? WHERE id = ?
                """,
                (reputation + 1, id)
            )
        else:
            print(f"answer does not exist, adding to database")
            cursor.execute(
                """
INSERT INTO answers (question_id, answer_text, answer_reputation) VALUES (?, ?, ?)
                """,
                (question_id, answer_text, 1)
                
            )
        
        conn.commit()
        conn.close()
        
    def lookup_answer_reputation(self, answer_text: str) -> float:
        print(f"lookup answer reputation where answer_text = {answer_text}")
        conn, cursor = self._get_conn()
        
        cursor.execute("""
SELECT answer_reputation FROM answers WHERE answer_text = ?
                       """, (answer_text,))
        
        row = cursor.fetchone()
        conn.close()
        
        return float(row[0]) if row else 0.0
    
    def lookup_question_id(self, question_hash: str) -> int:
        print(f"lookup question id where question_hash = {question_hash}")
        conn, cursor = self._get_conn()
        
        cursor.execute("SELECT id FROM questions WHERE question_text = ?", (question_hash,))
        
        row = cursor.fetchone()
        conn.close()
        
        return row[0] if row else -1

    def lookup_answer(self, question_id: int) -> str:
        print(f"lookup answer where question_id = {question_id}")
        conn, cursor = self._get_conn()
        
        cursor.execute("SELECT answer_text FROM answers WHERE question_id = ?", (question_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        return row[0] if row else ""
        
        
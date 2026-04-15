import sqlite3
import hashlib

from question_type import QuestionType
from similarity_provider import SimilarityProvider


class DatabaseProvider:

    def __init__(
        self,
        similarity_provider: "SimilarityProvider",
        database_path: str = "memory.db",
    ):
        self.database_path = database_path

        self.current_list: int = -1

        self.similarity_provider = similarity_provider

        self._initialize_database()
        self._apply_migrations()
        
        self.applied_verifications: set[int] = set()

    def _get_conn(self) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        return (conn, cursor)

    def _initialize_database(self):

        conn, cursor = self._get_conn()

        cursor.execute("""
CREATE TABLE IF NOT EXISTS lists (
    id INTEGER PRIMARY KEY,
    questionsCount INTEGER
)
                       """)

        cursor.execute("""
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY,
    list_id INTEGER,
    question_text TEXT UNIQUE,
    question_type TEXT
)
                       """)

        cursor.execute("""
CREATE TABLE IF NOT EXISTS answers (
    id INTEGER PRIMARY KEY,
    list_id INTEGER,
    question_id INTEGER,
    answer_text TEXT,
    answer_reputation REAL DEFAULT 1,
    FOREIGN KEY(question_id) REFERENCES questions(id)
)
                       """)

        cursor.execute("""
CREATE TABLE IF NOT EXISTS questionData (
    id INTEGER PRIMARY KEY,
    list_id INTEGER,
    question_hash TEXT,
    question_text TEXT,
    contextual_sentence TEXT,
    target_word TEXT,
    question_type INTEGER,
    answer_string_1 TEXT,
    answer_string_2 TEXT,
    answer_string_3 TEXT,
    answer_string_4 TEXT,
    correct_answer_index INTEGER,
    answer_embedding BLOB,
    seen INTEGER
)
                       """)

        conn.commit()
        conn.close()

    def _faulty_data_correction(self):
        
        assert self.current_list != -1
        
        conn, cursor = self._get_conn()
        
        # scan for correct answers
        
        cursor.execute("SELECT answer_string_1, answer_string_2, answer_string_3, answer_string_4, correct_answer_index, question_hash FROM questionData WHERE list_id = ? AND target_word=''", (self.current_list,))
        
        correct_answers: set[str] = set([])
        
        rows = cursor.fetchall()
        for row in rows:
            answer0, answer1, answer2, answer3, answer_index, qhash = row
            answers: list[str] = [answer0, answer1, answer2, answer3]
            
            answer: str = answers[answer_index]
            correct_answers.add(answer)

        
        # scan for answer reputations
        
        cursor.execute("SELECT id, answer_text, question_id FROM answers WHERE list_id = ?", (self.current_list,))
        rows = cursor.fetchall()
        
        
        # answer_text, row_id, question_id
        potential_faulty_answers: list[tuple[str, int, int]] = []
        
        for row in rows:
            row_id, answer_text, question_id = row
            if answer_text not in correct_answers:
                print(f"-- [internal] potential faulty answer: {answer_text} (row id: {row_id})")
                potential_faulty_answers.append((answer_text, row_id, question_id))
        
        # scan for non-existent question IDs
        
        cursor.execute("SELECT id, question_text FROM questions WHERE list_id = ?", (self.current_list,))
        
        existent_question_ids: set[int] = set()
        
        for row in cursor.fetchall():
            question_id, _ = row
            existent_question_ids.add(question_id)
        
        
        
        
        for row in potential_faulty_answers:
            answer_text, row_id, question_id = row
            if question_id not in existent_question_ids:
                print(f"-- Faulty answer: {answer_text} in {row_id} linked with non-existent QID: {question_id}")
        
        conn.close()
    
    def _apply_fixes(self):
        assert self.current_list != -1
        
        if self.current_list in self.applied_verifications:
            return
        print("Apply: faulty reputation correction")
        self._faulty_data_correction()
        print("Fixes applied")
        
        self.applied_verifications.add(self.current_list)

    def _apply_migrations(self):
        """
        Apply migrations to the database. Should run on initialize.
        """

        conn, cursor = self._get_conn()

        # fetch existing columns
        table_name = "answers"
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = [row[1] for row in cursor.fetchall()]

        # -- APPLY MIGRATION: NEW COLUMN: TARGET_WORD IN 'ANSWERS' TABLE

        column_name = "target_word"
        if column_name not in existing_columns:
            cursor.execute(
                f'ALTER TABLE {table_name} ADD COLUMN {column_name} TEXT DEFAULT ""'
            )

        conn.commit()
        conn.close()

    def create_list(self, list_id: int):

        conn, cursor = self._get_conn()
        print(f"Create list ID: {list_id}")
        cursor.execute(
            """
INSERT OR IGNORE INTO lists (id, questionsCount)
VALUES (?, ?)
                       """,
            (list_id, 0),
        )
        conn.commit()
        conn.close()

    def list_exists(self, list_id: int) -> bool:
        conn, cursor = self._get_conn()

        cursor.execute("SELECT 1 FROM lists WHERE id = ? LIMIT 1", (list_id,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def switch_to_list(self, list_id: int):

        print(f"Switch to list ID: {list_id}")
        if not self.list_exists(list_id):
            self.create_list(list_id)
        self.current_list = list_id
        
        self._apply_fixes()

    @staticmethod
    def clean_string(s: str) -> str:
        return (
            s.strip().replace("\n", " ").replace("\t", " ").replace("\r", " ").strip()
        )

    def check_question_data_exists(self, question_hash: str):

        assert self.current_list != -1

        conn, cursor = self._get_conn()
        cursor.execute(
            "SELECT 1 FROM questionData WHERE list_id = ? AND question_hash = ?",
            (self.current_list, question_hash),
        )
        exists = cursor.fetchone() is not None
        conn.close()

        return exists

    def add_question_data(
        self,
        question_hash_text: str,
        question_text: str,
        contextual_sentence: str,
        word: str,
        question_type: QuestionType,
        answers: list[str],
        correct_answer_index: int,
    ):

        assert self.current_list != -1
        assert len(answers) <= 4
        assert correct_answer_index < len(answers)
        assert correct_answer_index >= 0

        original_answer = answers[correct_answer_index]

        # sort by alphabetical
        answers.sort()

        # new index
        correct_answer_index = answers.index(original_answer)

        # add 1
        conn, cursor = self._get_conn()
        cursor.execute(
            "SELECT questionsCount FROM lists WHERE id = ?", (self.current_list,)
        )
        questions_n_current = cursor.fetchone()[0]
        cursor.execute(
            "UPDATE lists SET questionsCount = ? WHERE id = ?",
            (questions_n_current + 1, self.current_list),
        )

        print(f"Update question N count to: {questions_n_current + 1}")

        conn.commit()
        conn.close()

        # compute hash
        hash_ = self.compute_question_hash(question_hash_text, answers)

        exists = self.check_question_data_exists(hash_)
        if exists:
            print("Question data already exists, don't need to store again")

            conn, cursor = self._get_conn()
            cursor.execute(
                "SELECT seen FROM questionData WHERE question_hash = ? AND list_id = ?",
                (hash_, self.current_list),
            )
            row = cursor.fetchone()

            seen_value = row[0]

            cursor.execute(
                "UPDATE questionData SET seen = ? WHERE question_hash = ? AND list_id = ?",
                (seen_value + 1, hash_, self.current_list),
            )

            conn.commit()
            conn.close()

            print(f"Incremented to: {seen_value+1}")

            return

        # compute embedding

        answer = answers[correct_answer_index]

        embeddings = self.similarity_provider.compute_embeddings(answer)
        embeddings_blob = embeddings.tobytes()

        conn, cursor = self._get_conn()
        cursor.execute(
            """
INSERT OR IGNORE INTO questionData (list_id, question_hash, question_text, contextual_sentence, target_word, question_type, answer_string_1, answer_string_2, answer_string_3, answer_string_4, 
correct_answer_index, answer_embedding, seen) VALUES (?,?,  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       """,
            (
                self.current_list,
                hash_,
                self.clean_string(question_text),
                self.clean_string(contextual_sentence),
                self.clean_string(word),
                question_type,
                self.clean_string(answers[0]),
                self.clean_string(answers[1]),
                self.clean_string(answers[2]),
                self.clean_string(answers[3]),
                correct_answer_index,
                embeddings_blob,
                1,
            ),
        )
        conn.commit()
        conn.close()

    def compute_question_hash(self, question_text: str, question_answers: list[str]):
        """
        Computes the SHA-256 hexdigest hash of a question.
        
        Args:
            question_text (str): The question content. **Note**: this must include the contextual sentence!
            question_answers (list[str]): The answers to this question
        
        Returns:
            str: the hexdigest of the SHA-256 hash
        """
        
        question_answers_sorted = sorted(question_answers)
        
        
        question_hashtext = f"{question_text}+{'//'.join(question_answers_sorted)}"
        print(f"hashtext: {question_hashtext}")
        question_hash = hashlib.sha256(question_hashtext.encode()).hexdigest()
        return question_hash

    def _get_answer_reputation_lookup(
        self, cursor: sqlite3.Cursor, answer_text: str, target_word: str
    ) -> tuple[bool, str, tuple]:
        """
        Return the SQL query to be used for getting an answer's reputation value.

        Args:
            cursor (sqlite3.Cursor): Current actively being used SQLite3 database cursor
            answer_text (str): The answer text in string
            target_word (str): Target word being tested on
        Returns:
            tuple[bool, str, tuple]: Returns whether the row has been migrated (#1), the SQL query to use (#2), and the SQL parameters (#3)
        """
        # define queries
        query_strict = "SELECT id, answer_reputation FROM answers WHERE answer_text = ? AND list_id = ? AND target_word = ?"
        query_loose = "SELECT id, answer_reputation FROM answers WHERE answer_text = ? AND list_id = ?"
        query_impossible = "SELECT id, answer_reputation FROM answers WHERE 0" # ensure impossibility
        query_check_target_word = (
            "SELECT target_word FROM answers WHERE answer_text = ? AND list_id = ?"
        )
        query_check_target_word_strict = (
            "SELECT target_word FROM answers WHERE answer_text = ? AND list_id = ? AND target_word = ?"
        )
        
        # check if anything matches strict query
        cursor.execute(query_check_target_word_strict, (answer_text, self.current_list, target_word))
        row = cursor.fetchone()
        if row:
            # something already matches, just return the strict query
            return True, query_strict, (answer_text, self.current_list, target_word)

        # check if anything matches loose query
        cursor.execute(query_check_target_word, (answer_text, self.current_list))
        row = cursor.fetchone()
        if row:
            existing_target = row[0]
            
            # check if
            # - it matches the current target word (migrated) -> return strict query
            # - it hasn't been migrated yet                   -> return loose query
            # - it collides with another word                 -> return loose query

            if existing_target == target_word:
                print(f"ANSWER REPUTATION LOOKUP: migrated and match: {existing_target} {target_word} {answer_text} {target_word}")
                # strict match
                return True, query_strict, (answer_text, self.current_list, target_word)
            if existing_target == "":
                print(f"ANSWER REPUTATION LOOKUP: not migrated {answer_text}")
                # not migrated
                return False, query_loose, (answer_text, self.current_list)

            if existing_target != target_word:
                print(f"ANSWER REPUTATION LOOKUP: does not match {existing_target} {target_word} {answer_text}")
                # some other word is already there, so there is a collision
                # in this case, return an impossible query that matches nothing
                # forcing the creation to create a new entry with a different target word
                return False, query_impossible, ()
        print(f"ANSWER REPUTATION LOOKUP: no query found for: {query_check_target_word} with answer: {answer_text} in {self.current_list}")
        # otherwise return loose query
        return False, query_loose, (answer_text, self.current_list)

    def store_question_answer(
        self,
        answer_text: str,
        question_type: str,
        question_hash: str,
        new: bool,
        target_word: str,
    ):
        """
        Store the answer to this specific question hash and also stores or increments the answer reputation.

        Args:
            answer_text (str): The answer to this question as a string
            question_type (str): The type of question
            question_hash (str): SHA-256 hash of this question
            new (bool): Unused parameter
            target_word (str): The word this question is testing on

        Raises:
            AssertionError: if current list was never set
        """

        assert self.current_list != -1

        print(f"""
store question:
  answer_text = {answer_text}
  question_type = {question_type}
  question_hash = {question_hash}
  list_id = {self.current_list}
              """)
        conn, cursor = self._get_conn()

        cursor.execute(
            "INSERT OR IGNORE INTO questions (list_id, question_text, question_type) VALUES (?, ?, ?)",
            (self.current_list, question_hash, question_type),
        )
        # Get question ID
        cursor.execute(
            """
            SELECT id FROM questions WHERE question_text = ? AND list_id = ?
                       """,
            (question_hash, self.current_list),
        )

        question_id = cursor.fetchone()[0]

        # Check if the answer already exists
        strict_match, sql_query, params = self._get_answer_reputation_lookup(
            cursor, answer_text, target_word
        )
        cursor.execute(sql_query, params)

        row = cursor.fetchone()
        if row:
            print("answer already exists")
            id, reputation = row
            if strict_match:
                # use strict match (already migrated, target_word must match)
                cursor.execute(
                    """
                    UPDATE answers SET answer_reputation = ?, target_word = ? WHERE id = ? AND list_id = ? AND target_word = ?
                    """,
                    (reputation + 1, target_word, id, self.current_list, target_word),
                )
            else:
                # use loose query (no target_word defined yet)
                cursor.execute(
                    "UPDATE answers SET answer_reputation = ?, target_word = ? WHERE id = ? AND list_id = ?",
                    (reputation + 1, target_word, id, self.current_list),
                )
        else:
            print("answer does not exist, adding to database")
            cursor.execute(
                """
INSERT INTO answers (question_id, answer_text, answer_reputation, list_id, target_word) VALUES (?, ?, ?, ?, ?)
                """,
                (question_id, answer_text, 1, self.current_list, target_word),
            )

        conn.commit()
        conn.close()

    def lookup_answer_reputation(self, answer_text: str, target_word: str) -> float:

        assert self.current_list != -1

        print(f"lookup answer reputation where answer_text = {answer_text}")
        conn, cursor = self._get_conn()

        _, sql_query, params = self._get_answer_reputation_lookup(
            cursor, answer_text, target_word
        )

        cursor.execute(sql_query, params)

        row = cursor.fetchone()
        conn.close()

        return float(row[1]) if row else 0.0

    def lookup_question_id(self, question_hash: str) -> int:

        assert self.current_list != -1

        print(f"lookup question id where question_hash = {question_hash}")
        conn, cursor = self._get_conn()

        cursor.execute(
            "SELECT id FROM questions WHERE question_text = ? AND list_id = ?",
            (question_hash, self.current_list),
        )

        row = cursor.fetchone()
        conn.close()

        return row[0] if row else -1

    def lookup_answer(self, question_id: int) -> str:

        assert self.current_list != -1

        print(f"lookup answer where question_id = {question_id}")
        conn, cursor = self._get_conn()

        cursor.execute(
            "SELECT answer_text FROM answers WHERE question_id = ? AND list_id = ?",
            (question_id, self.current_list),
        )

        row = cursor.fetchone()
        conn.close()

        return row[0] if row else ""

    def lookup_probability_new_varaint(self) -> float:

        assert self.current_list != -1

        conn, cursor = self._get_conn()
        cursor.execute(
            "SELECT questionsCount FROM lists WHERE id = ?", (self.current_list,)
        )

        questions_count = cursor.fetchone()[0]

        if questions_count <= 0:
            print(f"question count <= 0, returning 1")
            return 1

        # select all
        cursor.execute(
            "SELECT COUNT(*) FROM questionData WHERE list_id = ? AND seen = 1",
            (self.current_list,),
        )
        n = cursor.fetchone()[0]

        p = n / questions_count

        conn.close()

        return p

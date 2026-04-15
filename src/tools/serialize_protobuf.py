from typing import TypedDict
import sqlite3
from lib import *


'''
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
    answer_embedding BLOB
'''


class QuestionData(TypedDict):
    question_text: str
    contextual_sentence: str
    target_word: str
    question_type: int
    answers: list[str]
    correct_answer_index: int
    answer_embedding: bytes

class QuestionDeduplicatedStrings(TypedDict):
    question_text: int
    contextual_sentence: int
    target_word: int
    question_type: int
    answers: list[int]
    correct_answer_index: int
    answer_embedding: bytes

class DeduplicationResults(TypedDict):
    deduplicated: list[str]
    questions: list[QuestionDeduplicatedStrings]

class ProtobufSerializer:
    
    def __init__(self, list_id: int, database_path: str = "database.db"):
        self.db_path = database_path
        self.list_id = list_id
    
    def open_connection(self) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        return conn, c

    def get_all_questions(self) -> list[QuestionData]:
        conn, cursor = self.open_connection()
        cursor.execute("SELECT question_hash, question_text, contextual_sentence, target_word, question_type, answer_string_1, answer_string_2, answer_string_3, answer_string_4, correct_answer_index, answer_embedding FROM questionData WHERE list_id = ?", (self.list_id,))
        
        questions: list[QuestionData] = []
        
        for question_hash, question_text, contextual_sentence, target_word, question_type, answer_string_1, answer_string_2, answer_string_3, answer_string_4, correct_answer_index, answer_embedding in cursor.fetchall():
            questions.append({
                "question_text": question_text,
                "contextual_sentence": contextual_sentence,
                "target_word": target_word,
                "question_type": question_type,
                "answers": [answer_string_1, answer_string_2, answer_string_3, answer_string_4],
                "answer_embedding": answer_embedding,
                "correct_answer_index": correct_answer_index
            })
        
        conn.close()
        
        return questions
    
    def deduplicate_strings(self, questions: list[QuestionData]) -> DeduplicationResults:
        
        deduplicated_strings_set: set[str] = set([])
        deduplicated_strings_index_map: dict[str, int] = {}
        deduplicated_strings_indices: list[str] = []
        
        def add_string(s: str) -> int:
            if not (s in deduplicated_strings_set):
                current_index = len(deduplicated_strings_indices)
                deduplicated_strings_index_map[s] = current_index
                deduplicated_strings_indices.append(s)
                deduplicated_strings_set.add(s)
                return current_index
            else:
                return deduplicated_strings_index_map[s]
        
        new_questions: list[QuestionDeduplicatedStrings] = []
        for question in questions:
            
            new_question: QuestionDeduplicatedStrings = {
                "answer_embedding": question["answer_embedding"],
                "answers": [add_string(a) for a in question["answers"]],
                "contextual_sentence": add_string(question["contextual_sentence"]),
                "correct_answer_index": question["correct_answer_index"],
                "question_text": add_string(question["question_text"]),
                "question_type": question["question_type"],
                "target_word": add_string(question["target_word"]),
            }
            
            new_questions.append(new_question)
        
        return {
            "deduplicated":  deduplicated_strings_indices,
            "questions": new_questions
        }
                
                
    
    def compute_embedding_indices(self, questions: list[QuestionDeduplicatedStrings]) -> dict[int, bytes]:
        
        data: dict[int, bytes] = {}
        
        for q in questions:
            answer_word = q["answers"][q["correct_answer_index"]]
            if data.get(answer_word) is not None:
                continue
            
            data[answer_word] = q["answer_embedding"]
        
        return data
                
        

    def serialize(self):
        
        questions_raw = self.get_all_questions()
        deduplication_results = self.deduplicate_strings(questions_raw)
        embedding_indices = self.compute_embedding_indices(deduplication_results["questions"])
        
        
        
        
        pack = Data(
            metadata=Metadata(
                version=1,
                list_id=self.list_id,
                producer="VCL_VDCA_SERVER"
            ),
            embeddings=[WordEmbedding(word, embedding) for word, embedding in embedding_indices.items()],
            questions=[Questions(
                question_text=q["question_text"],
                contextual_sentence=q["contextual_sentence"],
                answers=q["answers"],
                correct_answer_index=q["correct_answer_index"],
                target_word=q["target_word"]
            ) for q in deduplication_results["questions"]],
            deduplicated_strings=deduplication_results["deduplicated"]
        )
        
        binary_pack = pack.SerializeToString()
        
        with open(f"pack_{self.list_id}.vcl", "wb") as f:
            f.write(binary_pack)
            
if __name__ == "__main__":
    
    list_id = input("Enter list ID: ")
    try:
        _ = int(list_id)
    except ValueError as _:
        print(f"Invalid integer: {list_id}")
    
    list_id_int = int(list_id)
    
    serializer = ProtobufSerializer(list_id_int)
    serializer.serialize()
        
                
        

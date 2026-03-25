from SimilarityProvier import SimilarityProvider
from DatabaseProvider import DatabaseProvider
from LLMProvider import LLMProvider
from typing import TypedDict, cast
import textdistance
import numpy
from Math import Math

class LLMProviderResult(TypedDict):
    success: bool
    answer: str

class QuestionProbabilityProvider:
    
    def __init__(self):
        
        self._load()
        
    def _load(self):
        
        self.similarity_provider = SimilarityProvider()
        self.database_provider = DatabaseProvider()
        self.llm_provider = LLMProvider()
        
    def get_llm_similarities(self, target: str, answers: list[str]) -> LLMProviderResult:
        try:
            phrases_stripped = [answer.strip() for answer in answers]
            
            
            prompt = f"""{target} Is it {phrases_stripped[0]}, {phrases_stripped[1]}, {phrases_stripped[2]}, or {phrases_stripped[3]}?
    Give your answer without any explanation."""
            
            llm_response = self.llm_provider.getResponse(prompt)
            
            phrases_string_similarities: list[tuple[float, str]] = []
            
            for answer in answers:
                sim = textdistance.damerau_levenshtein.normalized_similarity(answer, llm_response)
                phrases_string_similarities.append((answer, sim))
            
            phrases_string_similarities.sort(key=lambda x: x[1])
            chosen = phrases_string_similarities[len(phrases_string_similarities) - 1][0]
            
            result: LLMProviderResult = cast(LLMProviderResult, {
                "answer": chosen,
                "success": True
            })
            
            return result
        except Exception as e:
            print(f"Failed to get similarities using an LLM: {e}")
            
            result: LLMProviderResult = cast(LLMProviderResult, {
                "answer": "",
                "success": False
            })
            
            return result
        
        
    def get_probability(self, question_text: str, target_word: str, phrases: list[str]):
        
        question_text = question_text.strip()
        target_word = target_word.strip()
        phrases = [p.strip() for p in phrases]
        
        answer_probabilities: dict[str, float] = {}
        
        computation_results = self.similarity_provider.compute_similarity(target_word, phrases)
        is_confident = computation_results["confident"]
        cosine_similarities = computation_results['similarities']
        
        if is_confident is False:
            print("not confident, trying LLM")
            llm_computation_results = self.get_llm_similarities(question_text, phrases)
            if llm_computation_results["success"] is True:
                # get index
                try:
                    answer_index = phrases.index(llm_computation_results["answer"])
                    
                    cosine_similarities[answer_index] = 10
                except ValueError:
                    print("llm postprocess failed: cannot find index")
            else:
                print("llm processing failed")
                 
        question_hash  =self.database_provider.compute_question_hash(question_text.strip(), [p.strip() for p in phrases])
        question_answer = self.database_provider.lookup_answer(self.database_provider.lookup_question_id(question_hash))
        
        
        
        for answer, sim in zip(phrases, cosine_similarities):
            answer_probabilities[answer] = sim
            if question_answer != "" and answer.strip() == question_answer.strip():
                print(f"known question answer found: answer = {answer}, qhash = {question_hash} boosted by 20")
                answer_probabilities[answer] = sim + 20
            
        # lookup answer reputation
        
        for answer, probability in answer_probabilities.items():
            reputation = self.database_provider.lookup_answer_reputation(answer)
            
            answer_probabilities[answer] = probability + reputation
            print(f"boosted answer: {answer} by {reputation} as stated by reputation")
        
        probabilities_array = list(answer_probabilities.values())
        probability_nparray = numpy.array(probabilities_array)
        probabilities_softmax = Math.softmax(probability_nparray)
        
        for answer, probability in zip(answer_probabilities.keys(), probabilities_softmax):
            answer_probabilities[answer] = float(probability)
        
        print(f"Answer probabilities: {answer_probabilities}")
        
        return answer_probabilities
    
    def record_question_data(self, question_text: str, question_type: str, question_answer: str, question_possible_answers: list[str]):
        question_hash  =self.database_provider.compute_question_hash(question_text.strip(), [p.strip() for p in question_possible_answers])
        self.database_provider.store_question_answer(question_answer.strip(), question_type, question_hash, True)
            
        
        
        
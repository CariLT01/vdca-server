from similarity_provider import SimilarityProvider
from database_provider import DatabaseProvider
from llm_provider import LLMProvider
from typing import TypedDict, cast
import textdistance
import numpy
from math_utils import Math


class LLMProviderResult(TypedDict):
    success: bool
    answer: str


class AnswerProbabilityProvider:

    def __init__(
        self,
        similarity_provider: "SimilarityProvider",
        database_provider: "DatabaseProvider",
    ):

        self.similarity_provider = similarity_provider
        self._load(database_provider)

    def _load(self, database_provider: "DatabaseProvider"):

        self.database_provider = database_provider
        self.llm_provider = LLMProvider()

    def get_llm_similarities(
        self, target: str, answers: list[str]
    ) -> LLMProviderResult:
        try:
            phrases_stripped = [answer.strip() for answer in answers]

            prompt = f"""{target} Is it {phrases_stripped[0]}, {phrases_stripped[1]}, {phrases_stripped[2]}, or {phrases_stripped[3]}?
    Give your answer without any explanation."""

            llm_response = self.llm_provider.get_response(prompt)

            phrases_string_similarities: list[tuple[str, float]] = []

            for answer in answers:
                sim = textdistance.damerau_levenshtein.normalized_similarity(
                    answer, llm_response # type: ignore
                )
                phrases_string_similarities.append((answer, sim))

            phrases_string_similarities.sort(key=lambda x: x[1])
            chosen = phrases_string_similarities[len(phrases_string_similarities) - 1][
                0
            ]

            result: LLMProviderResult = cast(
                LLMProviderResult, {"answer": chosen, "success": True}
            )

            return result
        except Exception as e:
            print(f"Failed to get similarities using an LLM: {e}")

            result: LLMProviderResult = cast(
                LLMProviderResult, {"answer": "", "success": False}
            )

            return result

    def get_answers_reputations(
        self, question_text: str, phrases: list[str], target_word: str
    ) -> tuple[bool, dict[str, float]]:
        """
        Get the reputation of the answer. Returns whether one of the answer is reputated and the added reputation for each answer.

        Args:
            question_text (str): The content of the question
            phrases (list[str]): The list of answers to this question
            target_word (str): The word being tested on

        Returns:
            tuple[bool, dict[str, float]]: A tuple containing if any of the answers is reputated (first value) and a answer to reputation map (second value).
        """

        reputation_probabilities: dict[str, float] = {}
        any_answer_has_reputation: bool = False

        question_hash = self.database_provider.compute_question_hash(
            question_text.strip(), [p.strip() for p in phrases]
        )
        question_answer = self.database_provider.lookup_answer(
            self.database_provider.lookup_question_id(question_hash)
        )

        print(f"lookup answers reputations: looked up answer: {question_answer}")

        for answer in phrases:
            if question_answer != "" and answer.strip() == question_answer.strip():
                print(
                    f"known question answer found: answer = {answer}, qhash = {question_hash} boosted by 20"
                )
                reputation_probabilities[answer] = 9999
            else:
                reputation_probabilities[answer] = 0

        for answer, original_reputation in reputation_probabilities.items():
            reputation = self.database_provider.lookup_answer_reputation(
                answer, target_word
            )

            print(f"lookup answers reputation: added reputation: {reputation}")
            reputation_probabilities[answer] = original_reputation + reputation

        # print reputation

        for answer, reputation in reputation_probabilities.items():
            print(f"Reputation: {answer} = +{reputation}")

            if reputation > 0:
                any_answer_has_reputation = True

        return any_answer_has_reputation, reputation_probabilities

    def get_probability(
        self, question_text: str, target_word: str, phrases: list[str]
    ) -> dict[str, float]:
        """
        Compute the probabilities for each answer with a given question and target word.

        Args:
            question_text (str): The content of the question
            target_word (str): The word that this question is testing on
            phrases (list[str]): The answers to the question

        Returns:
            dict[str, float]: An unsorted map of each answer to its probability (normalized in 0~1 range)
        """

        question_text = question_text.strip()
        target_word = target_word.strip()
        phrases = [p.strip() for p in phrases]

        answer_probabilities: dict[str, float] = {}

        # -- Compute reputation first --
        any_reputation, answer_reputations = self.get_answers_reputations(
            question_text, phrases, target_word
        )

        if any_reputation:
            # if there's any reputation, set directly to the answer probabilities and skip all the similarity logic

            print("Found answer reputation, skipping similarity logic")

            answer_probabilities = answer_reputations
            return answer_probabilities

        # otherwise, compute similarity-based, with LLM fallback if necessary

        computation_results = self.similarity_provider.compute_similarity(
            target_word, phrases
        )
        is_confident = computation_results["confident"]
        cosine_similarities = computation_results["similarities"]

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

        for answer, sim in zip(phrases, cosine_similarities):
            answer_probabilities[answer] = sim

        probabilities_array = list(answer_probabilities.values())
        probability_nparray = numpy.array(probabilities_array)
        probabilities_softmax = Math.softmax(probability_nparray)

        for answer, probability in zip(
            answer_probabilities.keys(), probabilities_softmax
        ):
            answer_probabilities[answer] = float(probability)

        print(f"Answer probabilities: {answer_probabilities}")

        return answer_probabilities

    def record_question_data(
        self,
        question_text: str,
        question_type: str,
        question_answer: str,
        question_possible_answers: list[str],
        target_word: str,
    ):
        question_hash = self.database_provider.compute_question_hash(
            question_text.strip(), [p.strip() for p in question_possible_answers]
        )
        self.database_provider.store_question_answer(
            question_answer.strip(), question_type, question_hash, True, target_word
        )

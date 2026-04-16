"""
Main file and entry point.

Handles I/O, networking, and glues everything together.
"""

# performance import
from perf_utils import Utils

Utils.begin_time("load_libs")

# Built-in imports
import threading
from typing import TypedDict

# Third-party imports
Utils.begin_time("load_libs")
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
import keyboard
from pydantic import BaseModel, ValidationError
Utils.end_time("load_libs")



Utils.begin_time("load_services")
# services & providers
from similarity_provider import SimilarityProvider
from answer_probability_provider import AnswerProbabilityProvider
from database_provider import DatabaseProvider
from question_type import QuestionType
from llm_provider import LLMProvider
from automation_controller import AutomationController
Utils.end_time("load_services")


class SimilarityData(TypedDict):
    """
    Represents the similarity data.
    """
    words: list[str]
    target: str
    word: str


class QuestionReportData(TypedDict):
    """
    Datastruct given by client for reporting an answer.
    """
    question_content: str
    question_type: str
    answer: str
    possible_answers: list[str]
    target_word: str


class QuestionReportResponse(BaseModel):
    """
    Datastruct given by client for reporting a question.
    """
    question_text: str
    question_hash: str
    contextual_sentence: str
    target_word: str
    question_type: int
    answers: list[str]
    correct_answer_index: int


class ListSwitchResponse(BaseModel):
    """
    Datastruct given by client for switching to a new list ID.
    """
    list_id: int


class App:

    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app, origins="*", supports_credentials=True)  # fix cors...
        self.app.config["SECRET_KEY"] = "no secret"

        self.killed = False
        self.similarity_provider = SimilarityProvider()
        self.database_provider = DatabaseProvider(
            self.similarity_provider, database_path="database.db"
        )
        self.question_probability_provider = AnswerProbabilityProvider(
            self.similarity_provider, self.database_provider
        )

        self.socketio = SocketIO(
            app=self.app, cors_allowed_origins="*"
        )  # Allow for testing

        self.loadModel()

        self.app.add_url_rule(
            "/api/v1/question/report",
            view_func=self.report_question_endpoint,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/api/v1/list/switch", view_func=self.switch_list_endpoint, methods=["POST"]
        )
        self.app.add_url_rule(
            "/api/v1/list/new_variant_probability",
            view_func=self.new_variant_probability_endpoint,
            methods=["GET"],
        )

        # SocketIO click handler
        @self.socketio.on("click")
        def handle_click_message(data):
            if self.killed:
                print("Kill switch active. Ignoring click.")
                return

            x = data.get("x")
            y = data.get("y")

            # console.print(f"Clicking at {x} {y}")
            # pyautogui.moveTo(x=x, y=y, duration=0.5)
            AutomationController.move_mouse(x, y)
            AutomationController.mouse_click()
            return "DONE"

        @self.socketio.on("locateChoices")
        def handle_locate_choices(data):
            # Take screenshot
            correct_icons = AutomationController.locate_icons(
                "correct.png", threshold=0.9, min_distance=10, icon_type="correct"
            )
            incorrect_icons = AutomationController.locate_icons(
                "incorrect.png", threshold=0.9, min_distance=10, icon_type="incorrect"
            )

            all_icons = correct_icons + incorrect_icons

            # Sort by y, then x (top-left → bottom-right)
            all_icons.sort(key=lambda p: (p["y"], p["x"]))

            print(all_icons)

            return all_icons

        @self.socketio.on("locateSpell")
        def handle_locate_spell(data):

            if self.killed:
                print("Kill switch active. Ignoring click.")
                return

            play_buttons = AutomationController.locate_icons(
                "spellit_play.png",
                threshold=0.9,
                min_distance=10,
                icon_type="play_button",
            )

            return play_buttons

        @self.socketio.on("locateSpellButton")
        def handle_locate_spell_button(data):
            spell_buttons = AutomationController.locate_icons(
                "spellit_spell.png",
                threshold=0.9,
                min_distance=10,
                icon_type="spell_button",
            )

            return spell_buttons

        @self.socketio.on("type")
        def handle_type(data: str):

            AutomationController.type_text(data)

        @self.socketio.on("similarity")
        def handle_similarity(data: SimilarityData):

            if self.killed == True:
                return {}

            probabilities: dict[str, float] = (
                self.question_probability_provider.get_probability(
                    target_word=data["word"],
                    question_text=data["target"],
                    phrases=data["words"],
                )
            )

            return probabilities

        @self.socketio.on("report_question_data")
        def handle_question_report(data: QuestionReportData):

            self.question_probability_provider.record_question_data(
                data["question_content"],
                data["question_type"],
                data["answer"],
                data["possible_answers"],
                data["target_word"],
            )

        # Start kill switch listener in background
        threading.Thread(target=self._kill_switch_listener, daemon=True).start()

    def switch_list_endpoint(self):

        try:

            json_data = request.get_json()
            data = ListSwitchResponse(**json_data)
            list_id_int = data.list_id

            if list_id_int <= 0:
                return jsonify(ok=False, message="List ID must be more than zero"), 400

            self.database_provider.switch_to_list(list_id_int)

            return jsonify(ok=True, message="OK"), 200

        except ValidationError as e:
            return jsonify(ok=False, message="Validation error", errors=e.errors()), 400

        except Exception as e:
            print(f"Failed to switch to list: {e}")
            return jsonify(ok=False, message="Internal Server Error"), 500

    def new_variant_probability_endpoint(self):
        try:

            p = self.database_provider.lookup_probability_new_varaint()
            return jsonify(ok=True, message="OK", data={"probability": p}), 200
        except Exception as e:
            print(f"Failed to lookup probability for new variant: {e}")
            return jsonify(ok=False, message="Internal Server Error"), 500

    def report_question_endpoint(self):
        try:

            if self.killed:
                return jsonify(ok=False, message="Unavailable"), 503

            data = request.get_json()

            try:

                validated_data = QuestionReportResponse(**data)

                self.database_provider.add_question_data(
                    validated_data.question_hash,
                    validated_data.question_text.replace("\n", "")
                    .replace("\t", "")
                    .replace("\r", ""),
                    validated_data.contextual_sentence.replace("\n", "")
                    .replace("\t", "")
                    .replace("\r", ""),
                    validated_data.target_word,
                    QuestionType(validated_data.question_type),
                    validated_data.answers,
                    validated_data.correct_answer_index,
                )

                return jsonify(ok=True, message="OK"), 200

            except ValidationError as e:

                return (
                    jsonify(ok=False, message="Validation error", errors=e.errors()),
                    400,
                )
        except Exception as e:
            print(f"Failed to report question: {e}")
            return jsonify(ok=False, message="Internal server error"), 500

    def loadModel(self):
        self.LLM_provider = LLMProvider(self.socketio)

    def _kill_switch_listener(self):
        while True:
            print("Press ESC to stop the application.")
            keyboard.wait("esc")  # blocks until ESC is pressed
            print("Kill switch activated! Exiting... Press ESC again to restart")
            self.killed = True
            keyboard.wait("esc")
            print("Application restarted...")
            self.killed = False

    def run(self):
        # Use socketio.run instead of app.run to handle websockets properly
        self.socketio.run(self.app, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    Utils.begin_time("init")
    app = App()
    Utils.end_time("init")
    app.run()

ENABLE_DEEP_THINK = True
from rich.console import Console
from rich.spinner import Spinner
from rich.live import Live
from QuestionChanceProvider import QuestionProbabilityProvider
import re

console = Console()

spinner = Spinner("dots", text="Loading libraries...")
live = Live(spinner, console=console, refresh_per_second=10)
live.start()

from flask import Flask
from flask_socketio import SocketIO, emit
import pyautogui
import threading
import keyboard
import sys
from typing import cast
import time
import math
import numpy as np
import cv2
import ctypes
import random
import LLMProvider as LLMProvider
import requests
import textdistance
from PIL import ImageGrab

live.stop()

spinner = Spinner("dots", text="Loading transformers...")
live = Live(spinner, console=console, refresh_per_second=10)
live.start()

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from typing import TypedDict

live.stop()

class SimilarityData(TypedDict):
    words: list[str]
    target: str
    word: str
    
class QuestionReportData(TypedDict):
    question_content: str
    question_type: str
    answer: str
    possible_answers: list[str]


SetCursorPos = ctypes.windll.user32.SetCursorPos

def locate_icons(template_path, threshold=0.8, min_distance=10, icon_type="correct"):
    # Take a screenshot
    screenshot = np.array(ImageGrab.grab())
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

    # Load template
    template = cv2.imread(template_path)
    if template is None: raise RuntimeError("not exist")
    w, h = template.shape[1], template.shape[0]

    # Match template
    res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    # Collect all points
    points = np.array([[pt[0], pt[1]] for pt in zip(*loc[::-1])])

    # Apply Non-Maximum Suppression (to remove overlapping detections)
    final_points = []
    for pt in points:
        keep = True
        for fpt in final_points:
            if np.linalg.norm(pt - fpt) < min_distance:
                keep = False
                break
        if keep:
            final_points.append(pt)

    # Convert to dict with center coordinates
    results = [{"x": int(pt[0] + w/2), "y": int(pt[1] + h/2), "type": icon_type} for pt in final_points]

    return results

def bezier_point(t, points):
    """
    Compute a point on a Bézier curve at parameter t.

    :param t: float in [0,1], position along curve
    :param points: list of control points [(x,y), (x,y), ...]
                   Includes start point A, control points, and end point B
    :return: (x,y) coordinates at t
    """
    # Work on a copy to avoid modifying input
    pts = [list(p) for p in points]
    n = len(pts)
    
    # De Casteljau's algorithm
    for r in range(1, n):
        for i in range(n - r):
            pts[i][0] = (1 - t) * pts[i][0] + t * pts[i+1][0]
            pts[i][1] = (1 - t) * pts[i][1] + t * pts[i+1][1]
    
    return tuple(pts[0])

def generate_control_points(start_x, start_y, end_x, end_y):
    # Distance between start and end
    dx = end_x - start_x
    dy = end_y - start_y
    distance = math.hypot(dx, dy)

    # Number of control points grows with distance
    num_points = max(1, int(distance // 100))  # ~1 per 200px

    points = [(start_x, start_y)]

    for i in range(1, num_points + 1):
        # Progress fraction along the straight line
        t = i / (num_points + 1)

        # Base linear interpolation point
        px = start_x + dx * t
        py = start_y + dy * t

        # Add random displacement
        offset_x = random.randint(-int(distance * 0.3), int(distance * 0.3))
        offset_y = random.randint(-int(distance * 0.3), int(distance * 0.3))

        points.append((px + offset_x, py + offset_y))

    points.append((end_x, end_y))
    return points

# Example usage

def ease_in_out_sine(t: float) -> float:
    """
    Remaps t in [0,1] to a sine-based ease-in-out curve.
    """
    return -(math.cos(math.pi * t) - 1) / 2

def moveMouse(end_x: int, end_y: int):
    start_x, start_y = pyautogui.position()
    points = generate_control_points(start_x, start_y, end_x, end_y)
    
    combined = points


    dx = end_x - start_x
    dy = end_y - start_y
    distance = math.hypot(dx, dy)
    
    movement_speed = max(1, min(distance, 1000))
   
    steps = int(distance / (movement_speed / 60))
    frame_time = 1/60
    if (steps <= 0): return points
    for i in range(steps + 1):
        start = time.perf_counter()

        if keyboard.is_pressed("esc"):
            break

        t = ease_in_out_sine(i / steps)
        x, y = bezier_point(t, combined)
        SetCursorPos(int(x), int(y))

        elapsed = time.perf_counter() - start
        remaining = frame_time - elapsed
        if remaining > 0:
            time.sleep(remaining)
        else:
            console.print(f"Exceeded by: {remaining}")
            
    #console.print(points)
    return points


spinner = Spinner("dots", text="Loading AI similarity model...")
live = Live(spinner, console=console, refresh_per_second=10)
live.start()

model = SentenceTransformer("thenlper/gte-base")  # small & fast
model_name = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
modelRoberta = AutoModelForMaskedLM.from_pretrained(model_name)

live.stop()

def rank_choices_old(sentence, choices):
    """
    Return MLM scores for candidate words in the given sentence.
    Returns a list of floats corresponding to `choices` (unsorted).
    """
    mask_token = tokenizer.mask_token
    sentence = sentence.replace("[MASK]", mask_token)
    input_ids = tokenizer(sentence, return_tensors="pt")["input_ids"]
    mask_idx = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    with torch.no_grad():
        outputs = modelRoberta(input_ids)
        logits = outputs.logits[0, mask_idx, :]  # shape: [mask_position, vocab_size]

    scores = []
    for word in choices:
        token_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        score = logits[:, token_ids].mean().item()  # average if multiple tokens
        scores.append(score)

    return scores

def score_choice(choice, sentence: str) -> float:
    prompt = sentence.replace("[MASK]", choice)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = modelRoberta(input_ids, labels=input_ids)
        loss = outputs.loss.item()
    return -loss  # higher = better

def rank_choices(sentence, choices: list[str]):



    scores = [(c, score_choice(c, sentence)) for c in choices]

    # Step 1: find the minimum (most negative) value
    min_score = min(score for _, score in scores)

    # Step 2: shift all scores so the lowest becomes zero
    shifted_scores = [(word, score - min_score) for word, score in scores]

    finalList = [c[1] for c in shifted_scores]

    print(finalList)

    return finalList


def fetchDefinitions(target: str):
    if len(target.split(" ")) != 1: return None
    URL = f"https://api.dictionaryapi.dev/api/v2/entries/en/{target}"
    response = requests.get(URL)
    if (response.status_code != 200): return None
    
    # Fetch all meanings
    
    data = response.json()[0]
    
    index = -1
    maxDefinitionsLength = 0
    
    # Loop through meanings, found the most used indexes
    for i, meaning in enumerate(data["meanings"]):
        definitionsLength = len(meaning["definitions"])
        if definitionsLength > maxDefinitionsLength:
            index = i
    
    if index == -1: return None
    
    
    meanings = data["meanings"][index]
    definition = meanings["definitions"][0]["definition"]
    
    return definition
    
    



class App:

    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'no secret'

        self.killed = False
        self.question_probability_provider = QuestionProbabilityProvider()
        

        self.socketio = SocketIO(app=self.app, cors_allowed_origins='*')  # Allow for testing

        self.loadModel()




        # SocketIO click handler
        @self.socketio.on("click")
        def handle_click_message(data):
            if self.killed:
                console.print("Kill switch active. Ignoring click.")
                return

            x = data.get("x")
            y = data.get("y")

            #console.print(f"Clicking at {x} {y}")
            #pyautogui.moveTo(x=x, y=y, duration=0.5)
            moveMouse(x, y)
            pyautogui.click()
            return "DONE"
        
        @self.socketio.on("locateChoices")
        def handle_locate_choices(data):
            # Take screenshot
            correct_icons = locate_icons("correct.png", threshold=0.9, min_distance=10, icon_type="correct")
            incorrect_icons = locate_icons("incorrect.png", threshold=0.9, min_distance=10, icon_type="incorrect")

            all_icons = correct_icons + incorrect_icons

            # Sort by y, then x (top-left → bottom-right)
            all_icons.sort(key=lambda p: (p["y"], p["x"]))

            console.print(all_icons)

            return all_icons

        @self.socketio.on("locateSpell")
        def handle_locate_spell(data):

            if self.killed:
                console.print("Kill switch active. Ignoring click.")
                return

            play_buttons = locate_icons("spellit_play.png", threshold=0.9, min_distance=10, icon_type="play_button")

            return play_buttons
        
        @self.socketio.on("locateSpellButton")
        def handle_locate_spell_button(data):
            spell_buttons = locate_icons("spellit_spell.png", threshold=0.9, min_distance=10, icon_type="spell_button")

            return spell_buttons

        @self.socketio.on("type")
        def handle_type(data: str):

            pyautogui.typewrite(data, interval=0.1)

        @self.socketio.on("similarity")
        def handle_similarity(data: SimilarityData):

            probabilities: dict[str, float] = self.question_probability_provider.get_probability(target_word=data["word"], question_text=data["target"], phrases=data["words"])

            return probabilities
        
        @self.socketio.on("report_question_data")
        def handle_question_report(data: QuestionReportData):
            
            self.question_probability_provider.record_question_data(data["question_content"], data["question_type"], data["answer"], data["possible_answers"])

        

        # Start kill switch listener in background
        threading.Thread(target=self._kill_switch_listener, daemon=True).start()

    def loadModel(self):
        if ENABLE_DEEP_THINK:
            self.llmProvider = LLMProvider.LLMProvider(self.socketio)
            self.llmProvider.loadProvider()


    def _kill_switch_listener(self):
        console.print("Press ESC to stop the application.")
        keyboard.wait('esc')  # blocks until ESC is pressed
        console.print("Kill switch activated! Exiting...")
        self.killed = True
        sys.exit(0)

    def run(self):
        # Use socketio.run instead of app.run to handle websockets properly
        self.socketio.run(self.app, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    app = App()
    app.run()

"""
Utilities for automation.

This module provides the AutomationController class, used for simulating natural human
movements for automation and avoiding bot detection.
"""

# import performance tests
from perf_utils import Utils

import math
import time
import random
from typing import TypedDict
import types

import ctypes
import pyautogui
import keyboard

from PIL import ImageGrab


class IconLocationEntry(TypedDict):
    """
    Represents one icon location returned by `locate_icons`.

    Attributes:
        x (int): the X position of the icon on screen (center)
        y (int): the Y position of the icon on screen (center)
        type (str): the type of the icon
    """

    x: int
    y: int
    type: str


class AutomationController:
    """
    Controller that wraps a few methods that control automation feature such as mouse movements.
    """

    SetCursorPos = ctypes.windll.user32.SetCursorPos
    
    # lazy-load packages
    
    open_cv_loaded: bool = False
    numpy_loaded: bool = False
    
    np: types.ModuleType
    cv2: types.ModuleType
    
    @staticmethod
    def locate_icons(
        template_path: str, threshold=0.8, min_distance=10, icon_type="correct"
    ) -> list[IconLocationEntry]:
        """
        Locate an icon from a template.

        Args:
            template_path (str): the path to the template image (recommended format: PNG)
            threshold (float): the confidence threshold (default=0.8)
            min_distance (int): minimum distance
            icon_type (str): the type of the icon as a string

        Raises:
            RuntimeError: when the icon does not exist

        Returns:
            list[IconLocationEntry]: the locations of all locations that matched the template
        """

        if not AutomationController.open_cv_loaded:
            Utils.tbegin("import_cv2")
            import cv2
            Utils.tend("import_cv2")
            AutomationController.open_cv_loaded = True
            AutomationController.cv2 = cv2
        if not AutomationController.numpy_loaded:
            Utils.tbegin("import_np")
            import numpy as np
            Utils.tend("import_np")
            AutomationController.numpy_loaded = True
            AutomationController.np = np

        np = AutomationController.np
        cv2 = AutomationController.cv2

        print(f"Locating icon: {icon_type}")
        # Take a screenshot
        screenshot = np.array(ImageGrab.grab())
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

        # Load template
        template = cv2.imread(template_path)
        if template is None:
            raise RuntimeError("not exist")
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
        results: list[IconLocationEntry] = [
            {"x": int(pt[0] + w / 2), "y": int(pt[1] + h / 2), "type": icon_type}
            for pt in final_points
        ]

        return results

    @staticmethod
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
                pts[i][0] = (1 - t) * pts[i][0] + t * pts[i + 1][0]
                pts[i][1] = (1 - t) * pts[i][1] + t * pts[i + 1][1]

        return tuple(pts[0])

    @staticmethod
    def generate_control_points(start_x, start_y, end_x, end_y):
        """
        Generates control points for the bezier curve.

        Args:
            start_x (int): starting position of the X axis
            start_y (int): starting position of the Y axis
            end_x (int): ending position of the X axis
            end_y (int): ending position of the Y axis

        Returns:
            list[tuple[int, int]]: a list of position pairs (x, y)
        """

        # Distance between start and end
        dx = end_x - start_x
        dy = end_y - start_y
        distance = math.hypot(dx, dy)

        # Number of control points grows with distance
        num_points = max(1, int(distance // 100))  # ~1 per 200px

        points: list[tuple[int, int]] = [(start_x, start_y)]

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

    @staticmethod
    def ease_in_out_sine(t: float) -> float:
        """
        Remaps t in [0,1] to a sine-based ease-in-out curve.

        Args:
            t (float): the time

        Returns:
            float: mapped value to sine-based ease-in-out curve
        """
        return -(math.cos(math.pi * t) - 1) / 2

    @staticmethod
    def move_mouse(end_x: int, end_y: int) -> list[tuple[int, int]]:
        """
        Moves the mouse naturally by simulating human movement.

        Args:
            end_x (int): Ending position of the X axis
            end_y (int): Ending position of the Y axis

        Returns:
            list[tuple[int, int]]: returns the list of control points used
        """

        start_x, start_y = pyautogui.position()
        points = AutomationController.generate_control_points(
            start_x, start_y, end_x, end_y
        )

        combined = points

        dx = end_x - start_x
        dy = end_y - start_y
        distance = math.hypot(dx, dy)

        movement_speed = max(1, min(distance, 1000))

        steps = int(distance / (movement_speed / 60))
        frame_time = 1 / 60
        if steps <= 0:
            return points
        for i in range(steps + 1):
            start = time.perf_counter()

            if keyboard.is_pressed("esc"):
                break

            t = AutomationController.ease_in_out_sine(i / steps)
            x, y = AutomationController.bezier_point(t, combined)
            AutomationController.SetCursorPos(int(x), int(y))

            elapsed = time.perf_counter() - start
            remaining = frame_time - elapsed
            if remaining > 0:
                time.sleep(remaining)
            else:
                print(f"Exceeded by: {remaining}")

        # console.print(points)
        return points

    @staticmethod
    def type_text(text: str) -> None:
        """
        Type text like human.
        
        Args:
            text (str): text to type
        """
        
        pyautogui.typewrite(text, interval=0.1)
    
    @staticmethod
    def mouse_click() -> None:
        """Click the mouse"""
        pyautogui.click()
        
from typing import List, Dict, Any

import cv2
import numpy as np


class HudRenderer:
    def __init__(self):
        self.gesture_color = (57, 255, 20)  # neon green
        self.flame_color = (0, 0, 255)  # red
        self.object_color = (255, 140, 0)  # orange

    def draw(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        overlay = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det.get("confidence", 0.0)
            module = det.get("module", "")

            if module == "Air-Canvas":
                color = self.gesture_color
                points = det.get("extra", {}).get("points", [])
                for i in range(1, len(points)):
                    cv2.line(overlay, points[i - 1], points[i], color, 3)
            elif module == "FlameSentinel":
                color = self.flame_color
                self._draw_pulsing_box(overlay, (x1, y1, x2, y2), color, label, conf)
            else:
                color = self.object_color
                self._draw_pulsing_box(overlay, (x1, y1, x2, y2), color, label, conf)

        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    @staticmethod
    def _draw_pulsing_box(img, bbox, color, label, conf):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label} ({conf:.2f})"
        cv2.putText(img, text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

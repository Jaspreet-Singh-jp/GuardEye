from dataclasses import dataclass
from typing import List, Dict, Any
import cv2
import numpy as np

@dataclass
class Detection:
    module: str
    label: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)

class AirCanvasEngine:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        
        # THE SWEET SPOT: Relaxes the strictness to catch normal blue objects/phones, 
        # but keeps enough saturation to completely ignore red curtains and white walls.
        self.lower_color = np.array([90, 80, 80]) 
        self.upper_color = np.array([130, 255, 255])
        
        self.canvas = None
        self.xp, self.yp = 0, 0 
        
        # 🛑 THE SMART LOGGER: Remembers if we are already logging to prevent table spam!
        self.is_drawing = False

    def clear_canvas(self):
        if self.canvas is not None:
            self.canvas = np.zeros_like(self.canvas)
        self.xp, self.yp = 0, 0

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        h, w, c = frame.shape
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # --- DRAW THE HUD ---
        cv2.putText(frame, "AIR CANVAS RULES:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "1. Show Blue Object to DRAW", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "2. Hide Object to PAUSE", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- COLOR TRACKING LOGIC ---
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        except:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)

        # Clean the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # This holds the log data we send back to Streamlit
        detections_to_log = []

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)

            # Moderate size limit to ignore tiny blue specs
            if cv2.contourArea(largest_contour) > 1000:
                x, y, bw, bh = cv2.boundingRect(largest_contour)
                cx, cy = x + (bw // 2), y + (bh // 2)

                cv2.circle(frame, (cx, cy), 15, (255, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                cv2.putText(frame, "DRAWING...", (w//2 - 80, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                
                if self.xp == 0 and self.yp == 0:
                    self.xp, self.yp = cx, cy

                cv2.line(self.canvas, (self.xp, self.yp), (cx, cy), (255, 0, 255), 10)
                self.xp, self.yp = cx, cy
                
                # 🛑 THE SMART LOGGING TRIGGER
                if not self.is_drawing:
                    # We just started drawing! Log it exactly ONCE.
                    self.is_drawing = True
                    detections_to_log.append({
                        "module": "Air-Canvas",
                        "label": "Blue Marker Engaged",
                        "confidence": 1.0,
                        "bbox": (x, y, x + bw, y + bh)
                    })

            else:
                # The blue object is too small, assume it's paused
                self.xp, self.yp = 0, 0
                self.is_drawing = False # Reset the memory switch!
                cv2.putText(frame, "PAUSED", (w//2 - 50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # No blue object at all
            self.xp, self.yp = 0, 0
            self.is_drawing = False # Reset the memory switch!
            cv2.putText(frame, "PAUSED", (w//2 - 50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # --- MERGE CANVAS WITH CAMERA FEED ---
        imgGray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

        cv2.bitwise_and(frame, imgInv, dst=frame)
        cv2.bitwise_or(frame, self.canvas, dst=frame)

        # Return the detections list (it will be empty 99% of the time, stopping the flicker)
        return detections_to_log
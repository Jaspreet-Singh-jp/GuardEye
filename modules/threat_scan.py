from collections import deque
from pathlib import Path
from typing import List, Dict, Any


import numpy as np
from ultralytics import YOLO


class ThreatScanEngine:
    def __init__(
        self,
        enabled: bool = True,
        weights_path: str = "models/yolov8n_weapons_kagglesecnd.pt",
        conf_thres: float = 0.4,
    ):
        self.enabled = enabled

        # Resolve weights relative to the project root (repo root)
        # so the app works regardless of current working directory.
        repo_root = Path(__file__).resolve().parent.parent
        resolved_weights_path = Path(weights_path)
        if not resolved_weights_path.is_absolute():
            resolved_weights_path = repo_root / resolved_weights_path

        if not resolved_weights_path.exists():
            raise FileNotFoundError(
                f"Threat/weapon weights not found: {resolved_weights_path}\n"
                "Place the YOLOv8 weapon/prohibited-object weights at the path above "
                "(recommended filename: models/yolov8n_weapons.pt)."
            )

        self.model = YOLO(str(resolved_weights_path))
        self.conf_thres = conf_thres
        self._motion_history = deque(maxlen=16)


    def _update_motion(self, frame: np.ndarray) -> float:
        gray = frame.mean(axis=2).astype("float32")
        if not self._motion_history:
            self._motion_history.append(gray)
            return 0.0
        prev = self._motion_history[-1]
        diff = np.abs(gray - prev)
        motion_score = float(diff.mean())
        self._motion_history.append(gray)
        return motion_score

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        motion_score = self._update_motion(frame)
        results = self.model.predict(frame, conf=self.conf_thres, verbose=False)

        detections: List[Dict[str, Any]] = []
        suspicious_motion = motion_score > 8.0

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = r.names.get(cls_id, "object")

                det = {
                    "module": "ThreatScan",
                    "label": label,
                    "confidence": conf,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "extra": {
                        "motion_score": motion_score,
                        "suspicious_motion": suspicious_motion,
                    },
                }
                detections.append(det)

        if suspicious_motion:
            h, w, _ = frame.shape
            detections.append({
                "module": "ThreatScan",
                "label": "Suspicious Motion (Match-Strike Pattern)",
                "confidence": min(motion_score / 20.0, 1.0),
                "bbox": (0, 0, w, h),
                "extra": {"motion_score": motion_score},
            })

        return detections

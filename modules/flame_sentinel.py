from pathlib import Path
from typing import List, Dict, Any


import numpy as np
from ultralytics import YOLO


class FlameSentinelEngine:
    def __init__(
        self,
        enabled: bool = True,
        weights_path: str = "models/yolov8n_flame_p.pt",
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
                f"Flame weights not found: {resolved_weights_path}\n"
                "Place the YOLOv8 flame weights at the path above (recommended filename: "
                "models/yolov8n_flame.pt)."
            )

        self.model = YOLO(str(resolved_weights_path))
        self.conf_thres = conf_thres


    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model.predict(frame, conf=self.conf_thres, verbose=False)
        detections: List[Dict[str, Any]] = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = r.names.get(cls_id, "flame")
                detections.append({
                    "module": "FlameSentinel",
                    "label": label,
                    "confidence": conf,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "extra": {},
                })

        return detections

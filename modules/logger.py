from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd


@dataclass
class Event:
    timestamp: datetime
    module: str
    label: str
    confidence: float
    bbox: tuple


class EventLogger:
    def __init__(self, max_events: int = 200):
        self.max_events = max_events
        self._events: List[Event] = []

    def log_event(self, det: Dict[str, Any]):
        ev = Event(
            timestamp=datetime.now(),
            module=det.get("module", ""),
            label=det.get("label", ""),
            confidence=float(det.get("confidence", 0.0)),
            bbox=det.get("bbox", (0, 0, 0, 0)),
        )
        self._events.append(ev)
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events :]

    def to_dataframe(self) -> pd.DataFrame:
        if not self._events:
            return pd.DataFrame(columns=["Time", "Module", "Label", "Confidence", "BBox"])
        data = [
            {
                "Time": e.timestamp.strftime("%H:%M:%S"),
                "Module": e.module,
                "Label": e.label,
                "Confidence": f"{e.confidence:.2f}",
                "BBox": e.bbox,
            }
            for e in reversed(self._events)
        ]
        return pd.DataFrame(data)

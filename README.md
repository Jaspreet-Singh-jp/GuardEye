# GuardEye: A Deep Learning Framework for Real-Time Gesture Interaction and Safety Surveillance

This repository implements the "GuardEye" major project as described in the synopsis. It provides a multimodal Deep Learning system that integrates:

- **Air-Canvas**: Gesture-based writing using MediaPipe 21-point hand landmark tracking.
- **FlameSentinel**: Fire/flame detection using a YOLOv8 detection head.
- **ThreatScan**: Prohibited-object (e.g., knife) and suspicious-motion (e.g., match striking) detection using YOLOv8 and simple temporal motion logic.

The system is packaged as a Streamlit "Security Command Centre" dashboard with dual-stream processing (live camera and uploaded video), a real-time AR-style HUD overlay, and a structured Live Event Log.

> Note: Model weights are **not** included in this repository. You must download or train YOLOv8 weights for flames and prohibited objects and update the paths in `modules/flame_sentinel.py` and `modules/threat_scan.py`.

## Tech Stack

- Python 3.10+
- Streamlit 1.30+
- streamlit-webrtc
- OpenCV (cv2)
- MediaPipe
- PyTorch 2.0+
- ultralytics (YOLOv8)
- NumPy, Pandas

## Project Structure

- `app.py` – Streamlit Command Centre UI and orchestration
- `modules/air_canvas.py` – Air-Canvas gesture tracking and drawing
- `modules/flame_sentinel.py` – Flame detection engine using YOLOv8
- `modules/threat_scan.py` – Prohibited-object and suspicious-motion engine
- `modules/hud.py` – AR Heads-Up Display overlay utilities
- `modules/logger.py` – Live Event Log and reporting support
- `modules/utils.py` – Shared utilities (frame preprocessing, threading helpers, etc.)

## Running the project

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place/adjust YOLOv8 weight files in the `models/` directory and update the `WEIGHTS_*` constants in the corresponding modules.
4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

5. Use the sidebar to select **Live Mode** or **Forensic Mode**, toggle individual AI modules (Gestures / Flame / Objects), and monitor detections in the HUD and Live Event Log.

import threading
import time
from typing import Literal

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from modules.air_canvas import AirCanvasEngine
from modules.flame_sentinel import FlameSentinelEngine
from modules.threat_scan import ThreatScanEngine
from modules.hud import HudRenderer
from modules.logger import EventLogger

# Simplified RTC config for local testing
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})


def main():
    st.set_page_config(page_title="GuardEye Command Centre", layout="wide")
    st.title("GuardEye: Real-Time Gesture Interaction and Safety Surveillance")

    mode: Literal["live", "forensic"] = st.sidebar.radio(
        "Input Mode", ["live", "forensic"], index=0, format_func=lambda x: x.title()
    )

    st.sidebar.markdown("### AI Brains (Triple-Threat)")
    # Check if we are currently in forensic mode
    is_forensic = (mode == "forensic")

    # Disable the Air Canvas checkbox if Forensic mode is active, and add a helpful tooltip!
    use_gestures = st.sidebar.checkbox(
        "Air-Canvas (Gestures)", 
        value=False, 
        disabled=is_forensic,
        help="Air Canvas is only available in Live Camera mode." if is_forensic else "Draw on the screen using a blue object."
    )
    
    # Safety catch: Force it to False internally if they switch to forensic mode while it was turned on
    if is_forensic:
        use_gestures = False
    use_flame = st.sidebar.checkbox("FlameSentinel (Fire)", value=True)
    use_objects = st.sidebar.checkbox("ThreatScan (Objects & Motions)", value=True)

    st.sidebar.markdown("### Event Log Controls")
    log_limit = st.sidebar.slider("Max events to display", 20, 500, 100, step=10)


    logger = EventLogger(max_events=log_limit)
    hud = HudRenderer()

    air_canvas = AirCanvasEngine(enabled=use_gestures)
    flame_engine = FlameSentinelEngine(enabled=use_flame)
    threat_engine = ThreatScanEngine(enabled=use_objects)

    # 🛑 NEW: Add a physical button to clear the canvas
    st.sidebar.markdown("### Air Canvas Controls")
    if st.sidebar.button("🧹 Erase Drawings"):
        air_canvas.clear_canvas()
    placeholder_frame = st.empty()
    placeholder_log = st.sidebar.empty()

    

    if mode == "forensic":
        video_file = st.sidebar.file_uploader("Upload video (MP4/AVI)", type=["mp4", "avi"])
        if video_file is not None:
            tpath = "temp_uploaded_video.mp4"
            with open(tpath, "wb") as f:
                f.write(video_file.read())
            run_forensic(tpath, air_canvas, flame_engine, threat_engine, hud, logger, placeholder_frame, placeholder_log)
        else:
            st.info("Upload a security clip to run forensic analysis.")
    else:
        run_live(air_canvas, flame_engine, threat_engine, hud, logger, placeholder_frame, placeholder_log)


def run_forensic(video_path, air_canvas, flame_engine, threat_engine, hud, logger, placeholder_frame, placeholder_log):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Unable to open uploaded video.")
        return

    progress = st.progress(0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = []
        if air_canvas.enabled:
            detections.extend(air_canvas.process_frame(frame))
        if flame_engine.enabled:
            detections.extend(flame_engine.process_frame(frame))
        if threat_engine.enabled:
            detections.extend(threat_engine.process_frame(frame))

        hud_frame = hud.draw(frame.copy(), detections)
        for det in detections:
            logger.log_event(det)

        # Fixed deprecation warning here: use_container_width
        placeholder_frame.image(hud_frame, channels="RGB", use_container_width=True)
        
        # FIX FOR FORENSIC MODE: Smooth dataframe instead of flickering table
        placeholder_log.dataframe(logger.to_dataframe(), use_container_width=True)

        progress.progress(min(current_frame / total_frames, 1.0))

    cap.release()
    progress.empty()
    st.success("Forensic scan complete.")


def run_live(air_canvas, flame_engine, threat_engine, hud, logger, placeholder_frame, placeholder_log):
    # Clear the manual image placeholder because WebRTC will render it natively
    placeholder_frame.empty() 
    
    lock = threading.Lock()
    shared_state = {"frame_count": 0, "last_detections": []}

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with lock:
            shared_state["frame_count"] += 1
            frame_id = shared_state["frame_count"]
            current_detections = shared_state["last_detections"]

        # 🛑 FIX 1: Run Air Canvas on EVERY frame so the drawings never vanish!
        # OpenCV is very fast, so this won't lag your computer.
        if air_canvas.enabled:
            air_canvas.process_frame(img)

        # 🛑 FIX 2: Only run the Heavy AI (YOLO Fire & Objects) every 5 frames
        if frame_id % 5 == 0:
            new_detections = []
            if flame_engine.enabled:
                new_detections.extend(flame_engine.process_frame(img))
            if threat_engine.enabled:
                new_detections.extend(threat_engine.process_frame(img))
            
            with lock:
                shared_state["last_detections"] = new_detections
                current_detections = new_detections

            # Only log new detections to the table
            for det in current_detections:
                logger.log_event(det)

        # Draw the bounding boxes for Fire/Objects
        hud_frame = hud.draw(img.copy(), current_detections)

        # Return the final frame with smooth drawings!
        return av.VideoFrame.from_ndarray(hud_frame, format="rgb24")

    st.markdown("Live camera feed with AR HUD overlays.")

    webrtc_ctx = webrtc_streamer(
        key="guardeye", 
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        # Standardized resolution for better stability
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        video_frame_callback=video_frame_callback,
    )

    # 🛑 THE FIX IS HERE 🛑 
    # Using .dataframe() instead of .table() stops the UI from flickering!
    while webrtc_ctx.state.playing:
        placeholder_log.dataframe(logger.to_dataframe(), use_container_width=True)
        time.sleep(0.5) 

if __name__ == "__main__":
    main()
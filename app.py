import sys
import os
import warnings
import streamlit as st
import cv2
#fic

# ---------------------
# Fix path for local deepface clone
# ---------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "deepface"))

# ✅ Correct import
from deepface import DeepFace

warnings.filterwarnings("ignore")

st.title("🎭 Real-time Emotion Detection with Anti-Spoofing")

cap = cv2.VideoCapture(0)

frame_placeholder = st.empty()
status_placeholder = st.empty()

frame_count = 0
process_every_n_frames = 5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("⚠️ Failed to access webcam.")
        break

    frame_count += 1

    try:
        if frame_count % process_every_n_frames == 0:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                anti_spoofing=True
            )

            if result and isinstance(result, list) and len(result) > 0:
                dominant_emotion = result[0]['dominant_emotion']
                spoofing_status = result[0].get("is_real", None)

                if spoofing_status is False:
                    status_placeholder.warning("🚨 Spoofing detected! (may be false alarm, check lighting)")
                else:
                    status_placeholder.success(f"😊 Detected Emotion: **{dominant_emotion}**")
            else:
                status_placeholder.info("⚠️ No face detected. Please face the camera.")

        frame_placeholder.image(frame, channels="BGR")

    except Exception as e:
        status_placeholder.error(f"Error: {str(e)}")
        continue

cap.release()

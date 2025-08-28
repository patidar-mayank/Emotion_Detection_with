import sys
import os
import warnings
import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ---------------------
# Fix path for local deepface clone
# ---------------------
#sys.path.append(os.path.join(os.path.dirname(_file_), "deepface"))

# ✅ Correct import
from deepface import DeepFace

warnings.filterwarnings("ignore")

st.title("🎭 Real-time Emotion Detection with Anti-Spoofing (WebRTC)")

# ---------------------
# Video Processor
# ---------------------
class EmotionProcessor(VideoProcessorBase):
    def _init_(self):
        self.frame_count = 0
        self.process_every_n_frames = 5
        self.last_status_text = "Waiting for face..."

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % self.process_every_n_frames == 0:
            try:
                result = DeepFace.analyze(
                    img,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv',
                    anti_spoofing=True
                )

                if result and isinstance(result, list) and len(result) > 0:
                    dominant_emotion = result[0]['dominant_emotion']
                    spoofing_status = result[0].get("is_real", None)

                    if spoofing_status is False:
                        self.last_status_text = "🚨 Spoofing detected!"
                        color = (0, 0, 255)
                    else:
                        self.last_status_text = f"😊 {dominant_emotion}"
                        color = (0, 255, 0)
                else:
                    self.last_status_text = "⚠ No face detected."
                    color = (255, 255, 0)

                # Draw status text on frame
                cv2.putText(
                    img,
                    self.last_status_text,
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            except Exception as e:
                self.last_status_text = f"Error: {str(e)}"
                cv2.putText(img, self.last_status_text, (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------------
# Run WebRTC
# ---------------------
webrtc_streamer(
    key="emotion-detector",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
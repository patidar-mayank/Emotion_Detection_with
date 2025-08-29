import warnings
import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from deepface import DeepFace
import time

warnings.filterwarnings("ignore")

st.title("🎭 Real-time Emotion Detection with Anti-Spoofing")

# ---------------------
# Video Processor
# ---------------------
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
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
                    face_box = result[0].get("region", {})
                    w, h = face_box.get("w", 0), face_box.get("h", 0)

                    # ✅ Ensure a valid face is present (ignore background false detections)
                    if w < 30 or h < 30:
                        self.last_status_text = "⚠ No person detected."
                    else:
                        dominant_emotion = result[0]['dominant_emotion']
                        spoofing_status = result[0].get("is_real", None)

                        if spoofing_status is False:
                            self.last_status_text = "🚨 Spoofing detected!"
                        else:
                            self.last_status_text = f"😊 {dominant_emotion}"
                else:
                    self.last_status_text = "⚠ No person detected."

            except Exception as e:
                self.last_status_text = f"Error: {str(e)}"

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------
# Run WebRTC
# ---------------------
ctx = webrtc_streamer(
    key="emotion-detector",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# ---------------------
# Live Text Display (below webcam)
# ---------------------
status_placeholder = st.empty()

if ctx.video_processor:
    while ctx.state.playing:  # keep updating while webcam is active
        status_placeholder.markdown(f"### {ctx.video_processor.last_status_text}")
        time.sleep(0.5)  # refresh every 0.5 sec

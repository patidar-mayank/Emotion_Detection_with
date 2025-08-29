# 🎭 Real-time Emotion Detection with Anti-Spoofing

This Streamlit app enables **real-time facial emotion detection** from webcam video, with built-in **anti-spoofing** to help distinguish real humans from spoofed faces (e.g., photos, videos, or masks).

It uses:
- [Streamlit](https://streamlit.io/) for the interactive web UI
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) for real-time webcam streaming
- [DeepFace](https://github.com/serengil/deepface) for face detection, emotion analysis, and anti-spoofing
- [OpenCV](https://opencv.org/) for image processing
- [PyAV](https://pyav.org/) for frame handling

## Features

- **Real-time** emotion detection from webcam video
- **Dominant emotion** displayed live (e.g., happy, sad, angry, etc.)
- **Anti-spoofing**: Warns if a fake/spoofed face is detected
- Lightweight and easy to use

## Demo
https://patidar-mayank-emotion-detection-with-app-mx61cb.streamlit.app


## Requirements

- Python 3.7+
- Streamlit
- streamlit-webrtc
- deepface
- opencv-python
- av

## Installation

```bash
pip install streamlit streamlit-webrtc deepface opencv-python av
```

## Usage

Save your script as `app.py` and run:

```bash
streamlit run app.py
```

Visit the local URL provided by Streamlit (usually http://localhost:8501).

## How it Works

- The webcam video is streamed to the browser.
- Every few frames, DeepFace analyzes the current frame to:
  - Detect a face and its bounding box
  - Predict the dominant emotion
  - Check if the face is real or spoofed
- The detected emotion and anti-spoofing status are displayed live.

## Notes

- The anti-spoofing feature depends on DeepFace's built-in support and may not be 100% accurate.
- For best results, use in a well-lit environment.

## Troubleshooting

- If you see errors about camera access, ensure your browser has permission to use the webcam.
- If DeepFace or OpenCV report issues, check your Python and package versions.

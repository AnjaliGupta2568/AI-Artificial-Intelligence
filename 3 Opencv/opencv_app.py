import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from rembg import remove
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Computer Vision Playground",
    page_icon="🎨",
    layout="wide"
)

# ------------------ DARK THEME ------------------
st.markdown("""
<style>
body {background-color: #0E1117;}
h1, h2, h3, h4 {color: #8A2BE2;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🎨 Computer Vision Playground</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Advanced OpenCV + YOLO Projects Hub</p>", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml"
)

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# ------------------------------------------------
# FUNCTIONS
# ------------------------------------------------

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 2)
    return img


def face_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        blur = cv2.GaussianBlur(face, (99,99), 30)
        img[y:y+h, x:x+w] = blur
    return img


def detect_face_eye(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
    return img


def detect_body(img):

    results = yolo_model(img)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = yolo_model.names[cls]

            if label == "person":   # Only detect pedestrians
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(img, "Person", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)

    return img




def sketch_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21,21), 0)
    inv_blur = 255 - blur
    sketch = cv2.divide(gray, inv_blur, scale=256.0)
    return sketch


def yolo_detection(img):
    results = yolo_model(img)
    return results[0].plot()


def remove_background(img):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    output = remove(img_pil)
    return np.array(output)


def download_button(img, filename):
    _, buffer = cv2.imencode(".png", img)
    st.download_button(
        label="⬇ Download Result",
        data=buffer.tobytes(),
        file_name=filename,
        mime="image/png"
    )

# ------------------------------------------------
# TABS
# ------------------------------------------------
tabs = st.tabs([
    "👤 Face",
    "👁 Face+Eyes",
    "📷 Webcam",
    "🚶 Pedestrian",
    "✏ Sketch",
    "🧠 YOLO",
    "🕶 Face Blur",
    "🌄 Background Removal"
])

# ------------------------------------------------
# GENERIC UPLOAD HANDLER
# ------------------------------------------------
def upload_section(process_function, download_name="result.png", grayscale=False):

    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg","png","jpeg"],
        key=f"upload_{download_name}"   # ✅ changed here
    )

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

        result = process_function(img.copy())

        with col2:
            st.subheader("Result")
            if grayscale:
                st.image(result, use_container_width=True, clamp=True)
            else:
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)

            download_button(result, download_name)


# ------------------------------------------------
# TAB IMPLEMENTATION
# ------------------------------------------------
with tabs[0]:
    upload_section(detect_face, "face_detected.png")

with tabs[1]:
    upload_section(detect_face_eye, "face_eye_detected.png")

with tabs[2]:
    st.subheader("📷 Real-Time Webcam Face Detection")

    class FaceDetector(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            result = detect_face(img)

            return result

    webrtc_streamer(
        key="real-time-face",
        video_transformer_factory=FaceDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True
    )


with tabs[3]:
    upload_section(detect_body, "body_detected.png")

with tabs[4]:
    upload_section(sketch_effect, "sketch.png", grayscale=True)

with tabs[5]:
    upload_section(yolo_detection, "yolo_detected.png")

with tabs[6]:
    upload_section(face_blur, "face_blur.png")

with tabs[7]:
    upload_section(remove_background, "background_removed.png")

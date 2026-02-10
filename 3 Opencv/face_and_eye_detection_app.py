import streamlit as st
import cv2
import numpy as np

# -------------------------------
# Face & Eye Detection Class
# -------------------------------
class FaceAndEyeDetection:
    def __init__(self, face_cascade_path, eye_cascade_path):
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

        if self.face_cascade.empty():
            st.error("❌ Face cascade not loaded")
        if self.eye_cascade.empty():
            st.error("❌ Eye cascade not loaded")

    def detect_faces(self, gray, frame):
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        return frame


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="Face & Eye Detection",
    page_icon="👁️",
    layout="centered"
)

st.title("👁️ Face & Eye Detection App")
st.write("Detect faces and eyes using Camera or Image Upload")

# Haar Cascade paths
face_cascade_path = r"C:\AVSCODE\Opencv\harscascade\haarcascade_frontalface_default.xml"
eye_cascade_path = r"C:\AVSCODE\Opencv\harscascade\haarcascade_eye.xml"

detector = FaceAndEyeDetection(face_cascade_path, eye_cascade_path)

# -------------------------------
# Mode Selection
# -------------------------------
mode = st.radio(
    "Select Input Mode",
    ("📷 Live Camera", "🖼️ Upload Image")
)

# -------------------------------
# CAMERA MODE
# -------------------------------
if mode == "📷 Live Camera":
    run = st.checkbox("▶ Start Camera")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to access webcam")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = detector.detect_faces(gray, frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        FRAME_WINDOW.image(frame)

    if not run:
        cap.release()
        st.info("📷 Camera stopped")

# -------------------------------
# IMAGE UPLOAD MODE
# -------------------------------
if mode == "🖼️ Upload Image":
    uploaded_file = st.file_uploader(
        "📤 Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        output = detector.detect_faces(gray, image)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        st.subheader("✅ Detection Result")
        st.image(output, use_container_width=True)
    else:
        st.info("📂 Please upload an image")

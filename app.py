# app.py
import streamlit as st
import cv2
import numpy as np
from face_recognition_model import save_to_database, recognize_face, load_known, ensure_db
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="AI Biometrics - Facial Recognition", layout="centered")
ensure_db()

st.title("AI-Based Biometrics — Facial Recognition System")
st.write("Register students using webcam and recognize them later.")

menu = st.sidebar.selectbox("Menu", ["Register (Webcam)", "Recognize (Webcam/File)", "Database"])

if menu == "Register (Webcam)":
    st.header("Register a new student (Webcam)")
    name = st.text_input("Student Name")
    reg_no = st.text_input("Registration Number")
    st.write("Take a clear frontal photo using your webcam. Ensure good lighting.")

    cam_file = st.camera_input("Click a photo")

    if cam_file is not None:
        # read image data as OpenCV BGR
        file_bytes = np.asarray(bytearray(cam_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Captured image", use_column_width=True)

        if st.button("Register Student"):
            if not name or not reg_no:
                st.error("Please enter name and registration number.")
            else:
                success = save_to_database(name.strip(), reg_no.strip(), img)
                if success:
                    st.success(f"Registered {name} ({reg_no}) ✅")
                else:
                    st.error("No face detected — make sure the face is clear and frontal.")

elif menu == "Recognize (Webcam/File)":
    st.header("Recognize student")
    tol = st.slider("Tolerance (lower = stricter)", min_value=0.25, max_value=0.75, value=0.5, step=0.01)

    input_mode = st.radio("Input mode", ["Webcam", "Upload file"])
    img = None

    if input_mode == "Webcam":
        cam_file = st.camera_input("Point camera at student's face")
        if cam_file is not None:
            file_bytes = np.asarray(bytearray(cam_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Captured", use_column_width=True)

    else:
        upload = st.file_uploader("Upload face image", type=["png", "jpg", "jpeg"])
        if upload is not None:
            file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded", use_column_width=True)

    if img is not None:
        if st.button("Recognize"):
            name, reg_no, dist = recognize_face(img, tolerance=tol)
            if name:
                st.success(f"Recognized: **{name}** ({reg_no}) — distance: {dist:.3f}")
                # show matched student image if available
                df = pd.read_csv("database/student_data.csv")
                match_row = df[df["reg_no"] == reg_no]
                if not match_row.empty:
                    img_path = match_row.iloc[-1].get("image_path", "")
                    try:
                        st.image(img_path, caption="Saved registered photo", width=240)
                    except Exception:
                        pass
            else:
                if dist is None:
                    st.error("No face detected in the input image.")
                else:
                    st.warning(f"No match found (best distance: {dist:.3f}). Try increasing tolerance or registering the student.")

elif menu == "Database":
    st.header("Registered Students (database)")
    try:
        df = pd.read_csv("database/student_data.csv")
        st.write(f"Total students: {len(df)}")
        st.dataframe(df[["name", "reg_no", "image_path"]])
    except Exception:
        st.info("No records found yet.")

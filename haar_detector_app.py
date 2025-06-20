import time
import cv2
import streamlit as st
from PIL import Image
import numpy as np
import tempfile

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://dda.ndus.edu/ddreview/wp-content/uploads/sites/18/2021/10/selfDriving.png");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()


def style_selected_elements():
    st.markdown("""
        <style>
        /* Light pink glow ONLY around the selectbox (dropdown) */
        div[data-baseweb="select"] {
            border: 2px solid #ffb6c1 !important;
            box-shadow: 0 0 10px #ffb6c1 !important;
            border-radius: 8px !important;
        }

        /* Green glowing border ONLY for the file uploader box */
        .stFileUploader {
            border: 2px solid #00ff88 !important;
            box-shadow: 0 0 12px #00ff88 !important;
            border-radius: 8px !important;
        }

        /* Optional: improve sidebar text contrast */
        .css-1v0mbdj, .stText, .stSelectbox {
            color: #ffffff !important;
            font-weight: bold;
            text-shadow: 0 0 2px black;
        }
        </style>
    """, unsafe_allow_html=True)

# Call it
style_selected_elements()



# Title
st.title("🚀 Haar Cascade Detection App")
detector_type = st.sidebar.selectbox("Choose Detection Type", ["Face", "Eyes", "Pedestrians", "Cars"])

cascade_paths = {
    "Face": r"C:\Users\vamsi\Downloads\haarcascade_frontalface_default.xml",
    "Eyes": r"C:\Users\vamsi\Downloads\haarcascade_eye.xml",
    "Pedestrians": r"C:\Users\vamsi\Downloads\haarcascade_fullbody.xml", 
    "Cars": r"C:\Users\vamsi\Downloads\cars.xml"
}

# For Face and Eyes - Image Upload
if detector_type in ("Face", "Eyes"):
    uploaded_image = st.sidebar.file_uploader("📷 Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:  
        image = Image.open(uploaded_image)
        img_array = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        classifier = cv2.CascadeClassifier(cascade_paths[detector_type])
        objects = classifier.detectMultiScale(gray, 1.1, 2)

        for (x, y, w, h) in objects:
            cv2.rectangle(img_array, (x,y), (x+w, y+h), (0,255,0), 2)

        st.image(img_array, caption="Detected Result", use_column_width=True) 

# For Pedestrians, Cars - Video Upload
else:
    uploaded_video = st.sidebar.file_uploader("🎞️ Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        classifier = cv2.CascadeClassifier(cascade_paths[detector_type])
        cap = cv2.VideoCapture(video_path)

        stframe = st.empty() # placeholder for video frames

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            time.sleep(0.05)
            frame = cv2.resize(frame, (960, 720))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # gray = cv2.resize(gray, None, fx=1.5, fy=1.5)
            # objects = classifier.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(60, 60))

            objects = classifier.detectMultiScale(gray, 1.1, 2)
            for (x, y, w, h) in objects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            stframe.image(frame, channels="BGR")

        cap.release()
    else:
        st.info("👈 Upload a video to begin.")







    


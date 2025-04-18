import streamlit as st
import tensorflow as tf
import keras
import numpy as np
import ultralytics
from ultralytics import YOLO
import cv2
import tempfile

def detect_keypoints(video_file):
    model = YOLO("yolo11n-pose.pt")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
      temp_file.write(video_file.getvalue())
      temp_file_path = temp_file.name


    # Pass the file-like object directly to the YOLO model
    results = model.track(source=temp_file_path, show=True, save=True, stream=True)

    video_keypoints = np.ndarray((128, 17, 2))
    for i, result in enumerate(results):
        normalized_keypoints = result.keypoints.xyn.cpu().numpy()
        video_keypoints[i] = normalized_keypoints
        result.show()  # display to screen

    x = np.zeros((128, 34))
    for i, frame in enumerate(video_keypoints):
        for j, keypoint in enumerate(frame):
            x[i][j*2] = keypoint[0]
            x[i][j*2+1] = keypoint[1]
    x = np.expand_dims(x, axis=0)
    return x

def main():
    st.title("My Streamlit App")
    st.write("Hello, world!")
    model = keras.saving.load_model("./bidirectional_model.keras")
    video_file = st.file_uploader("Video file", type="mp4")
    detect_keypoints_button = st.button("Detect Keypoints")

    if detect_keypoints_button and video_file is not None:
        # Pass the file-like object directly
        keypoints = detect_keypoints(video_file)
        st.write("Keypoints detected!")
        st.write(keypoints)
        y_pred = model.predict(keypoints)
        st.write("Prediction done!")
        st.write("Predicted values:")
        st.write(y_pred)

if __name__ == "__main__":
    main()
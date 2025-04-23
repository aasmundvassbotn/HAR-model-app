import streamlit as st
import tensorflow as tf
import keras
import numpy as np
import ultralytics
from ultralytics import YOLO
import cv2
import tempfile
import os

def detect_keypoints(video_file):
    model = YOLO("yolo11n-pose.pt")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
      temp_file.write(video_file.getvalue())
      temp_file_path = temp_file.name

    # Pass the file-like object directly to the YOLO model
    results = model.track(source=temp_file_path, show=True, save=True, stream=True)

    video_keypoints = np.ndarray((128, 17, 2))
    for i, result in enumerate(results):
        frame = result.orig_img  # Use the original frame from the result
        normalized_keypoints = result.keypoints.xyn.cpu().numpy()
        video_keypoints[i] = normalized_keypoints
        
    output_folder = "runs/pose/track"
    # Find the most recent file (assuming you want the last run's result)
    video_files = [f for f in os.listdir(output_folder) if f.endswith(('.mp4', '.avi'))]
    video_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_folder, x)), reverse=True)
        
    # Display video with keypoints
    if video_files:
        latest_video_path = os.path.join(output_folder, video_files[0])
        converted_video_path = os.path.join(output_folder, "converted_video.mp4")
        ffmpeg_command = f"ffmpeg -i {latest_video_path} -vcodec libx264 {converted_video_path}"
        os.system(ffmpeg_command)
        st.video(converted_video_path, format="video/mp4")
    else:
        st.warning("No output video found in the results folder.")

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
        y_pred = model.predict(keypoints)
        st.write("Predicted class:")
        y_class = np.argmax(y_pred, axis=1)
        LABELS = [
            "Jumping",
            "Jumping jacks",
            "Boxing",
            "Waving two hands",
            "Waving one hand",
            "Clapping",
        ]
        st.write(LABELS[y_class[0]])

if __name__ == "__main__":
    main()
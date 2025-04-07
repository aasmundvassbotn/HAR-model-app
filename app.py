import streamlit as st
import tensorflow as tf
import keras
import numpy as np
import ultralytics
from ultralytics import YOLO
import cv2

def detect_keypoints(video_path):
  model = YOLO("yolo11n-pose.pt")

  results = model.track(source=video_path, show=True, save=True, stream=True)

  video_keypoints = np.ndarray((128, 17, 2))
  for i, result in enumerate(results):
      normalized_keypoints = result.keypoints.xyn.cpu().numpy()
      video_keypoints[i] = normalized_keypoints
      result.show()  # display to screen
  return video_keypoints
  
def main():
  st.title("My Streamlit App")
  st.write("Hello, world!")
  model = keras.saving.load_model("./bidirectional_model.keras")
  video_file = st.file_uploader("Video file", type="mp4")
  detect_keypoints_button = st.button("Detect Keypoints")

  if detect_keypoints_button and video_file is not None:
    video_path = video_file.name
    keypoints = detect_keypoints(video_path)
    st.write("Keypoints detected!")
    st.write(keypoints)

    """ BLABLApredictions = model.predict(keypoints)
    st.write("Predictions:")
    st.write(predictions) """

if __name__ == "__main__":
  main()
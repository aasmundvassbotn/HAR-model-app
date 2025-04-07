import streamlit as st
import tensorflow as tf
import keras
import numpy as np
import ultralytics

""" def detect_keypoints(video_path):
  model = ultralytics.YOLO("yolov8n-pose.pt")

  cap = cv2.VideoCapture(video_path)

  keypoints_list = []

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    # Perform pose detection
    results = model(frame)

    # Extract keypoints from the results
    keypoints = results[0].keypoints.numpy()
    keypoints_list.append(keypoints)

  cap.release()
  return np.array(keypoints_list) """
  
def main():
  st.title("My Streamlit App")
  st.write("Hello, world!")
  model = keras.saving.load_model("bidirectional_model.keras")
  video_file = st.file_uploader("Video file", type="mp4")
  """ detect_keypoints_button = st.button("Detect Keypoints")
  if detect_keypoints_button and video_file is not None:
    video_path = video_file.name
    with open(video_path, "wb") as f:
      f.write(video_file.read())
    keypoints = detect_keypoints(video_path)
    st.write("Keypoints detected!")
    st.write(keypoints)

    # Predict using the model
    predictions = model.predict(keypoints)
    st.write("Predictions:")
    st.write(predictions) """

if __name__ == "__main__":
  main()
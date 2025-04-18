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
    st.video(temp_file_path)

    # Pass the file-like object directly to the YOLO model
    results = model.track(source=temp_file_path, show=True, save=True, stream=True)

    cap = cv2.VideoCapture(temp_file_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    video_keypoints = np.ndarray((128, 17, 2))
    for i, result in enumerate(results):
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = result.keypoints.xy.cpu().numpy()
        normalized_keypoints = result.keypoints.xyn.cpu().numpy()
        video_keypoints[i] = normalized_keypoints
        print(f"Keypoints: {keypoints}")
        
        for keypoint in keypoints:
            print(keypoint)
            print(type(keypoint))
            print(keypoint.shape)
            print(keypoint[0])
            print(keypoint[1])
            print(type(keypoint[0]))
            print(type(keypoint[1]))

            x, y = int(keypoint[0]), int(keypoint[1])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        out.write(frame)

    cap.release()
    out.release()

    st.video(out_path)

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
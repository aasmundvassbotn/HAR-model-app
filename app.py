import streamlit as st
import tensorflow as tf
import keras
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import subprocess
import shutil

def detect_keypoints(video_file):
    model = YOLO("yolo11n-pose.pt")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
      temp_file.write(video_file.getvalue())
      temp_file_path = temp_file.name

    # Pass the file-like object directly to the YOLO model
    results = None
    results = model.track(source=temp_file_path, save=True, stream=True, save_dir="runs/pose/track")
    num_frames = len(list(results))
    video_keypoints = np.ndarray((num_frames, 17, 2))
    for i, result in enumerate(results):
        frame = result.orig_img  # Use the original frame from the result
        normalized_keypoints = result.keypoints.xyn.cpu().numpy()
        video_keypoints[i] = normalized_keypoints

    x = np.zeros((num_frames, 34))
    for i, frame in enumerate(video_keypoints):
        for j, keypoint in enumerate(frame):
            x[i][j*2] = keypoint[0]
            x[i][j*2+1] = keypoint[1]
        st.write(f"Frame {i+1}: {x[i]}")
    x = np.expand_dims(x, axis=0)
    return x

def display_video():
    output_folder = "runs/pose/track"
    # Find the most recent file (assuming you want the last run's result)
    video_files = [f for f in os.listdir(output_folder) if f.endswith(('.avi'))]
    video_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_folder, x)))
    if video_files:
        latest_video_path = os.path.join(output_folder, video_files[0])
        converted_video_path = os.path.join(output_folder, "converted_video.mp4")
        ffmpeg_command = f"ffmpeg -i {latest_video_path} -vf scale=iw/4:ih/4 -vcodec libx264 -preset ultrafast {converted_video_path}"
        try:
            subprocess.run(ffmpeg_command, shell=True, check=True, timeout=45)
        except subprocess.TimeoutExpired:
            st.error("FFmpeg processing timed out.")
        except subprocess.CalledProcessError:
            st.error("FFmpeg command failed.")
        
        st.video(converted_video_path, format="video/mp4")
    else:
        st.warning("No output video found in the results folder.")

def main():
    st.title("Human Action Recognition App")
    st.write("Hello! This is a human action recognition app using a pre-trained pose detection model and a trained LSTM model. You can find the code in the GitHub repository: https://github.com/aasmundvassbotn/HAR-model-app")
    st.write("The model we trained is a unidirectional LSTM model. The model was trained on a subset of the Berkley MHAD dataset. The dataset we used can be found here: https://github.com/stuarteiffert/RNN-for-Human-Activity-Recognition-using-2D-Pose-Input?tab=readme-ov-file#dataset-overview. Our model is trained to classify the following actions: Jumping, Jumping jacks, Boxing, Waving two hands, Waving one hand and Clapping.")
    st.write("Upload a video file to detect keypoints and classify the action. The pose-detection model used is YOLO nano which is the smallest model in the YOLO family. This model is not that robust, so it usually performs poorly in bad lighting, low resolutions etc. Note that since this app is hosted on Streamlit Cloud, all the processes are run on a CPU and not a GPU. This means that the pose detection process can take a while, depending on the length of the video. It is not recommended to upload videos longer than 5 seconds.")
    st.write("If you dont feel like recording a video yourself. Feel free to use any of our test videos: https://drive.google.com/drive/folders/1zQBNdTXX8kwFk0NK7-4C2XbA22LZOnLD?usp=drive_link")

    if st.button("Reset session"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.session_state.clear()
        cleanup()
        st.rerun()

    st.write("Please press the button above to reset the session state. This is because the app is hosted on Streamlit Cloud and the session state is not reset automatically. After every run, press the x on the video uploaded, press this button again and refresh the page.")
    video_file = st.file_uploader("Video file", type="mp4")
    detect_keypoints_button = st.button("Detect Keypoints")

    if detect_keypoints_button and video_file is not None:
        # Pass the file-like object directly
        keypoints = detect_keypoints(video_file)
        model = keras.saving.load_model("./GRU_model_2.keras")
        y_pred = model.predict(keypoints)
        st.success("Success ✅")
        #display_video()
        st.write("Above is the video you uploaded after the keypoint detection process. If you spot any errors in the keypoints displayed, this is because of the YOLO model used. Had a more advanced model like the small, medium or large been used the results would have been better. However, these models are too large to be used in this app. If errors are present this can affect the classification result.")
        y_class = np.argmax(y_pred, axis=1)
        LABELS = [
            "JUMPING",
            "JUMPING_JACKS",
            "BOXING",
            "WAVING_2HANDS",
            "WAVING_1HAND",
            "CLAPPING_HANDS"
        ]
        st.write("Class predicted: ")
        st.markdown(f":blue-background[**{LABELS[y_class[0]]}**]")
        st.write("Confidence: ")
        st.markdown(f":blue-background[**{y_pred[0][y_class[0]]:.2f}%**]")

def init():
    if not os.path.exists("runs/pose/track"):
        os.makedirs("runs/pose/track")

def cleanup():
    shutil.rmtree("runs/pose/track", ignore_errors=False)

if __name__ == "__main__":
    main()

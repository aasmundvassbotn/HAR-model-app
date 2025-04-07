import streamlit as st
import tensorflow as tf
import keras
import numpy as np

def main():
    st.title("My Streamlit App")
    st.write("Hello, world!")
    keras.saving.load_model("bidirectional_model.keras")
    video_file = st.file_uploader("Video file", type="mp4")

if __name__ == "__main__":
    main()
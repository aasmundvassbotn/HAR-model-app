import streamlit as st



def main():
    st.title("My Streamlit App")
    st.write("Hello, world!")
    video_file = st.file_uploader("Video file", type="mp4")

if __name__ == "__main__":
    st.main()
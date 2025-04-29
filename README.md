# HAR model app
Includes training a HAR (Human Action Recognition) model and a Streamlit app.
## Table of Contents
1. [Project structure](#project-stucture)
2. [Dataset](#dataset)
3. [Model Training](#model-training)
4. [Streamlit App](#streamlit-app)
5. [License](#license)

## Project structure
All notebooks that were used to experiment with the dataset and train the model itself are found in this repository: https://github.com/aasmundvassbotn/HAR_model_training. We seperated them into seperate git-repos due to them causing issues with the Streamlit Cloud. All notebooks were run in Google Colab and Google Drive were used to store data. 
All files in this project are files for the webapps functionality. "app.py" is the app itself.

## Model Training (Notebooks)
All model training is performed in the notebookes. The final model is included in this repo as a saved keras model.

## Dataset
The original dataset can be found here: https://github.com/stuarteiffert/RNN-for-Human-Activity-Recognition-using-2D-Pose-Input?tab=readme-ov-file#dataset-overview
In the notebooks you can find one named "preprocessing_dataset.ipynb". This notebook is used to process the original dataset into numpy arrays that were used to train the model used. We normalized the data, removed the 18th keypoint, augmented data and balanced the distribution of the classes.

## Streamlit App
The Streamlit App can be accessed here: [https://harvideomodel-qvmgers7vkxxixgbwp9plc.streamlit.app/](https://har-model-app-cjh3xj6pzh9n2hdlswbqd9.streamlit.app/)
The app is limited to the free resources provided by Streamlit Cloud. Therefore the app is rather slow. In further work, runtime can be improved:
- Quantize and prune GRU-model thus making the prediction itself faster
- Cut every other (or more) frame from the video submitted. For example most mobile phone cameras capture videos at 30fps. This many frames are redundant and a lot of computation may be saved by editing the videos before it is passed to the YOLO-model.

## Further work
In additon to the two optimalizations that can be made to make the app run smoother we could improve the model itself. We acheived a high accuracy in this project so the next logical step would be to add more classes and see how well our model performs. It can be downloaded using the pythorch/vison dataset loader (https://github.com/pytorch/vision). This dataset contains another 6 classes. If we were to add even more classes it gets more complicated. In order to do this we would either have to find a similar dataset and transform the data to match ours (probably a lot of processing involved) or alternativly we could construct our own dataset using the YOLO model with any HAR-dataset (something better than Google Colab free recommended). 

## License
This project is licenced under the MIT licence.  

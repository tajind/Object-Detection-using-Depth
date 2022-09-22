# Introduction

This project focuses on building, traning, and then predicting objects based on their depth using a Machine Learning aproach. The project is able to create a dataset using the example video provided. Using that dataset, it can train a machine learning model which is created using TensorFlow to detect the objects in a video that is unseen by the program.

# Getting up and running

The `dataset_creation.py` file is responsible for creating the dataset from a video file by taking its depth frame by frame generating a dataset.

The `model.py` file hosts the TensorFlow machine learning model which is used by the creation file and the test file.

The `test.py` file is used to detect objects in an unseen video.

<br>
 The required libraries are listed in `requirements.txt` file. 

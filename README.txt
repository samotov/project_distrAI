+------------+
| OVERVIEW:  |
+------------+


This repository consist of two folders:
    - The computer_vision folder which contains all the files for the computer vision tasks
    - The RL folder which contains all the files for training are agent using reinforcement learning


+-------------------------+
| FOLDER: computer_vision |
+-------------------------+

This folder is used for 4 main tasks:
    1) The conversion of datasets to the YOLO format
    2) The filtering of YOLO datasets
    3) Training and testing of the YOLO detection model
    4) Training and testing of a regression model to predict 3D bounding boxes from 2D boundingboxes

All the python files to perform these function are visible in the main object_localization folder. In this folder there are also there other classes called:
    1) BoundingBox3D.py: used for handeling and drawing 3D boundingboxes
    2) BoundingBoxTransformer.py: used for transforming different 3D Boundingboxes initializations
    3) BoundingBoxRegressionDataset.py: the dataset used for the training of the regression model

In the main folders there are also subfolders:
    1) data_converters: which contain different classes that are used to convert different datatypes into the YOLO format
    2) datasets: the folder where all the datasets for training are stored
    3) models: this folder contains all the model classes and the saved weights of previous well performing models.


+------------+
| FOLDER: RL |
+------------+



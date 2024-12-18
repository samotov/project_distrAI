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

This folder contains all files used for training the RL agent:
    1) agents: contains all files needed to run Basic Agent which was used for steering.
    2) mock: simple environment for training and testing 
    3) models: contains trained weights for computer vision which are used as input for the RL
    4) results: results from trainings
    5) tools: usefull modules necessary for the Basic Agent to work
    *) CarlaEnv.py: the Carla environment converted into a Gym environment used for training
    *) depth_conversion.py: functions used for extracting distance from RGB image
    *) RL_agent.py: the SAC agent and training/evaluation methods
    *) spawn.py: script used for spawning and removing entities from the Carla environment.


+-------------------+
| FOLDER: ACC agent |
+-------------------+

This folder contains everything needed to implement our ACC agent into Carla example scripts
    1) agents: contains all files needed to run Basic Agent and the ACC agent which is based on it.
    2) controllers: contains the class which will implement the RL agent into the ACC agent.
    3) models: contains trained weights for computer vision and RL
    4) sensors: contains all modules implementing and controlling sensors
    5) tools: usefull modules necessary for the Basic Agent to work


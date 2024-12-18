import torch
from ultralytics import YOLO
from models.ObjectLocalizationModel import ObjectLocalizationModel
import numpy as np


class ObjectDetector():
    """
    Class that will detect and classify objects that are visible through RGB images. 
    Afterwards, it will also determine the distance to these objects using depth images.
    Finally, it will return an ordered list, based on the distance, of tuples (bounding_box, distance, type_of_object).
        
    # TODO: extract RGB and Depth images
    # TODO: run them through the neural network
    # TODO: Get list of tuples (distance, type vehicle)
    # TODO: go through list and check for the closest distance in front of the car

    """
    def __init__(self):
        self.model = None
        self.label_dict = None
        self.data: list[tuple] = None

        self.model = self._set_model("./sensors/VehiclesAndTrafficlights.pt")
    
    def _set_model(self, model_weights_path):
        return ObjectLocalizationModel(model_weights_path)
        

    def _extract_bounding_boxes(self, RGB_image, depth_image):
        """
        Use NN to extract 2D bounding boxes and type of object observed. 
        
        Input: RGB image
        Output: List of tuples (type, bounding box)
        """
        # Create empty data list
        data = []
        
        # Get results from image classifier
        classified_image = self.model.forward(RGB_image)

        # Extract usefull information
        if self.label_dict is None:
            self.label_dict = classified_image[0].names

        labels = classified_image[0].boxes.cls.to('cpu').tolist()
        bounding_box_list = classified_image[0].boxes.xyxy.to('cpu').tolist()

        # Loop over bounding boxes and process each one
        for i, bounding_box in enumerate(bounding_box_list):
            # Get the label
            label = int(labels[i])
            class_name = self.label_dict[label] 
            
            # Get the median distance to the object
            distance = self._calculate_distance(depth_image, bounding_box)

            # Add information to data list
            data.append((class_name, distance, bounding_box))
        
        return data

    def _calculate_distance(self, depth_image, bounding_box):
        """
        Use combination of depth image and 2D bounding box to get average distance of the object.
        
        Input: list of tuples (type, bounding box)
        Output: list of tuples (type, bounding box, distance)
        """
        # Get the region of the depth image corresponding to the bounding box
        image_width = depth_image.shape[0]
        image_height = depth_image.shape[1]

        x_min, y_min, x_max, y_max = map(int, bounding_box)
        cropped_depth = depth_image[max(0, y_min):min(image_height, y_max), max(0, x_min):min(image_width, x_max)]

        # Find the median depth value and its corresponding coordinate
        if cropped_depth.size > 0:
            median_depth = np.median(cropped_depth)
            
            # Identify the coordinate closest to the median depth value
            abs_diff = np.abs(cropped_depth - median_depth)
            median_coord = np.unravel_index(abs_diff.argmin(), cropped_depth.shape)

            # Compute distance in meters
            B, G, R = depth_image[
                median_coord[0] + max(0, y_min), median_coord[1] + max(0, x_min), :3
            ]
            normalized = (int(R) + int(G) * 256 + int(B) * 256 * 256) / (256 **3 - 1)
            return 1000 * normalized
        else:
            return float('nan')

    def _order_list(self, data_list):
        """
        Order the list of tuples based on the distance to get the closest object first.

        Input: list of tuples (type, bounding box, distance)
        Output: ordered list of tuples (type, bounding box, distance) 
        """
        data_list.sort(key=lambda x: x[1])

    def extract_objects_from_surrounding(self, RGB_image, depth_image):
        """
        Extract type of object, bounding box and distance to object from RGB and Depth images.
        Result is returned in a sorted list (based on distance) of tuples (type, bounding box, distance).

        Input: RGB and Depth image.
        Output: ordered list of tuples (type, bounding box, distance)
        """
        list_of_type_dist_bb = self._extract_bounding_boxes(RGB_image, depth_image)
        self._order_list(list_of_type_dist_bb)
        return list_of_type_dist_bb





    

    
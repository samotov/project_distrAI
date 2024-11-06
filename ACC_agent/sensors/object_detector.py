import torch.nn as nn

def ObjectDetector(nn.Module):
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
        pass

    def _extract_bounding_boxes(self, RGB_image):
        """
        Use NN to extract 2D bounding boxes and type of object observed. 
        
        Input: RGB image
        Output: List of tuples (type, bounding box)
        """
        return []

    def _calculate_distance(self, depth_image, list_of_type_bb):
        """
        Use combination of depth image and 2D bounding box to get average distance of the object.
        
        Input: list of tuples (type, bounding box)
        Output: list of tuples (type, bounding box, distance)
        """
        return []

    def _order_list(self, list_of_type_bb_dist)
        """
        Order the list of tuples based on the distance to get the closest object first.

        Input: list of tuples (type, bounding box, distance)
        Output: ordered list of tuples (type, bounding box, distance) 
        """
        return []

    def extract_objects_from_surrounding(self, RGB_image, depth_image):
        """
        Extract type of object, bounding box and distance to object from RGB and Depth images.
        Result is returned in a sorted list (based on distance) of tuples (type, bounding box, distance).

        Input: RGB and Depth image.
        Output: ordered list of tuples (type, bounding box, distance)
        """
        list_of_type_bb = self._extract_bounding_boxes(RGB_image)
        list_of_type_bb_dist = self._calculate_distance(depth_image, list_of_type_bb)
        ordered_list_of_type_bb = self._order_list(list_of_type_bb_dist)
        return ordered_list_of_type_bb





    

    
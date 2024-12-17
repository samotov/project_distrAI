from data_converters import CapturedDataConverter
import numpy as np
import argparse

def convert_data(input_dataset, output_dataset, last_index):
    # This map contains all the info as: label: [color, dilation_kernel_size, min_area_boundingbox]
    class_color_info_map = {'car': [np.array([0, 0, 142]), 10, 400],
                'motorcycle': [np.array([0, 0, 230]), 10, 300],
                'truck': [np.array([0, 0, 70]), 20, 400],
                'pedestrian': [np.array([220, 20, 60]), 5, 200],
                'bus': [np.array([0, 60, 100]), 20, 400],
                'traffic signs': [np.array([220, 220, 0]), 1, 150],
                'traffic light': [np.array([250, 170, 30]), 1, 60]
                }
    
    classes = ['car', 'motorcycle', 'truck', 'pedestrian', 'bus', 'traffic signs', 'traffic light green', 'traffic light orange', 'traffic light red', 'traffic light unimportant']
    data_converter = CapturedDataConverter.CapturedDataConverter(input_dataset, output_dataset, classes, class_color_info_map)

    next_index = data_converter.convert_data(last_index, 0.8)
    print('If you want to convert more images in the same folder use a next index of ' + str(next_index) + ' to make sure no images get overwritten.')

parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument("input_location", type=str, help="the location of the input dataset")
parser.add_argument("output_location", type=str, help="the location of the output dataset")
parser.add_argument("last_index", type=int, help="the last index of the image that has been converted")

args = parser.parse_args()
input_dataset = args.input_location
output_dataset = args.output_location
last_index = args.last_index
convert_data(input_dataset, output_dataset, last_index)
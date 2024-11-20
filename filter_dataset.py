from data_converters import YoloDatasetFilterer
import argparse

# Filter dataset keys:
# ENTER: Save current boundingboxes and image and go to the next image
# BACKSPACE: Don't save the current boundingboxes and images and go to the next image
# ARROW_LEFT: Go to the previous image (if you made a mistake) if it's save it should get overwritten by the next data
# ARROW_UP: Keep this boundingbox and go to the next boundingbox in the image.
#           This bounding box will be saved when all boundingboxes are iterated or you press ENTER)
# ARROW_DOWN: Remove this boundingbox and go to the next boundingbox in the image.
#           This bounding box will not be saved when all boundingboxes are iterated or when you press ENTER)

# Fileter dataset arguments:
# Start_index: If you wnat to start somewhere in the middle of the dataset you can change this to whatever you want if not use 0
# filter_start_train_index: If you don't want to override training images from a previous run use the index for the next image otherwise use 0
# filter_start_val_index: If you don't want to override validation images from a previous run use the index for the next image otherwise use 0
# You can find these values from the previous run printed in the terminal!!

parser = argparse.ArgumentParser(description="image filtering argument parsers ")
parser.add_argument("dataset_input_yaml", type=str, help="the location and name of the input yaml file in yolo format")
parser.add_argument("dest_dir", type=str, help="the name of the dataset folder of the output dataset and yaml file")
parser.add_argument('start_index', type=int, help="the image index where we start")
parser.add_argument('filter_start_train_index', type=int, help="the image index for the next train image")
parser.add_argument('filter_start_val_index', type=int, help="the image index for the next validation image")

args = parser.parse_args()

source_yaml_file = args.dataset_input_yaml
dest_dir = args.dest_dir
start_index = args.start_index
filter_start_train_index = args.filter_start_train_index
filter_start_val_index = args.filter_start_val_index

dataset_filterer = YoloDatasetFilterer.YoloDatasetFilterer(source_yaml_file, dest_dir)
dataset_filterer.filter_data(start_index, filter_start_train_index, filter_start_val_index)

print("Finished processing all images.")

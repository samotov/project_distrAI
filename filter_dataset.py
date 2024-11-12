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

parser = argparse.ArgumentParser(description="image filtering argument parsers ")
parser.add_argument("dataset_input", type=str, help="the location of the input dataset in yolo format")
parser.add_argument("dataset_ouput", type=str, help="the location of the ouput dataset in yolo format")
parser.add_argument('start_index', type=int, help="the image index where we start")

args = parser.parse_args()

source_dir = args.dataset_input
dest_dir = args.dataset_ouput
start_index = args.start_index

dataset_filterer = YoloDatasetFilterer.YoloDatasetFilterer(source_dir, dest_dir, ['car', 'motorcycle', 'truck', 'pedestrian', 'bus'])

dataset_filterer.filter_data(start_index, (960, 540))

print("Finished processing all images.")

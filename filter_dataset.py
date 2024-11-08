from data_converters import YoloDatasetFilterer
import argparse

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

from data_converters import YoloDatasetFilterer
import argparse

parser = argparse.ArgumentParser(description="image filtering argument parsers ")
parser.add_argument("dataset_input", type=str, help="the location of the input dataset in yolo format")
parser.add_argument("dataset_ouput", type=str, help="the location of the ouput dataset in yolo format")
parser.add_argument('start_index', type=int, help="the image index where we start")
parser.add_argument('image_refresh_rate', type=float, help='the speed at which the images disapear in seconds')

args = parser.parse_args()

source_dir = args.dataset_input
dest_dir = args.dataset_ouput
start_index = args.start_index
image_refresh_rate = args.image_refresh_rate

dataset_filterer = YoloDatasetFilterer.YoloDatasetFilterer(source_dir, dest_dir, ['Car', 'Pedestrian', 'Van', 'Cyclist', 'Truck', 'Misc', 'Tram', 'Dontcare'])

dataset_filterer.filter_data(start_index, image_refresh_rate)

print("Finished processing all images.")

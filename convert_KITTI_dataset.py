import torchvision.datasets
import yaml
from data_converters import KittiDataConverter
import argparse

parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument("input_location", type=str, help="the location of the input dataset")
parser.add_argument("output_location", type=str, help="the location of the output dataset")
parser.add_argument("download_needed", type=int, help="is the dataset already downloaded or not (1 or 0)")
parser.add_argument("training_amount", type=float, help="The percentage of traindata in the dataset between 0 and 1")

args = parser.parse_args()
input_dataset = args.input_location
output_dataset = args.output_location
download_needed = args.download_needed
training_amount = args.training_amount

if download_needed == 0:
    download = False
else:
    download = True

kitti_dataset = torchvision.datasets.Kitti(input_dataset, download=download)
type_map = ['Car', 'Pedestrian', 'Van', 'Cyclist', 'Truck', 'Misc', 'Tram', 'Dontcare']
data_converter = KittiDataConverter.KittiDataConverter(input_dataset, output_dataset, type_map, kitti_dataset)

data_converter.convert_data(training_amount)



        

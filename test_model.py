from models import ObjectLocalizationModel
# import argparse

# parser = argparse.ArgumentParser(description="Argument parsers ")
# parser.add_argument("dataset", type=str, help="the location of the dataset")

# args = parser.parse_args()

# dataset = args.dataset

dataset = 'datasets/KITTI/Kitti/raw/testing/image_2'

yolomodel = ObjectLocalizationModel.ObjectLocalizationModel('best.pt')

yolomodel.visualize_data(20, 'kitti_transformed')

#yolomodel.testYoloModel(dataset)
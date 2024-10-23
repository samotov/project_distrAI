import ObjectLocalizationModel
import DataConverter
import numpy as np
import argparse

def convert_data(dataset):
    # This map contains all the info as: label: [color, dilation_kernel_size, min_area_boundingbox]
    class_color_info_map = {'car': [np.array([0, 0, 142]), 10, 250],
                'motorcycle': [np.array([0, 0, 230]), 10, 250],
                'truck': [np.array([0, 0, 70]), 10, 250],
                'pedestrian': [np.array([220, 20, 60]), 8, 200],
                'traffic signs': [np.array([220, 220, 0]), 1, 150],
                'traffic lights': [np.array([250, 170, 30]), 1, 150]}
    
    data_converter_noon = DataConverter.DataConverter("captured_data/noon", dataset, class_color_info_map)
    data_converter_cloudy_noon = DataConverter.DataConverter("captured_data/cloudy_noon", dataset, class_color_info_map)
    data_converter_cloudy_night = DataConverter.DataConverter("captured_data/cloudy_night", dataset, class_color_info_map)

    next_index = data_converter_noon.convert_data(1, 0.8)
    next_index = data_converter_cloudy_noon.convert_data(next_index, 0.8)
    data_converter_cloudy_night.convert_data(next_index, 0.8)

def train_model(number_of_epochs, dataset):
    yolomodel = ObjectLocalizationModel.ObjectLocalizationModel()

    yolomodel.train_model(dataset + '.yaml', number_of_epochs, 16)

    yolomodel.testYoloModel()


def main(args):
    number_of_epochs = args.number_of_epochs
    dataset = args.dataset

    if args.data_conversion == 1:
        convert_data(dataset)
    
    train_model(number_of_epochs, dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="number of epochs argument parsers ")
    parser.add_argument("dataset", type=str, help="the location of the dataset")
    parser.add_argument("dataset_conversion", type=int, help="0 if no data conversion is needed 1 if data conversion is needed. Dataset automatically becomes the location of the converted data")
    parser.add_argument("number_of_eposchs", type=int, help="number of epochs")
    
    args = parser.parse_args()
    main(args)
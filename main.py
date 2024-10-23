import RegressionModel
import ObjectLocalizationModel
import DataConverter
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np


def main():
    # This map contains all the info as: label: [color, dilation_kernel_size, min_area_boundingbox]
    class_color_info_map = {'car': [np.array([0, 0, 142]), 10, 250],
                'motorcycle': [np.array([0, 0, 230]), 10, 250],
                'truck': [np.array([0, 0, 70]), 10, 250],
                'pedestrian': [np.array([220, 20, 60]), 8, 200],
                'traffic signs': [np.array([220, 220, 0]), 1, 150],
                'traffic lights': [np.array([250, 170, 30]), 1, 150]}
    
    data_converter_noon = DataConverter.DataConverter("captured_data/noon", "test2", class_color_info_map)
    data_converter_cloudy_noon = DataConverter.DataConverter("captured_data/cloudy_noon", "test2", class_color_info_map)
    data_converter_cloudy_night = DataConverter.DataConverter("captured_data/cloudy_night", "test2", class_color_info_map)

    next_index = data_converter_noon.convert_data(1, 0.8)
    next_index = data_converter_cloudy_noon.convert_data(next_index, 0.8)
    data_converter_cloudy_night.convert_data(next_index, 0.8)

    yolomodel = ObjectLocalizationModel.ObjectLocalizationModel()
    
    results = yolomodel.train_model('test.yaml', 3, 16)
    results.show()

    yolomodel.test()
    
    

if __name__ == "__main__":
    main()
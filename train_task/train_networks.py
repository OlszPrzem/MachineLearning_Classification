import os
from datetime import datetime
import torch
import pandas as pd
# import parallelTestModule

print(torch.cuda.is_available())

# import sys
# sys.path.insert(1, 'C:\\Users\\przemeko\\Desktop\\hemi\\Hemitech_AI\\Clever_ES23')
# sys.path.append("..")


main_script = "C:\\Users\\przemeko\\Desktop\\classification\\train\\train.py"

path_dataset_data = None

path_dataset_csv = "C:\\Users\\przemeko\\Desktop\\classification\\data\\classes_dataset.csv" 
classes_names_path = "C:\\Users\\przemeko\\Desktop\\classification\\data\\classes.txt"



whole_network = True
pretrained = True

list_network_to_test = [
    "efficientnet_b0",
    # "efficientnet_b1",
    # "mobilenet_v3_large",
    "mobilenet_v3_small",
    "mobilenet_v2",
    "resnet18",
    "resnet34",
    # "my_efficientnet_b0",
]

if __name__ == "__main__":
    for network in list_network_to_test:
        time_start = datetime.today().strftime('%Y-%m-%d_%H-%M')
        name_experiment = time_start + network
        os.system('python {} --network {} --pretrained {} --whole_network {} --name_experiment {} --dataset_folder {} --dataset_csv {} --classes_names {}'.format(main_script, network, pretrained, whole_network,name_experiment, path_dataset_data, path_dataset_csv, classes_names_path))




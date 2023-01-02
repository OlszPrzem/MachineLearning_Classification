import sys

sys.path.append("..") 
sys.path.append("...") 

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

from utils.eval_function import conf_matrix
import utils.custom_function as mcf



path_csv = "C:\\Users\\przemeko\\Desktop\\classification\\data\\classes_dataset_test.csv"
path_root = None
classes_names_path = "C:\\Users\\przemeko\\Desktop\\classification\\data\\classes.txt"


path_model = "C:\\Users\\przemeko\\Desktop\\classification\\train_task\\models\\2023-01-02_20-43efficientnet_b0_fold0.pt"


model = torch.load(path_model)

albumentations_transform = A.Compose([
    A.Resize(256, 256), 
    A.RandomCrop(224, 224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2()
])


if __name__ == "__main__":

    class_names_test = list(pd.read_csv(classes_names_path, header=None)[0])


    albumentations_dataset = mcf.MyDataset_A(path_csv, path_root)

    albumentations_dataset = mcf.MapDataset_A(albumentations_dataset, A.Compose([A.Resize(224, 224),
            
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                ),
            ToTensorV2(), 
            ]))
                    
    dataloaders_test = torch.utils.data.DataLoader(albumentations_dataset, batch_size=16, shuffle=True, num_workers=0)

                
    dataset_sizes_test = len(albumentations_dataset)

    y_true, y_pred = conf_matrix(dataloaders_test, model, albumentations_dataset, class_names_test)
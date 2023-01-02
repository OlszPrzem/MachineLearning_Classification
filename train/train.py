import torch
print(torch.cuda.is_available())

import sys
import pandas as pd
# sys.path.insert(1, 'D:hemi\\Hemitech_AI\\Clever_ES23')
# sys.path.insert(1, 'C:\\Users\\przemeko\\Desktop\\hemi\\Hemitech_AI\\Clever_ES23\\my_package\\')
sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# import numpy as np
# import torchvision
# from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
# import time
# import os
# import copy
# from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import cv2
from numpy import uint8
from torch.utils.tensorboard import SummaryWriter

plt.ion()   # interactive mode

import cnn_networks as netfunc

import utils.custom_function as mcf
import argparse

parser = argparse.ArgumentParser(description='Script so useful.')
parser.add_argument("--network", default="mobilenet_v2")
parser.add_argument("--pretrained", default=True)
parser.add_argument("--whole_network", default=True)
parser.add_argument("--name_experiment", default="Default")
parser.add_argument("--dataset_folder", default="Default")
parser.add_argument("--dataset_csv", default="Default")
parser.add_argument("--classes_names_path", default=None)

args = parser.parse_args()

cnn_network = args.network
pretrained = args.pretrained
whole_network = bool(args.whole_network)
name_network_experiment = args.name_experiment
path_folder = args.dataset_folder
path_csv = args.dataset_csv
classes_names_path = args.classes_names_path

########################################################
### Change:

k_fold=5
num_epochs = 20

#########################################################


class_names = list(pd.read_csv(classes_names_path, header=None)[0])


sourceFile = open(f'log\\{name_network_experiment}.txt', 'w')
print(f'Start cross validation: {name_network_experiment}', file = sourceFile)
#_____________________________________________________________________________
print(f'Start cross validation: {name_network_experiment}')


print(f'path csv: {path_csv}', file = sourceFile)
#_____________________________________________________________________________
print(f'path csv: {path_csv}')


print(f'path_folder: {path_folder}', file = sourceFile)
#_____________________________________________________________________________
print(f'path_folder: {path_folder}')


print(f'Cnn_network: {cnn_network}, pretrained: {pretrained}, whole_network: {whole_network}', file = sourceFile)
#_____________________________________________________________________________
print(f'Cnn_network: {cnn_network}, pretrained: {pretrained}, whole_network: {whole_network}')


if __name__ == '__main__':
    print(f'Class_names: {class_names}', file = sourceFile)
    #_____________________________________________________________________________
    print(f'Class_names: {class_names}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # tensorboard
    writer = SummaryWriter(f'runs/{name_network_experiment}')

    image_datasets = mcf.MyDataset(path_csv,path_folder)


    total_size = len(image_datasets)
    fraction = 1/k_fold
    seg = int(total_size * fraction)

    number_iteration = 1

    image_datasets = None

    for i in range(k_fold):
    # for i in range(number_iteration):

        print('-'*100, file = sourceFile)
        print(f'{i} fold training', file = sourceFile)
        print(' ', file = sourceFile)
        #_____________________________________________
        print('-'*100)
        print(f'{i} fold training')
        print(' ')
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size
        # msg
        print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)" 
                % (trll,trlr,trrl,trrr,vall,valr),  file = sourceFile)
        #____________________________________________________________
        print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)" 
                % (trll,trlr,trrl,trrr,vall,valr))
        
        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))
        
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))
        
        
        image_datasets = mcf.MyDataset_A(path_csv,path_folder)
        

        train_set = torch.utils.data.dataset.Subset(image_datasets,train_indices)
        valid_set = torch.utils.data.dataset.Subset(image_datasets,val_indices)
        
        print(len(train_set),len(valid_set), file = sourceFile)
        print(' ', file = sourceFile)
        #______________________________________________________
        print(len(train_set),len(valid_set))
        print()

        train_set2 = mcf.MapDataset_A(train_set, A.Compose([
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.Resize(224, 224),
            
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
            
        ]))

        valid_set2 = mcf.MapDataset_A(valid_set, A.Compose([
            A.Resize(224, 224),
            
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                ),
            ToTensorV2(), 
            ]))


        image_datasets = {'train': train_set2, 'valid': valid_set2}

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                shuffle=True, num_workers=0)
                for x in ['train', 'valid', ]}
        dataset_sizes_ = {x: len(image_datasets[x]) for x in ['train', 'valid']}

        print(' ', file = sourceFile)
        print(f'Device: {device}', file = sourceFile)
        print(f'First class: {class_names[0]}', file = sourceFile)
        print(f'Number of classes: {len(class_names)}', file = sourceFile)
        print(' ', file = sourceFile)
        #_________________________________________________________________
        print(' ')
        print(f'Device: {device}')
        print(f'First class: {class_names[0]}')
        print(f'Number of classes: {len(class_names)}')
        print('\n*3')
        print(' ')
        print(' ')


        model_nn = netfunc.network(name = cnn_network, pretrained = pretrained, num_classes = len(class_names), whole_network = whole_network)
        model_nn = model_nn.to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(model_nn.parameters(), lr=0.01, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        mcf.show_parameters(model_nn, source_writer=sourceFile)

        model_nn, acc_model, loss_model = mcf.train_model_without_test(model = model_nn, dataloaders = dataloaders, dataset_sizes = dataset_sizes_, criterion = criterion, optimizer = optimizer, scheduler = exp_lr_scheduler, num_epochs = num_epochs, name_network = f'{name_network_experiment}_fold{i}', device = device, writer = writer, source_writer = sourceFile)              

        torch.save(model_nn, f'{name_network_experiment}_fold{i}.pt')
        print(f'Model save: {name_network_experiment}_fold{i}.pt', file=sourceFile)
        print(f'Model save: {name_network_experiment}_fold{i}.pt')

    sourceFile.close()
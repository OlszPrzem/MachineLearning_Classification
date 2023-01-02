# from __future__ import print_function, division

import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
import numpy as np
# import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
import time
import os
import copy
import PIL
from PIL import Image
# import random
import cv2
# from numpy import float16, uint8
# from sklearn.metrics import confusion_matrix
# import seaborn as sn
import pandas as pd
# import sys
# import sklearn
from datetime import timedelta

# from torch.utils.tensorboard import SummaryWriter
# plt.ion()   # interactive mode


# class RandomBackground_opencv:
#     """Crop randomly the image in a sample.

#     Args:
#         output_size (tuple or int): Desired output size. If int, square crop
#             is made.
#     """

#     def __init__(self):
        

#         # self.path_folder_background = "E:/tla"
#         # self.path_folder_background = "D:\\data_test_demon\\!tlo"
#         self.path_folder_background = "D:\\!MGR\\data_test_demon(dataset2)\\!tlo"
        

#         if not os.path.exists(self.path_folder_background):
#             print(f"Path {self.path_folder_background} don't exist!")
#             sys.exit()

#         self.list_backgrounds = os.listdir((self.path_folder_background))
#         self.number_backgrounds = len(self.list_backgrounds)
#         self.list_path_backgrounds=[]
#         self.bacground_array = []

#         for element in self.list_backgrounds:
#             self.list_path_backgrounds.append(f"{self.path_folder_background}/{element}")

#         for image in range (0,len(self.list_backgrounds)):
#             img = cv2.imread(self.list_path_backgrounds[image])
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             self.bacground_array.append(img)
#             # self.bacground_array.append(self.list_path_backgrounds[image])
        
#         # end_img = np.zeros([224,224,3])
#         # image_to_prepare = np.zeros([224,224,3])


#     def __call__(self, sample):
#         sample = np.array(sample) 
#         # print(f"sample.shape: {sample.shape}")

#         # print(sample.shape)
#         # sample = cv2.co
#         # sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
#         # front_img_array = sample.convert('RGB')
#         # image_to_prepare = sample
#         selected_background = random.randint(0,self.number_backgrounds-1)
#         # print(sample.shape)

#         max_size = 224
#         roi_max_size = 200 
#         coef_resize = 0.6
#         # background = Image.open(list_path_backgrounds[selected_background])
#         # background = cv2.imread(list_path_backgrounds[selected_background])
#         background = self.bacground_array[selected_background]
#         # print(self.bacground_array)
#         # print(background)

#         roi = sample
#         # print(roi.shape)
#         # background=background.convert('RGB')
        
#         background_y = background.shape[0]
#         background_x = background.shape[1]
        
#         if background_x != background_y:
#             if background_x > background_y:
#                 diff = background_x-background_y
#                 start_point_x = random.randint(0,diff-1)
#                 start_point_y = 0

#                 # background = background.crop((start_point_y, start_point_x, background_y,(start_point_x+background_y)))
#                 background = background[start_point_y:(start_point_y+background_y),start_point_x:(start_point_x+background_y)]
#                 # print(f"backround 1 shape : {background.shape}") 
#             else:
#                 diff = background_y - background_x
#                 start_point_x = 0
#                 start_point_y = random.randint(0, diff - 1)
#                 # background = background.crop((start_point_y, start_point_x, (start_point_y+background_x),background_x))
#                 background = background[start_point_y:(background_x+start_point_y), start_point_x: (start_point_x+background_x)]
#                 # print(f"backround 2 shape : {background.shape}")
#         background_y = background.shape[0]
#         background_x = background.shape[1]

#         coefficient_back = max_size / background_x
#         # background_resized = background.resize((int(background_y*coefficient_back), int(background_x*coefficient_back)))

#         if (int(background_y*coefficient_back)) < 224 :
#                 dim_1 = 224
#         else:
#                 dim_1 = int(background_y*coefficient_back)

#         if ((int(background_x*coefficient_back))< 224) :
#                 dim_2 = 224
#         else:
#                 dim_2 = int(background_y*coefficient_back)



#         background_resized = cv2.resize(background, (dim_1, dim_2) , interpolation=cv2.INTER_AREA)
        
#         background_y = background_resized.shape[0]
#         background_x = background_resized.shape[1]
#         # print(f'backround shape resized: {background.shape}')

#         shape_array = (background_y, background_x,1)

#         alpha_channel = np.ones(shape_array, dtype=np.uint8)
#         alpha_channel.fill(255)

#         # null_images = np.ones(shape_array, dtype=np.uint8)
#         # null_images.fill(0)
#         # print(f'alpha_channel_shape: {alpha_channel.shape}')
#         # print(f'backgroundl_shape: {background_resized.shape}')

#         ##### wazne!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ########################################
#         # background_resized = cv2.merge((background_resized[:,:,0], background_resized[:,:,1],background_resized[:,:,2], alpha_channel))



#         # background_null = cv2.merge((null_images, null_images, null_images, null_images))

#         roi_y = len(roi[0,:,0])
#         # print(roi_y) 
#         roi_x = len(roi[:,0,0])
#         # print(roi_x) 

#         # print(f'roi_x: {roi_x}, roi_y: {roi_y}')

#         if coef_resize<1:
#             if coef_resize>0.5:
#                 coef_resize = random.uniform(coef_resize, 1)
#             else:
#                 coef_resize = 0.5

#         if roi_x >= roi_max_size or roi_y >= roi_max_size:
#             if roi_x >= roi_y:
#                 coefficient_roi = roi_max_size / roi_x
#             else:
#                 coefficient_roi = roi_max_size / roi_y

#             # print(roi.shape)
#             # roi = roi.resize((int((roi_y * coefficient_roi)*coef_resize), int((roi_x * coefficient_roi)*coef_resize)))
#             roi = cv2.resize(roi, (int((roi_y * coefficient_roi)*coef_resize), int((roi_x * coefficient_roi)*coef_resize)),
#                                         interpolation=cv2.INTER_AREA)
#             # print(roi.shape)
#         roi_y = roi.shape[0]
#         roi_x = roi.shape[1]
#         # print(f'roi2.shape: {roi.shape}')


#         width = background_y-roi_y
#         height = background_x-roi_x

#         y_image = random.randint(0, height)
#         x_image = random.randint(0, width)


#         front_roi = np.zeros([224,224,4], dtype=uint8)
#         front_roi.fill(0)
#         front_roi[ x_image:x_image+roi_y, y_image:y_image+roi_x,:] = roi[:,:,:]

#         bg_img=background_resized

#         r,g,b,alpha = cv2.split(front_roi)
#         overlay_color = cv2.merge((r,g,b))

#         overlay_color = overlay_color.astype(float)
#         bg_img = bg_img.astype(float)

#         alpha = cv2.medianBlur(alpha,5)

#         alpha = alpha.astype(float)/255

#         alpha2 = np.zeros([224,224,3], dtype=float16)
#         alpha2.fill(0)
#         alpha2[:,:,0] = alpha
#         alpha2[:,:,1] = alpha
#         alpha2[:,:,2] = alpha
#         # print(alpha)

#         # print(alpha.shape, alpha2.shape, overlay_color.shape, bg_img.shape)


#         background_resized = overlay_color*alpha2 + bg_img*(1-alpha2)

#         # background_resized=background_resized/255
#         # print(f'back roi shape: {back_roi[ x_image:x_image+roi_y, y_image:y_image+roi_x,:].shape}')
#         # print(f'roi.shape: {roi.shape}')
#         # print('___')
#         ###############################################################
#         # background_resized_old = background_resized.copy()
#         # cnd = back_roi[:,:,3] > 0
#         # background_resized[cnd] = back_roi[cnd]
#         ###############################################################
#         # im11 = (background_resized).astype(np.uint8)
#         # im11 = cv2.cvtColor(im11, cv2.COLOR_RGB2BGR)
#         # cv2.imshow('ga',im11)
#         # cv2.waitKey(0)
#         background_resized_1 = Image.fromarray((background_resized).astype(np.uint8))
#         background_resized_1 = background_resized_1.convert('RGB')
#         return background_resized_1

# #################################################
# ## Custom loader for files png

# def custom_loader(path):
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGBA')


# #################################################
# ### Function for show pictures

# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.axis('off')
#     plt.imshow(inp)
    
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated


# ##################################################
# #### Function train




# def train_model_old(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25, name_network = ' ', device = 'cpu', writer = None):
#     step = 0
#     since = time.time()
#     # global loss_valid
#     # global loss_train
#     # global acc_valid 
#     # global acc_train 

    
#     loss_valid = np.array([])
#     loss_train = np.zeros([])
    
#     acc_valid =  np.array([])
#     acc_train =  np.array([])

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#     all_iteration = {'train':len(dataloaders['train']), 'valid' : len(dataloaders['valid']) }
#     # all_iteration[0] = len(dataloaders['train'])
#     # all_iteration[1] = len(dataloaders['valid'])

#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)

        
#         # Each epoch has a training and validation phase
#         for phase in ['train', 'valid']:

#             start_time_phase_epoch = time.time()

#             if phase == 'train':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()   # Set model to evaluate mode

#             running_loss = 0.0
#             running_corrects = 0
#             iteration = 1 
#             # Iterate over data.

#             for inputs, labels in dataloaders[phase]:
                
#                 # if phase == 'train':
#                 percent_iter = int(100*iteration/all_iteration[phase])
#                 oki = '█'
#                 noki = '-'
#                 number_all_element_bar= int(100/3)
#                 number_elements_bar = int(percent_iter/3)
#                 print(f'\r Number iteration of phase {phase:8s}: {iteration:5d}/{all_iteration[phase]} | Complet: {percent_iter:3d}%  | [{oki*number_elements_bar}{noki*(number_all_element_bar-number_elements_bar)}]', end = '', flush = False)
                


#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
                

#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#                 running_corrects2 = torch.sum(preds == labels.data)
#                 accuaracy_batch = float(running_corrects2)/float(inputs.shape[0])

#                 iteration +=1

#                 if writer is not None:
#                     writer.add_scalars('Training loss',{f'{name_network}': loss}, global_step=step)
#                     writer.add_scalars('Training accuracy', {f'{name_network}': accuaracy_batch}, global_step=step )
#                 step +=1

#             if phase == 'train':
#                 scheduler.step()

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             stop_time_phase_epoch = time.time()

#             time_epoch = stop_time_phase_epoch - start_time_phase_epoch

#             print(' |  {} Loss: {:.4f} Acc: {:.4f} | Time for {} phase: {:.2f}s'.format(
#                 phase, epoch_loss, epoch_acc, phase, time_epoch))

#             if phase == 'train':
#                 loss_train = np.append(loss_train,epoch_loss)
#                 acc_epoch = epoch_acc.cpu().numpy()
#                 acc_train = np.append(acc_train,acc_epoch)

#             elif phase == 'valid':
#                 loss_valid = np.append(loss_valid,epoch_loss)
#                 acc_epoch = epoch_acc.cpu().numpy()
#                 acc_valid = np.append(acc_valid,acc_epoch)


#                 np.append(acc_valid,acc_epoch)
#                 # print(f'acc_valid=')
#                 # print(acc_valid)
          

#             # deep copy the model
#             if phase == 'valid' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())

#         print()


#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))



#     loss_=[loss_train, loss_valid]
#     acc_ = [acc_train, acc_valid]

#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model, acc_, loss_

def train_model_without_test(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25, name_network = ' ', device = 'cpu', writer = None, source_writer = None):
    step_train = 0
    step_valid = 0

    since = time.time()

    start_loss_writer = 3
    start_acc_writer = 0
    writer.add_scalars('Training loss',{f'{name_network}': start_loss_writer }, global_step=step_train)
    writer.add_scalars('Training accuracy', {f'{name_network}': start_acc_writer}, global_step=step_train)
    writer.add_scalars('Validaton loss',{f'{name_network}': start_loss_writer}, global_step=step_valid)
    writer.add_scalars('Validation accuracy', {f'{name_network}': start_acc_writer}, global_step=step_valid)
    writer.add_scalars('Test loss',{f'{name_network}': start_loss_writer}, global_step=step_valid)
    writer.add_scalars('Test accuracy', {f'{name_network}': start_acc_writer}, global_step=step_valid)
    writer.add_scalars(f'LossEpoch', {
                        f'{name_network}/train': 3,
                        f'{name_network}/valid': 3,
                        f'{name_network}/test': 3,
                    }, global_step = 0)

    writer.add_scalars(f'AccEpoch', {
                        f'{name_network}/train': 0,
                        f'{name_network}/valid': 0,
                        f'{name_network}/test': 0,
                    }, global_step = 0)

    
    loss_valid = np.array([])
    loss_train = np.zeros([])
    loss_test =  np.array([])
    
    acc_valid =  np.array([])
    acc_train =  np.array([])
    acc_test =  np.array([])
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    all_iteration = {'train':len(dataloaders['train']), 'valid' : len(dataloaders['valid']) }

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1), file = source_writer)
        print('-' * 10, file = source_writer)
        #_______________________________________________________________________
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:

            start_time_phase_epoch = time.time()

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            iteration = 1 
            # Iterate over data.

            for inputs, labels in dataloaders[phase]:
                
                # if phase == 'train':
                percent_iter = int(100*iteration/all_iteration[phase])
                oki = 'X'#█'
                noki = '-'
                number_all_element_bar= int(100/3)
                number_elements_bar = int(percent_iter/3)
                print(f'\r Number iteration of phase {phase:8s}: {iteration:5d}/{all_iteration[phase]} | Complet: {percent_iter:3d}%  | [{oki*number_elements_bar}{noki*(number_all_element_bar-number_elements_bar)}]', end = '', flush = False, file = source_writer)
                print(f'\r Number iteration of phase {phase:8s}: {iteration:5d}/{all_iteration[phase]} | Complet: {percent_iter:3d}%  | [{oki*number_elements_bar}{noki*(number_all_element_bar-number_elements_bar)}]', end = '', flush = False)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_corrects2 = torch.sum(preds == labels.data)
                accuaracy_batch = float(running_corrects2)/float(inputs.shape[0])

                iteration +=1

                if writer is not None:
                    if phase == 'train':
                        writer.add_scalars('Training loss',{f'{name_network}': loss}, global_step=step_train)
                        writer.add_scalars('Training accuracy', {f'{name_network}': accuaracy_batch}, global_step=step_train)
                        step_train +=1
                    elif phase == 'valid':
                        writer.add_scalars('Validaton loss',{f'{name_network}': loss}, global_step=step_valid)
                        writer.add_scalars('Validation accuracy', {f'{name_network}': accuaracy_batch}, global_step=step_valid)
                        step_valid +=1

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            stop_time_phase_epoch = time.time()

            time_epoch = stop_time_phase_epoch - start_time_phase_epoch
            td = timedelta(seconds=time_epoch)
            # print('Time in hh:mm:ss:', td)

            print(' |  {} Loss: {:.4f} Acc: {:.4f} | Time for {} phase: {}'.format(
                phase, epoch_loss, epoch_acc, phase, td), file = source_writer)
            #___________________________________________________________________________
            print(' |  {} Loss: {:.4f} Acc: {:.4f} | Time for {} phase: {}'.format(
                phase, epoch_loss, epoch_acc, phase, td))

            if phase == 'train':
                loss_train = np.append(loss_train,epoch_loss)
                epoch_loss_train = epoch_loss                
                acc_epoch_train = epoch_acc.cpu().numpy()
                acc_train = np.append(acc_train,acc_epoch_train)
                # step_epoch_train+=1

            elif phase == 'valid':
                loss_valid = np.append(loss_valid,epoch_loss)
                epoch_loss_valid = epoch_loss
                acc_epoch_valid = epoch_acc.cpu().numpy()
                acc_valid = np.append(acc_valid,acc_epoch_valid)

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, f'{name_network}.pt')

        writer.add_scalars(f'LossEpoch', {
                        f'{name_network}/train': epoch_loss_train,
                        f'{name_network}/valid': epoch_loss_valid,
                    }, global_step = epoch+1)

        writer.add_scalars(f'AccEpoch', {
                        f'{name_network}/train': acc_epoch_train,
                        f'{name_network}/valid': acc_epoch_valid,
                    }, global_step = epoch+1)
        
        print( ' ',file = source_writer)
        print()


    time_elapsed = time.time() - since
    td = timedelta(seconds=time_elapsed)

    print('Training complete in {}'.format(
        td), file = source_writer)
    print('Best val Acc: {:4f}'.format(best_acc), file = source_writer)
    #________________________________________________________________
    print('Training complete in {}'.format(
        td))
    print('Best val Acc: {:4f}'.format(best_acc))


    loss_=[loss_train, loss_valid, loss_test]
    acc_ = [acc_train, acc_valid, acc_test]

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, acc_, loss_



# def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25, name_network = ' ', device = 'cpu', writer = None, source_writer = None):
#     step_train = 0
#     step_valid = 0
#     # step_epoch_valid = 0
#     # step_epoch_train = 0

#     since = time.time()
#     # global loss_valid
#     # global loss_train
#     # global acc_valid 
#     # global acc_train 
#     start_loss_writer = 3
#     start_acc_writer = 0
#     writer.add_scalars('Training loss',{f'{name_network}': start_loss_writer }, global_step=step_train)
#     writer.add_scalars('Training accuracy', {f'{name_network}': start_acc_writer}, global_step=step_train)
#     writer.add_scalars('Validaton loss',{f'{name_network}': start_loss_writer}, global_step=step_valid)
#     writer.add_scalars('Validation accuracy', {f'{name_network}': start_acc_writer}, global_step=step_valid)
#     writer.add_scalars('Test loss',{f'{name_network}': start_loss_writer}, global_step=step_valid)
#     writer.add_scalars('Test accuracy', {f'{name_network}': start_acc_writer}, global_step=step_valid)
#     writer.add_scalars(f'LossEpoch', {
#                         f'{name_network}/train': 3,
#                         f'{name_network}/valid': 3,
#                         f'{name_network}/test': 3,
#                     }, global_step = 0)

#     writer.add_scalars(f'AccEpoch', {
#                         f'{name_network}/train': 0,
#                         f'{name_network}/valid': 0,
#                         f'{name_network}/test': 0,
#                     }, global_step = 0)

    
#     loss_valid = np.array([])
#     loss_train = np.zeros([])
#     loss_test =  np.array([])
    
#     acc_valid =  np.array([])
#     acc_train =  np.array([])
#     acc_test =  np.array([])
    

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
    
#     all_iteration = {'train':len(dataloaders['train']), 'valid' : len(dataloaders['valid']), 'test' : len(dataloaders['test']) }
#     # all_iteration[0] = len(dataloaders['train'])
#     # all_iteration[1] = len(dataloaders['valid'])

#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1), file = source_writer)
#         print('-' * 10, file = source_writer)
#         #_______________________________________________________________________
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)

        
#         # Each epoch has a training and validation phase
#         for phase in ['train', 'valid', 'test']:

#             start_time_phase_epoch = time.time()

#             if phase == 'train':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()   # Set model to evaluate mode

#             running_loss = 0.0
#             running_corrects = 0
#             iteration = 1 
#             # Iterate over data.

#             for inputs, labels in dataloaders[phase]:
                
#                 # if phase == 'train':
#                 percent_iter = int(100*iteration/all_iteration[phase])
#                 oki = 'X'#█'
#                 noki = '-'
#                 number_all_element_bar= int(100/3)
#                 number_elements_bar = int(percent_iter/3)
#                 print(f'\r Number iteration of phase {phase:8s}: {iteration:5d}/{all_iteration[phase]} | Complet: {percent_iter:3d}%  | [{oki*number_elements_bar}{noki*(number_all_element_bar-number_elements_bar)}]', end = '', flush = False, file = source_writer)
#                 print(f'\r Number iteration of phase {phase:8s}: {iteration:5d}/{all_iteration[phase]} | Complet: {percent_iter:3d}%  | [{oki*number_elements_bar}{noki*(number_all_element_bar-number_elements_bar)}]', end = '', flush = False)



#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
                

#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#                 running_corrects2 = torch.sum(preds == labels.data)
#                 accuaracy_batch = float(running_corrects2)/float(inputs.shape[0])

#                 iteration +=1

#                 if writer is not None:
#                     if phase == 'train':
#                         writer.add_scalars('Training loss',{f'{name_network}': loss}, global_step=step_train)
#                         writer.add_scalars('Training accuracy', {f'{name_network}': accuaracy_batch}, global_step=step_train)
#                         step_train +=1
#                     elif phase == 'valid':
#                         writer.add_scalars('Validaton loss',{f'{name_network}': loss}, global_step=step_valid)
#                         writer.add_scalars('Validation accuracy', {f'{name_network}': accuaracy_batch}, global_step=step_valid)
#                         step_valid +=1
#                     elif phase == 'test':
#                         writer.add_scalars('Test loss',{f'{name_network}': loss}, global_step=step_valid)
#                         writer.add_scalars('Test accuracy', {f'{name_network}': accuaracy_batch}, global_step=step_valid)
#                         step_valid +=1
                

#             if phase == 'train':
#                 scheduler.step()

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             stop_time_phase_epoch = time.time()

#             time_epoch = stop_time_phase_epoch - start_time_phase_epoch

#             print(' |  {} Loss: {:.4f} Acc: {:.4f} | Time for {} phase: {:.2f}s'.format(
#                 phase, epoch_loss, epoch_acc, phase, time_epoch), file = source_writer)
#             #___________________________________________________________________________
#             print(' |  {} Loss: {:.4f} Acc: {:.4f} | Time for {} phase: {:.2f}s'.format(
#                 phase, epoch_loss, epoch_acc, phase, time_epoch))

#             if phase == 'train':
#                 loss_train = np.append(loss_train,epoch_loss)
#                 epoch_loss_train = epoch_loss                
#                 acc_epoch_train = epoch_acc.cpu().numpy()
#                 acc_train = np.append(acc_train,acc_epoch_train)
#                 # step_epoch_train+=1

#             elif phase == 'valid':
#                 loss_valid = np.append(loss_valid,epoch_loss)
#                 epoch_loss_valid = epoch_loss
#                 acc_epoch_valid = epoch_acc.cpu().numpy()
#                 acc_valid = np.append(acc_valid,acc_epoch_valid)
#                 # step_epoch_valid+=1

#             elif phase == 'test':
#                 loss_test = np.append(loss_test,epoch_loss)
#                 epoch_loss_test = epoch_loss
#                 acc_epoch_test = epoch_acc.cpu().numpy()
#                 acc_test = np.append(acc_test,acc_epoch_test)
#                 # step_epoch_valid+=1


#                 # np.append(acc_valid,acc_epoch_valid)
#                 # print(f'acc_valid=')
#                 # print(acc_valid)

#             # deep copy the model
#             if phase == 'test' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())

#         writer.add_scalars(f'LossEpoch', {
#                         f'{name_network}/train': epoch_loss_train,
#                         f'{name_network}/valid': epoch_loss_valid,
#                         f'{name_network}/test': epoch_loss_test,
#                     }, global_step = epoch+1)

#         writer.add_scalars(f'AccEpoch', {
#                         f'{name_network}/train': acc_epoch_train,
#                         f'{name_network}/valid': acc_epoch_valid,
#                         f'{name_network}/test': acc_epoch_test,
#                     }, global_step = epoch+1)
        
        
        
#         print( ' ',file = source_writer)
#         print()


#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60), file = source_writer)
#     print('Best val Acc: {:4f}'.format(best_acc), file = source_writer)
#     #________________________________________________________________
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))



#     loss_=[loss_train, loss_valid, loss_test]
#     acc_ = [acc_train, acc_valid, acc_test]

#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model, acc_, loss_



# ######################################################
# ## vizualization models

# def visualize_model(model, class_names, dataloader, num_images=6):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()

#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloader):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)

#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images//2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title('predicted: {}'.format(class_names[preds[j]]))
#                 imshow(inputs.cpu().data[j])

#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)

# def visualize_model_3(model, class_names, dataloader, num_images=6, device='cpu'):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()

#     with torch.no_grad():
#         for i, (labels, inputs1, inputs2, inputs3 ) in enumerate(dataloader):
#             inputs1 = inputs1.to(device)
#             inputs2 = inputs2.to(device)
#             inputs3 = inputs3.to(device)
#             labels = labels.to(device)

#             outputs1 = model(inputs1)
#             outputs2 = model(inputs2)
#             outputs3 = model(inputs3)
#             _, preds1 = torch.max(outputs1, 1)
#             _, preds2 = torch.max(outputs2, 1)
#             _, preds3 = torch.max(outputs3, 1)

#             for j in range(labels.size()[0]):
#                 for k in range (0,3):
#                     images_so_far += 1
#                     ax = plt.subplot(labels.size()[0], 3, images_so_far)
#                     ax.axis('off')
#                     ax.set_title('predicted: {}'.format(class_names[preds1[j]]))
#                     imshow(inputs1[j][k].cpu())

#                     if images_so_far == num_images:
#                         model.train(mode=was_training)
#                         return
#         model.train(mode=was_training)

# #######################################################
# ## connfusion matrix

# def conf_matrix (datloader, model, dataset, class_names):
#     y_pred = []
#     y_true = []
#     running_corrects_top1 = 0
#     sum_cor_pred = 0
#     sum_all_pred = 0

#     dataloaders_sizes_test = len(datloader)
#     dataset_sizes_test = len(dataset)

#     running_corrects_top1 = 0
#     running_corrects_top3 = 0
#     running_corrects_top5 = 0
#     print('- 1 -')

#     for i in range(64):
#         y_pred.append(i)
#         y_true.append(i)



#     # iterate over test data
#     for i, (inputs, labels) in enumerate(datloader):
#             # print("\r", f'{i}/{dataloaders_sizes_test}', end='')
#             inputs = inputs.cuda()
#             labels = labels.cuda()
#             output = model(inputs) # Feed Network
#             # print(f'outputshape: {output.size()}')
        
#             output2 = (torch.max(torch.exp(output),1)[1]).data.cpu().numpy()
#             y_pred.extend(output2) # Save Prediction
#             labels2 = labels.data.cpu().numpy()
#             y_true.extend(labels2) # Save Prediction

#             inputs = inputs.cpu()
#             labels = labels.reshape(len(labels),1).cpu()
#             output = output.cpu()

#             running_corrects_top1 += (torch.sum(labels == torch.topk(torch.exp(output),1).indices) ).numpy()
#             running_corrects_top3 += (torch.sum(labels == torch.topk(torch.exp(output),3).indices) ).numpy()
#             running_corrects_top5 += (torch.sum(labels == torch.topk(torch.exp(output),5).indices) ).numpy()

#             print("\r", f'{i}/{dataloaders_sizes_test}', end='')

#     ##########################
    

#     # Build confusion matrix

#     ####
#     # Ver 1
#     # cf_matrix = confusion_matrix(y_true, y_pred, normalize = 'true')
#     # print(class_names)
#     # df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in class_names],
#     #                     columns = [i for i in class_names])
#     # plt.figure(figsize = (12,7))
#     # sn.heatmap(df_cm, annot=True)

#     ####
#     # Ver 2

#     cf_matrix = confusion_matrix(y_true, y_pred, normalize = 'true')
#     print(len(cf_matrix[0]))
#     print(class_names)
#     df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *64, index = [i for i in class_names],
#                         columns = [i for i in class_names])

#     df_cm[df_cm.eq(0)] = np.nan
#     mask = df_cm.isnull()
#     # plt.figure(figsize = (24,10))
#     plt.figure(figsize = (24,10))

#     font_size_0 = 4

#     res = sn.heatmap(df_cm, annot=True, cmap='PuBu', linewidths=.5, mask=df_cm.isnull(), fmt = '.2%', yticklabels = True, annot_kws={"size":font_size_0, "weight": "bold"}, cbar_kws={"pad":0.01})
#     res.set_facecolor('#E8E8E8')
    

#     cbar = res.collections[0].colorbar
#     cbar.set_ticks([0, .25, .5, .75, 1])
#     cbar.set_ticklabels(['low', '25%', '50%', '75%', '100%'])
#     cbar.ax.tick_params(labelsize=8)
    
#     font_size_1 = 10
#     font_size_2 = 6



#     res.set_xticklabels(res.get_xmajorticklabels(), fontsize = font_size_2, style='italic', rotation=90, horizontalalignment='center')
#     res.set_yticklabels(res.get_ymajorticklabels(), fontsize = font_size_2, style='italic', rotation="horizontal")

#     plt.xlabel("PRZEWIDYWANA KLASA", fontsize = font_size_1, labelpad= 5, fontweight='bold')
#     plt.ylabel("PRAWDZIWA KLASA", fontsize = font_size_1, labelpad = 5, fontweight='bold')
#     # plt.show()

#     ##################################

#     print('Number all test samples: {}'.format(dataset_sizes_test))
#     print('_'*50)
#     print('Number correct predict top1 test samples: {}'.format(running_corrects_top1))
#     # num_corect = running_corrects11.cpu().detach().numpy()
#     print('Top 1 acc: {:3.2f}%'.format((running_corrects_top1/dataset_sizes_test)*100))
#     print('_'*50)
#     print('Number correct predict top1 test samples: {}'.format(running_corrects_top3))
#     # num_corect = running_corrects11.cpu().detach().numpy()
#     print('Top 3 acc: {:3.2f}%'.format((running_corrects_top3/dataset_sizes_test)*100))
#     print('_'*50)
#     print('Number correct predict top5 test samples: {}'.format(running_corrects_top5))
#     # num_corect = running_corrects11.cpu().detach().numpy()
#     print('Top 5 acc: {:3.2f}%'.format((running_corrects_top5/dataset_sizes_test)*100))

#     print(sklearn.metrics.classification_report(y_true, y_pred, target_names=class_names, digits=4))

#     plt.subplots_adjust(top = 0.99, bottom = 0.2, right = 1.1, left = 0.1, 
#             hspace = 0, wspace = 0)

#     plt.show(block = True)
    
#     fig = res.get_figure()
#     fig.savefig("out.pdf") 

#     return y_true, y_pred




# def conf_matrix_3 (dataloader, model, dataset, class_names):
#     y_pred = []
#     y_true = []
    
#     y_pred, y_true = check_accuracy_3(dataloader, model, dataset)
    
#     # Build confusion matrix
#     cf_matrix = confusion_matrix(y_true, y_pred, normalize = 'true')

#     ## ver1
#     # print(class_names)
#     # df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in class_names],
#     #                     columns = [i for i in class_names])
#     # plt.figure(figsize = (12,7))
#     # sn.heatmap(df_cm, annot=True)

#    # Ver 2
#     cf_matrix = confusion_matrix(y_true, y_pred, normalize = 'true')
#     print(class_names)
#     df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in class_names],
#                         columns = [i for i in class_names])

#     df_cm[df_cm.eq(0)] = np.nan
#     mask = df_cm.isnull()
#     plt.figure(figsize = (24,10))

#     res = sn.heatmap(df_cm, annot=True, cmap='PuBu', linewidths=.5, mask=df_cm.isnull(), fmt = '.2%', annot_kws={"size":19, "weight": "bold"})
#     res.set_facecolor('#E8E8E8')

#     cbar = res.collections[0].colorbar
#     cbar.set_ticks([0, .25, .5, .75, 1])
#     cbar.set_ticklabels(['low', '25%', '50%', '75%', '100%'])
#     cbar.ax.tick_params(labelsize=20)
    
#     res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 25, style='italic', rotation=45, horizontalalignment='right')
#     res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 25, style='italic', rotation="horizontal")

#     plt.xlabel("PRZEWIDYWANA KLASA", fontsize = 30, labelpad= 35, fontweight='bold')
#     plt.ylabel("PRAWDZIWA KLASA", fontsize = 30, labelpad = 35, fontweight='bold')
#     plt.show()

#     ##################################



#     # mAP = sklearn.metrics.average_precision_score(y_true, y_pred, 'micro')

#     # for i,(position) in enumerate(mAP):
#     #     if i > len(class_names)-1:
#     #         break
#     #     print(class_names[i], mAP[f'{i}'])


#     report = sklearn.metrics.classification_report(y_true, y_pred, output_dict=True)

#     print(sklearn.metrics.classification_report(y_true, y_pred, target_names=class_names, digits=4))
#     # for i,(position) in enumerate(report):
#     #     if i > len(class_names)-1:
#     #         break
#     #     print(class_names[i], report[f'{i}'])

#     return y_true, y_pred
    

    
# def check_accuracy_3(datloader, model, dataset):
#     y_pred = []
#     y_true = []
#     running_corrects_top1 = 0
#     sum_cor_pred = 0
#     sum_all_pred = 0
#     dataloaders_sizes_test = len(datloader)
#     dataset_sizes_test = len(dataset)
    

#     running_corrects_top1 = 0
#     running_corrects_top3 = 0
#     running_corrects_top5 = 0
#     print('- - -')
#     # iterate over test data
#     for i, (labels, inputs1, inputs2, inputs3) in enumerate(datloader):
        
#         print("\r", f'{i}/{dataloaders_sizes_test}', end='')
#         labels = labels.cuda()
#         inputs1 = inputs1.cuda()
#         inputs2 = inputs2.cuda()
#         inputs3 = inputs3.cuda()
        
#         output1 = model(inputs1) # Feed Network
#         output2 = model(inputs2) # Feed Network
#         output3 = model(inputs3) # Feed Network


#         output_all = output1.add(output2)
#         output_all = output_all.add(output3)

#         output_div = torch.div(output_all, 3)
#         output_div_cpu = (torch.max(torch.exp(output_div),1)[1]).data.cpu().numpy()

#         y_pred.extend(output_div_cpu) # Save Prediction
#         labels2 = labels.data.cpu().numpy()
#         y_true.extend(labels2) # Save Prediction

#         # if labels2 != output_div_cpu:
#         #     print('blad')
#             # y = torch.squeeze(inputs1, 0)
#             # inp = y.cpu().numpy().transpose((1, 2, 0))
            
#             # plt.figure()
#             # plt.imshow(inp)
#         #     # plt.imshow(img2)
#         #     # plt.imshow(img3)
#         inputs1 = inputs1.cpu()
#         inputs2 = inputs2.cpu()
#         inputs3 = inputs3.cpu()
#         labels = labels.reshape(len(labels),1).cpu()
#         output_div_cpu = output_div.cpu()

#         running_corrects_top1 += (torch.sum(labels == torch.topk(torch.exp(output_div_cpu),1).indices) ).numpy()
#         running_corrects_top3 += (torch.sum(labels == torch.topk(torch.exp(output_div_cpu),3).indices) ).numpy()
#         running_corrects_top5 += (torch.sum(labels == torch.topk(torch.exp(output_div_cpu),5).indices) ).numpy()
#     print(' ')
#     print('Number all test samples: {}'.format(dataset_sizes_test))
#     print('_'*50)
#     print('Number correct predict top1 test samples: {}'.format(running_corrects_top1))
#     # num_corect = running_corrects11.cpu().detach().numpy()
#     print('Top 1 acc: {:3.2f}%'.format((running_corrects_top1/dataset_sizes_test)*100))
#     print('_'*50)
#     print('Number correct predict top1 test samples: {}'.format(running_corrects_top3))
#     # num_corect = running_corrects11.cpu().detach().numpy()
#     print('Top 3 acc: {:3.2f}%'.format((running_corrects_top3/dataset_sizes_test)*100))
#     print('_'*50)
#     print('Number correct predict top5 test samples: {}'.format(running_corrects_top5))
#     # num_corect = running_corrects11.cpu().detach().numpy()
#     print('Top 5 acc: {:3.2f}%'.format((running_corrects_top5/dataset_sizes_test)*100))

#     return y_pred, y_true

# #######################################################################################
# ### my custom dataset

# from torch.utils.data import Dataset
# import pandas as pd
# from skimage import io
# from skimage.color import rgba2rgb

class MyDataset(Dataset):
    def __init__(self, csv_file, root_dir=None, transform=None):
        self.annotations = pd.read_csv(csv_file, header=None)#, encoding="latin1")
        self.root_dir = root_dir
        self.transform = transform
        # self.classes = 

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        if self.root_dir is not None:
            img_path = os.path.join(self.root_dir, self.annotations.iloc[index,2])
        else:
            img_path = self.annotations.iloc[index,2]


        image = Image.open(img_path)
        image = image.convert('RGB')
        # image = io.imread(img_path)
        y_label = torch.tensor(self.annotations.iloc[index,1])
        # image = rgba2rgb(image)
        if self.transform:
            image = self.transform(image)
        return (image, y_label)

class MyDataset_A(Dataset):
    def __init__(self, csv_file, root_dir = None, transform=None):
        self.annotations = pd.read_csv(csv_file, header=None)#, encoding="latin1")
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        if self.root_dir is not None:
            img_path = os.path.join(self.root_dir, self.annotations.iloc[index,2])
        else:
            img_path = self.annotations.iloc[index,2]

        # print(img_path)

        label = torch.tensor(self.annotations.iloc[index,1])
        
        image = cv2.imread(img_path)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print()
            print("error path:")
            print(img_path)
            raise "Img problem my friend"

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return (image, label)
        



# class MyDataset_3(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.annotations = pd.read_csv(csv_file, header=None)
#         self.root_dir = root_dir
#         self.transform = transform
#         # self.classes = 

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, index):
#         y_label = torch.tensor(self.annotations.iloc[index,1])
#         img_path1 = os.path.join(self.root_dir, self.annotations.iloc[index,2])
#         img_path2 = os.path.join(self.root_dir, self.annotations.iloc[index,3])
#         img_path3 = os.path.join(self.root_dir, self.annotations.iloc[index,4])
#         image1 = Image.open(img_path1)
#         image2 = Image.open(img_path2)
#         image3 = Image.open(img_path3)
#         image1 = image1.convert('RGB')
#         image2 = image2.convert('RGB')
#         image3 = image3.convert('RGB')
#         imageAll = Image.new('RGB', (image1.width + image2.width, image1.height+image3.height))
#         imageAll.paste(image1, (0, 0))
#         imageAll.paste(image2, (image1.width, 0))
#         imageAll.paste(image3, (0, image1.height))
#         # image = io.imread(img_path)
#         # y_label = torch.tensor(self.annotations.iloc[index,2])
#         # image = rgba2rgb(image)
#         if self.transform:
#             imageAll = self.transform(imageAll)
#         if self.transform:
#             image2 = self.transform(image2)
#         if self.transform:
#             image3 = self.transform(image3)
#         return (y_label, image1, image2, image3)


# class MyDataset_3_alg2(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.annotations = pd.read_csv(csv_file, header=None)
#         self.root_dir = root_dir
#         self.transform = transform
#         # self.classes = 

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, index):
#         y_label = torch.tensor(self.annotations.iloc[index,1])
#         img_path1 = os.path.join(self.root_dir, self.annotations.iloc[index,2])
#         img_path2 = os.path.join(self.root_dir, self.annotations.iloc[index,3])
#         img_path3 = os.path.join(self.root_dir, self.annotations.iloc[index,4])
#         image1 = Image.open(img_path1)
#         image2 = Image.open(img_path2)
#         image3 = Image.open(img_path3)
#         image1 = image1.convert('RGB')
#         image2 = image2.convert('RGB')
#         image3 = image3.convert('RGB')
#         imageAll = Image.new('RGB', (image1.width + image2.width, image1.height+image3.height))
#         imageAll.paste(image1, (0, 0))
#         imageAll.paste(image2, (image1.width, 0))
#         imageAll.paste(image3, (0, image1.height))

#         if self.transform:
#             imageAll = self.transform(imageAll)

#         return (imageAll, y_label)

# class MyDataset_3_alg2_ver2(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.annotations = pd.read_csv(csv_file, header=None)
#         self.root_dir = root_dir
#         self.transform = transform
#         # self.classes = 

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, index):
#         y_label = torch.tensor(self.annotations.iloc[index,1])
#         img_path1 = os.path.join(self.root_dir, self.annotations.iloc[index,2])
#         img_path2 = os.path.join(self.root_dir, self.annotations.iloc[index,3])
#         img_path3 = os.path.join(self.root_dir, self.annotations.iloc[index,4])
#         image1 = Image.open(img_path1)
#         image2 = Image.open(img_path2)
#         image3 = Image.open(img_path3)
#         image1 = image1.convert('RGB')
#         image2 = image2.convert('RGB')
#         image3 = image3.convert('RGB')
#         imageAll = Image.new('RGB', (image1.width + image2.width + image3.width, image1.height))
#         imageAll.paste(image1, (0, 0))
#         imageAll.paste(image2, (image1.width, 0))
#         imageAll.paste(image3, (image1.width+image2.width,0))

#         if self.transform:
#             imageAll = self.transform(imageAll)

#         return (imageAll, y_label)

# class MapDataset(torch.utils.data.Dataset):
#     """
#     Given a dataset, creates a dataset which applies a mapping function
#     to its items (lazily, only when an item is called).

#     Note that data is not cloned/copied from the initial dataset.
#     """

#     def __init__(self, dataset, map_fn):
#         self.dataset = dataset
#         self.map = map_fn

#     def __getitem__(self, index):
#         y = self.dataset[index][1]   # label
#         if self.map:     
#             x1 = self.map(self.dataset[index][0])
#         else:     
#             x1 = self.dataset[index][0]  # image

              
#         return (x1, y)

#     def __len__(self):
#         return len(self.dataset)

class MapDataset_A(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        y = self.dataset[index][1]   # label
        if self.map:     
            x1 = self.map(image = self.dataset[index][0])
            x1 = x1['image']
        else:     
            x1 = self.dataset[index][0]  # image
 
        return (x1, y)

    def __len__(self):
        return len(self.dataset)

# class MapDataset_3(torch.utils.data.Dataset):
#     """
#     Given a dataset, creates a dataset which applies a mapping function
#     to its items (lazily, only when an item is called).

#     Note that data is not cloned/copied from the initial dataset.
#     """

#     def __init__(self, dataset, map_fn):
#         self.dataset = dataset
#         self.map = map_fn

#     def __getitem__(self, index):
#         y = self.dataset[index][0]   # label
#         if self.map:     
#             x1 = self.map(self.dataset[index][1])
#             x2 = self.map(self.dataset[index][2])
#             x3 = self.map(self.dataset[index][3]) 
#         else:     
#             x1 = self.dataset[index][1]  # image
#             x2 = self.dataset[index][2]
#             x3 = self.dataset[index][3]
              
#         return y, x1, x2, x3

#     def __len__(self):
#         return len(self.dataset)


def show_parameters(model, source_writer = None):
    total_params = sum(param.numel() for param in model.parameters())
    print(f'{total_params:,} total parameters', file = source_writer)
    print(f'{total_params:,} total parameters')

    total_trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f'{total_trainable_params:,} training parameters', file = source_writer)
    print(f'{total_trainable_params:,} training parameters')


# ############################################

# from typing import Callable, Any, Optional, List
# class ConvBNActivation1(nn.Sequential):
#     def __init__(
#         self,
#         in_planes: int,
#         out_planes: int,
#         kernel_size: int = 3,
#         stride: int = 1,
#         groups: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#         activation_layer: Optional[Callable[..., nn.Module]] = None,
#         dilation: int = 1,
#     ) -> None:
#         padding = (kernel_size - 1) // 2 * dilation
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if activation_layer is None:
#             activation_layer = nn.ReLU6
#         super().__init__(
#             nn.Conv2d(in_planes, out_planes, (5,5), (1,4), (2,1), dilation=dilation, groups=groups,
#                       bias=False),
#             nn.Conv2d(out_planes, out_planes, (3,3), (2,2), padding, dilation=dilation, groups=groups,
#                       bias=False),
#             norm_layer(out_planes),
#             activation_layer(inplace=True)

#         )
#         self.out_channels = out_planes


# # necessary for backwards compatibility
# ConvBNReLU = ConvBNActivation1
    
# if __name__ == "__main__":
#     print("ok")
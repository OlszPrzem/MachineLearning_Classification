import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import sklearn


def conf_matrix (datloader, model, dataset, class_names):
    y_pred = []
    y_true = []
    running_corrects_top1 = 0

    dataloaders_sizes_test = len(datloader)
    dataset_sizes_test = len(dataset)

    running_corrects_top1 = 0
    running_corrects_top3 = 0
    running_corrects_top5 = 0

    for i in range(len(class_names)):
        y_pred.append(i)
        y_true.append(i)

    # iterate over test data
    for i, (inputs, labels) in enumerate(datloader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            output = model(inputs) # Feed Network
        
            output2 = (torch.max(torch.exp(output),1)[1]).data.cpu().numpy()
            y_pred.extend(output2) 
            labels2 = labels.data.cpu().numpy()
            y_true.extend(labels2) 

            inputs = inputs.cpu()
            labels = labels.reshape(len(labels),1).cpu()
            output = output.cpu()

            running_corrects_top1 += (torch.sum(labels == torch.topk(torch.exp(output),1).indices) ).numpy()
            running_corrects_top3 += (torch.sum(labels == torch.topk(torch.exp(output),3).indices) ).numpy()
            running_corrects_top5 += (torch.sum(labels == torch.topk(torch.exp(output),5).indices) ).numpy()

            print("\r", f'{i}/{dataloaders_sizes_test}', end='')
    

    cf_matrix = confusion_matrix(y_true, y_pred, normalize = 'true')
    print(len(cf_matrix[0]))
    print(class_names)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *len(class_names), index = [i for i in class_names],
                        columns = [i for i in class_names])

    df_cm[df_cm.eq(0)] = np.nan
    mask = df_cm.isnull()
    plt.figure(figsize = (12,7))

    font_size_0 = 6

    res = sn.heatmap(df_cm, annot=True, cmap='flare', linewidths=.2, mask=df_cm.isnull(), fmt = '.2%', yticklabels = True, annot_kws={"size":font_size_0, "weight": "bold"}, cbar_kws={"pad":0.01}, square=True)
    res.set_facecolor('#E8E8E8')
    

    cbar = res.collections[0].colorbar
    cbar.set_ticks([0, .25, .5, .75, 1])
    cbar.set_ticklabels(['low', '25%', '50%', '75%', '100%'])
    cbar.ax.tick_params(labelsize=8)
    
    font_size_1 = 12
    font_size_2 = 8

    res.set_xticklabels(res.get_xmajorticklabels(), fontsize = font_size_2, style='italic', rotation=90, horizontalalignment='center')
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = font_size_2, style='italic', rotation="horizontal")

    plt.xlabel("PREDICTED CLASS", fontsize = font_size_1, labelpad= 5, fontweight='bold')
    plt.ylabel("TRUE CLASS", fontsize = font_size_1, labelpad = 5, fontweight='bold')

    ##################################

    print('Number all test samples: {}'.format(dataset_sizes_test))
    print('_'*50)
    print('Number correct predict top1 test samples: {}'.format(running_corrects_top1))

    print('Top 1 acc: {:3.2f}%'.format((running_corrects_top1/dataset_sizes_test)*100))
    print('_'*50)
    print('Number correct predict top1 test samples: {}'.format(running_corrects_top3))

    print('Top 3 acc: {:3.2f}%'.format((running_corrects_top3/dataset_sizes_test)*100))
    print('_'*50)
    print('Number correct predict top5 test samples: {}'.format(running_corrects_top5))

    print('Top 5 acc: {:3.2f}%'.format((running_corrects_top5/dataset_sizes_test)*100))

    print(sklearn.metrics.classification_report(y_true, y_pred, target_names=class_names, digits=4))

    plt.subplots_adjust(top = 0.99, bottom = 0.2, right = 0.9, left = 0.1, hspace = 0, wspace = 0)
            
    plt.savefig("conf_matrix.png")
    plt.show(block = True)

    
    fig = res.get_figure()
    fig.savefig("conf_matrix.pdf") 

    return y_true, y_pred
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import random


main_path = "C:\\Users\\przemeko\\Desktop\\classification\\data\\images"
list_dir = os.listdir(main_path)


print(list_dir)
min_images = 400

product_dict = {}
product_dict_names = {}
product_done = []
product_undone = []
num_all_images = 0

list_to_save = []


for i, product in enumerate(list_dir):
    list_folders_product = os.listdir(os.path.join(main_path, product))

    files_full_path = glob.glob(os.path.join(main_path, product, "*"))
    num_images_for_product = len(files_full_path)
    num_all_images +=num_images_for_product

    to_save_files = [[product,str(i), s]  for s in files_full_path]
    list_to_save += to_save_files

    product_dict_names[product] = num_images_for_product


f, (ax1) = plt.subplots(1,1, figsize=(13,5), sharex=True)

y1 = np.array(list(product_dict_names.values()))
x = np.array(list(product_dict_names.keys()))


colors_all = sns.color_palette("dark:#fb7b61_r", len(x))

actual_fruit = 0
actual_sweets = 0
actual_rest = 0

show_string = "Total number of product: {}".format(len(list_dir)) + " |  Total number of images: {: d}".format(num_all_images) + "\n "

sns.barplot(x=x, y= y1, palette=colors_all, ax = ax1).set(title=show_string)

    
for i, row in enumerate(y1):
    if row >0:
        ax1.text(i, (row), row, color='white', ha='center', va = "top", rotation = "vertical")

ax1.axhline(min_images, color = "red")
ax1.set(xlabel='Classes\n', ylabel='Num. images\n')

x_labels = []

for i, label in enumerate(x):
    x_labels.append("{}. {}".format(i+1, label))

plt.xticks(ticks = np.linspace(0, len(x)-1, num =len(x)), labels = x_labels, rotation = 90)
plt.ylim((0,max(y1)+150))
sns.despine(bottom=True)
plt.setp(f.axes)
plt.grid(axis="y")
plt.tight_layout(h_pad=0.1)
plt.savefig('data\\dataset_class_counts.png')
plt.show()

explode = (0.05,0.05,0.05,0.05,0.05,0.05)

plt.title('Ratio number of images \n')

_, _, autotexts = plt.pie(y1, labels = x, colors = colors_all, autopct='%.0f%%', explode = explode)

for autotext in autotexts:
    autotext.set_color('white')

plt.savefig('data\\dataset_ratio_counts.png')
plt.show()

random.shuffle(list_to_save)

list_to_save_train = list_to_save[:int(0.8*len(list_to_save))]
list_to_save_test = list_to_save[int(0.8*len(list_to_save)):]


with open('data\\classes_dataset_train.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter =',')
    writer.writerows(list_to_save_train)

with open('data\\classes_dataset_test.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter =',')
    writer.writerows(list_to_save_test)
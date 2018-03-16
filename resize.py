import PIL.Image as Image
import os
import random
import numpy as np
import h5py
import pandas


def _preprocess_and_save(features, labels, filename):
    file = h5py.File(filename, 'w')
    file['features'] = features
    file['labels'] = labels
    file.close()
    print("write finish")



train_paths = []
valid_paths = []
csv = pandas.read_csv('driver_imgs_list.csv')
groupby = csv.groupby(['subject'])
for key in groupby.indices.keys():
    pic_arr = groupby.indices[key]
    current_data = train_paths if len(train_paths) < 19000 else valid_paths
    for index in pic_arr:
        data = []
        col = csv.iloc[index]
        img_path = col['img']
        clazz = col['classname']
        data.append(img_path)
        data.append(clazz)
        current_data.append(data)

current_path = os.getcwd()
dir_path = current_path + '/imgs/train'


def getData(paths):
    arr = []
    for path in paths:
        img_path = dir_path + '/' + path[1] + '/' + path[0]
        image = Image.open(open(img_path, 'rb'))
        image = image.resize((224, 224), Image.ANTIALIAS)
        data = []
        data.append(np.array(image))
        data.append([int(path[1][1])])
        arr.append(data)
    return arr


train_datas = getData(train_paths)

valid_datas = getData(valid_paths)

random.shuffle(train_datas)
random.shuffle(valid_datas)

print("load finish!!")
feature = list(data[0] for data in train_datas)
label = list(data[1][0] for data in train_datas)

valid_features = list(data[0] for data in valid_datas)
valid_labels = list(data[1][0] for data in valid_datas)

arr_count = 5
arr_size = len(feature) / arr_count
for i in range(arr_count):
    start = int(i * arr_size)
    end = int(min(start + arr_size, len(feature)))
    _preprocess_and_save(feature[start:end], label[start:end], 'train_data_{}.h5'.format(i + 1))

_preprocess_and_save(valid_features, valid_labels, 'validation_data.h5')

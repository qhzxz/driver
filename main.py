from keras.optimizers import SGD, Adam
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, Input, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import applications
from keras.models import Model, Sequential
from keras.regularizers import l2
import gc
import h5py
import numpy as np
import PIL.Image as Image

train_features = []
train_label = []
for i in range(5):
    file = h5py.File('train_data_' + str(i + 1) + '.h5')
    f = file['features'][:]
    l = file['labels'][:]
    for j in range(len(f)):
        train_features.append(f[j])
        train_label.append(l[j])
    file.close()

train_features = np.asarray(train_features)
train_label = np_utils.to_categorical(np.asarray(train_label))

mean = [103.939, 116.779, 123.68]


def normalize(x):
    x = x.astype('float32')
    x[:, :, 0] = (x[:, :, 0] - mean[0])
    x[:, :, 1] = (x[:, :, 1] - mean[1])
    x[:, :, 2] = (x[:, :, 2] - mean[2])
    return x


file = h5py.File('validation_data.h5', 'r')
valid_features, valid_labels = np.asarray(file['features'][:]).astype('float32'), np_utils.to_categorical(
    np.asarray(file['labels'][:]))
file.close()
print(valid_features.shape)

print("load finish!!!")

print(gc.collect())

print(valid_features[0].shape)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

##############################
## Build the Neural Network ##
##############################


print(gc.collect())

train_generator = ImageDataGenerator(
    preprocessing_function=normalize)

test_generator = ImageDataGenerator(preprocessing_function=normalize)

train_generator = ImageDataGenerator(
    preprocessing_function=normalize)

test_generator = ImageDataGenerator(preprocessing_function=normalize)


def get_custom_model(num_class):
    base_model = applications.VGG16(include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[:4]:
        print(layer.name)
        layer.trainable = False
    x = GlobalAveragePooling2D(name='global')(base_model.get_layer('block5_pool').output)
#     x = Dense(20, activation='linear', name='fc1', kernel_regularizer=l2(0.01))(x)
    x= Dense(512,activation='relu',name='fc1')(x)
    x = Dense(num_class, activation='softmax', name='pred')(x)
    model = Model(input=base_model.input, output=x)
    sgd = SGD(lr=1e-3, momentum=0.9, decay=1e-6, nesterov=True)
    #     sgd = SGD(lr=1e-10)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


custom_model = get_custom_model(10)

print("compile finish!!!")

custom_model.fit_generator(train_generator.flow(train_features, train_label, batch_size=128),
                           use_multiprocessing=True,
                           epochs=20,
                           shuffle=True,
                           validation_data=test_generator.flow(valid_features, valid_labels, batch_size=128),
                           verbose=1)

custom_model.save_weights('weight_1.meta')

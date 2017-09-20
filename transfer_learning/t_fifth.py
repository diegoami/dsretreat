import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model.h5'
root_dir = '/media/diegoami/40e5135e-5905-41f3-a006-2cd73b52e803/datasets/catexp/'
train_data_dir = root_dir + 'train/'
validation_data_dir = root_dir + 'validation/'

nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16


features_train = np.load('features_train.npy' )


train_labels = np.array(
    [0] * 900 + [1] * 900
)

second_model = Sequential()
second_model.add(Flatten(input_shape=features_train.shape[1:]))
second_model.add(Dense(256, activation='relu'))
second_model.add(Dropout(0.5))
second_model.add(Dense(1, activation='sigmoid'))
second_model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', metrics=['accuracy'])

second_model.fit(features_train, train_labels,
          epochs=epochs,
          batch_size=batch_size)
second_model.save('second_model.npy')





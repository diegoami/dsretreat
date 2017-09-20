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

nb_train_samples = 1800
nb_validation_samples = 800
epochs = 50
batch_size = 18

model = ResNet50(weights='imagenet',  include_top=False)
datagen = ImageDataGenerator(rescale=1. / 255)
generator = datagen.flow_from_directory(
    train_data_dir,
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
features_train = model.predict_generator(
    generator, nb_train_samples // batch_size)

import numpy as np
print(features_train.shape)
np.save('features_train.npy', features_train)

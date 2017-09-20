from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import matplotlib
matplotlib.use('Agg')
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda,floatX=float32"

from keras import models
from keras import layers
from keras import optimizers
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications import ResNet50
from keras import backend as K



train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_data_dir,
        # All images will be resized to 150x150
        target_size=(197, 197),
        batch_size=10,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')
print(train_generator)
print(train_generator.image_shape)


validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(197, 197),
        batch_size=10,
        class_mode='categorical')

conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(197 ,197 ,3))



epochs = 50
batch_size = 8

model = models.Sequential()
model.add(conv_base)
model.add(Flatten(name='flatten'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(100, activation='sigmoid'))

print(model.summary())

print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))
conv_base.trainable = False
print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))
print(model.summary())





model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=20,
      validation_data=validation_generator,
      validation_steps=50)

model.save('keras.model2')



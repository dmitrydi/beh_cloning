import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import keras as K
import tensorflow as tf
import os, json
from scipy import misc
import cv2
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.applications import VGG16
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.models import Sequential, Model

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('image_data_path', '.', "root data folder")
flags.DEFINE_string('datafile', 'driving_log.csv', ".csv file with images data")
flags.DEFINE_integer('batch_size', 4, "batch size")
flags.DEFINE_integer('image_width', 320, "image width")
flags.DEFINE_integer('image_height', 160, "image height")
flags.DEFINE_integer('crop_from_top', 50, "pixels to crop from top of image")
flags.DEFINE_integer('crop_from_bottom', 20, "pixels to crop from the bottom of image")
flags.DEFINE_float('train_size', 0.8, "train size, <=1")
flags.DEFINE_integer('epochs', 1, "number of epochs")
flags.DEFINE_string('name', 'VGG16', "name of the model")
flags.DEFINE_string('saving_path', 'trained_weights', "path to save model")
flags.DEFINE_string('saving_name', 'model', "name of model save")
flags.DEFINE_string('data_mode', 'csv', "feed data mode: csv - data from csv, p - data from pickled dataframe")
flags.DEFINE_string('train_df', 'train_df.p', "train dataset pickle file")
flags.DEFINE_string('test_df', 'test_df.p', "test dataset pickle file")
flags.DEFINE_string('data_columns', 'center', "image data column in data file")
flags.DEFINE_string('val_column', 'steering', "steering value column")


IMAGE_DATA_PATH = FLAGS.image_data_path
datafile = FLAGS.datafile
batch_size = FLAGS.batch_size
w = FLAGS.image_width
h = FLAGS.image_height
crop_from_top = FLAGS.crop_from_top
crop_from_bottom = FLAGS.crop_from_bottom
train_size = FLAGS.train_size
epochs = FLAGS.epochs
name = FLAGS.name
saving_path = FLAGS.saving_path
saving_name = '{}_{}_{}_{}.h5'.format(FLAGS.saving_name, name, batch_size, epochs)
data_mode = FLAGS.data_mode
train_df_file = FLAGS.train_df
test_df_file = FLAGS.test_df
data_columns = FLAGS.data_columns.split('-')
val_column = FLAGS.val_column

image_shape = (h, w, 3)

if data_mode == 'csv':
    image_data = pd.read_csv(os.path.join(IMAGE_DATA_PATH, datafile))
    ind_train, ind_test = train_test_split(np.array(range(len(image_data))), train_size=train_size, random_state = 42)
    train_df = image_data.loc[ind_train]
    test_df = image_data.loc[ind_test]
elif data_mode == 'p':
    with open(train_df_file, 'rb') as f:
        train_df = pickle.load(f)
    with open(test_df_file, 'rb') as f:
        test_df = pickle.load(f)
else:
    ValueError

def get_VGG16(input_shape = (160,320,3), crop_from_top=50, crop_from_bottom=20):
    bottom = Sequential()
    bottom.add(Lambda(lambda x: x/255. - 0.5, input_shape=input_shape))
    bottom.add(Cropping2D(cropping=((crop_from_top,crop_from_bottom), (0,0)), input_shape=input_shape))
    m_bottom_model = Model(bottom.input, bottom.output)
    initial_model = VGG16(weights="imagenet", include_top=False, input_shape=m_bottom_model.output.get_shape().as_list()[1:])(m_bottom_model.output)
    x = Flatten()(initial_model)
    x = Dense(256, activation='relu')(x)
    output = Dense(1)(x)
    model = Model(m_bottom_model.input, output)

    for layer in model.layers[3].layers:
        layer.trainable = False

    return model

def get_VGG16_e(input_shape = (160,320,3), crop_from_top=50, crop_from_bottom=20):
    bottom = Sequential()
    bottom.add(Lambda(lambda x: x/255. - 0.5, input_shape=input_shape))
    bottom.add(Cropping2D(cropping=((crop_from_top,crop_from_bottom), (0,0)), input_shape=input_shape))
    m_bottom_model = Model(bottom.input, bottom.output)
    initial_model = VGG16(weights="imagenet", include_top=False, input_shape=m_bottom_model.output.get_shape().as_list()[1:])(m_bottom_model.output)
    x = Flatten()(initial_model)
    x = Dense(1024, activation='relu')(x)
    x = Dense(256)(x)
    output = Dense(1)(x)
    model = Model(m_bottom_model.input, output)

    for layer in model.layers[3].layers:
        layer.trainable = False

    return model

def get_NVIDIA(input_shape = (160,320,3), crop_from_top=50, crop_from_bottom=20):
    model = Sequential()
    model.add(Cropping2D(cropping=((crop_from_top,crop_from_bottom), (0,0)), input_shape=input_shape))
    model.add(Lambda(lambda x: x/255. - 0.5, input_shape=input_shape))
    model.add(Conv2D(24, (5,5), strides=(2,2), padding='VALID'))
    model.add(Conv2D(36, (5,5), strides=(2,2), padding='VALID'))
    model.add(Conv2D(48, (3,3), padding='SAME'))
    model.add(Conv2D(64, (3,3), padding='SAME'))
    model.add(Conv2D(64, (3,3), padding='SAME'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model

def get_model(name, input_shape = (160,320,3), crop_from_top=50, crop_from_bottom=20):
    if name == 'VGG16':
        return get_VGG16(input_shape = input_shape, crop_from_top = crop_from_top, crop_from_bottom = crop_from_bottom)
    elif name == 'VGG16_e':
        return get_VGG16_e(input_shape = input_shape, crop_from_top = crop_from_top, crop_from_bottom = crop_from_bottom)
    elif name == 'NVIDIA':
    	return get_NVIDIA(input_shape = input_shape, crop_from_top = crop_from_top, crop_from_bottom = crop_from_bottom)
    else:
        ValueError
        
    return model

def generator_df(samples_df_, source_path='data', data_columns = ['center'], val_column = 'steering', batch_size=4):
# yields batches from dataframe samples_df: ['images', 'steering']
    samples_df = samples_df_.copy()
    num_samples = len(samples_df)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples_df)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples_df[offset:offset+batch_size]

            images = None
            angles = np.array([], dtype='float32')
            for i, batch_sample in batch_samples.iterrows():
                name = batch_sample[np.random.choice(data_columns, 1)].values[0]
                name = name.strip()
                center_image = cv2.imread(os.path.join(source_path,name))
                if center_image is not None:
                    center_angle = batch_sample[val_column]
                    if images is None:
                        images = center_image[np.newaxis]
                    else:
                        images = np.vstack([images, center_image[np.newaxis]])
                    angles = np.append(angles, center_angle)

            yield images, angles

model = get_model(name, input_shape=image_shape, crop_from_top=crop_from_top, crop_from_bottom=crop_from_bottom)
model.compile(loss='mse', optimizer='adam')

train_gen = generator_df(train_df, batch_size=batch_size, source_path=IMAGE_DATA_PATH,
                         data_columns=data_columns, val_column=val_column)
val_gen = generator_df(test_df, batch_size=batch_size, source_path=IMAGE_DATA_PATH,
                      data_columns=data_columns, val_column=val_column)

history = model.fit_generator(train_gen, steps_per_epoch=len(train_df)/batch_size, validation_data=val_gen,
                              validation_steps=len(test_df)/batch_size, epochs=epochs)

if not os.path.exists(saving_path):
    os.mkdir(saving_path)

model.save(os.path.join(saving_path, saving_name))




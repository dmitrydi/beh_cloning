from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model


def get_NVIDIA(input_shape = (160,320,3), crop_from_top=50, crop_from_bottom=20):
    model = Sequential()
    model.add(Cropping2D(cropping=((crop_from_top,crop_from_bottom), (0,0)), input_shape=input_shape))
    model.add(Lambda(lambda x: x/255. - 0.5, input_shape=input_shape))
    model.add(Conv2D(24, (5,5), strides=(2,2), padding='VALID'))
    model.add(MaxPooling2D())
    model.add(Conv2D(36, (5,5), strides=(2,2), padding='VALID'))
    model.add(MaxPooling2D())
    model.add(Conv2D(48, (3,3), padding='SAME'))
    model.add(Conv2D(64, (3,3), padding='SAME'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3,3), padding='SAME'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))

    return model
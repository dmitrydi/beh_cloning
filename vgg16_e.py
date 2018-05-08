from keras.applications import VGG16
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D, Conv2D
from keras.models import Sequential, Model

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
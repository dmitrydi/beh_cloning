from keras.applications import VGG16
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D, Conv2D
from keras.models import Sequential, Model


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
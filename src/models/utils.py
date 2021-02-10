from keras.layers import Dense, Convolution2D, MaxPool2D, BatchNormalization, Dropout, MaxPooling2D, Flatten
from keras.layers.advanced_activations import LeakyReLU

def conv_block(x ,n_filters, kernel_size=(2,2),n_convs=2):
    """(Conv2d->LeakyReLU->BatchNorm)*N -> MaxPool"""
    for i in range(n_convs):
        x = Convolution2D(n_filters,kernel_size, padding='same', use_bias=False)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    return x

def vanilla_head(x, num_classes):
    x = Flatten()(x)
    x = Dense(512,activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(num_classes)(x)
    return x

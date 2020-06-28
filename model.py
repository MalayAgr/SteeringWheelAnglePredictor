from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import (Dense, Conv2D, BatchNormalization,
                                     ReLU, ELU, LeakyReLU, Dropout, Flatten)
import matplotlib.pyplot as plt
from tensorflow.keras.initializers import Constant
from processing import *


def activation_layer(ip, activation):
    return {'relu': ReLU()(ip),
            'elu': ELU()(ip),
            'lrelu': LeakyReLU()(ip)}[activation]


def conv2D(
    ip, filters, kernel_size, strides, layer_num, activation,
    kernel_initializer='he_uniform', bias_val=0.01
):

    conv_name = f'conv{layer_num}_{filters}_{kernel_size[0]}_{strides[0]}'
    bn_name = f'bn{layer_num}'

    layer = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   kernel_initializer=kernel_initializer,
                   bias_initializer=Constant(value=bias_val),
                   name=conv_name,)(ip)

    layer = BatchNormalization(name=bn_name)(layer)
    return activation_layer(ip=layer, activation=activation)


def fullyconnected_layers(
    ip, activation, inititalizer='he_uniform', bias_val=0.01
):

    layer = Dense(
        100, kernel_initializer=inititalizer,
        bias_initializer=Constant(value=bias_val), name='dense1'
    )(ip)

    layer = activation_layer(ip=layer, activation=activation)

    layer = Dense(
        50, kernel_initializer=inititalizer,
        bias_initializer=Constant(value=bias_val), name='dense2'
    )(layer)

    layer = activation_layer(ip=layer, activation=activation)

    return Dense(
        10, kernel_initializer=inititalizer,
        bias_initializer=Constant(value=bias_val), name='dense3'
    )(layer)

    return activation_layer(ip=layer, activation=activation)


def build_model(
    ip=Input(shape=(128, 128, 3)), activation='relu', dropout=0.5,
    compile_model=True, lr=1e-3
):

    layer = conv2D(
        ip, filters=24, kernel_size=(5, 5), strides=(2, 2), layer_num=1,
        activation=activation
    )

    layer = conv2D(
        layer, filters=36, kernel_size=(5, 5), strides=(2, 2), layer_num=2,
        activation=activation
    )

    layer = conv2D(
        layer, filters=48, kernel_size=(5, 5), strides=(2, 2), layer_num=3,
        activation=activation
    )

    layer = conv2D(
        layer, filters=64, kernel_size=(3, 3), strides=(1, 1), layer_num=4,
        activation=activation
    )

    layer = conv2D(
        layer, filters=64, kernel_size=(3, 3), strides=(1, 1), layer_num=5,
        activation=activation
    )

    layer = Dropout(dropout)(layer)

    layer = Flatten()(layer)

    layer = fullyconnected_layers(layer, activation=activation)
    op_layer = Dense(1, name="op_layer")(layer)

    model = Model(ip, op_layer)
    if compile_model:
        model.compile(loss='mse', optimizer=Adam(lr=lr))
    return model


def get_batch(image_paths, labels, batch_size, is_training=False):
    while True:
        indices = np.random.choice(len(image_paths), batch_size, replace=False)
        images = vectorized_imread(image_paths[indices])
        if is_training:
            images, final_labels = augment_images(images, labels[indices])
        else:
            final_labels = labels[indices]
        yield preprocess(images), final_labels


def plot_model_history(model):
    plt.plot(model.history.history['loss'], 'r', label='train')
    plt.plot(model.history.history['val_loss'], 'g', label='validation')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()


def train_model(
    model, im_train, labels_train, im_val, labels_val, batch_size=64,
    epochs=50, plot_history=True
):
    model.fit_generator(
        get_batch(im_train, labels_train, batch_size, is_training=True),
        steps_per_epoch=len(im_train) // batch_size,
        epochs=epochs,
        validation_data=get_batch(im_val, labels_val, batch_size),
        validation_steps=len(im_val) // batch_size,
        verbose=1
    )

    if plot_history:
        plot_model_history(model)

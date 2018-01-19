from typing import List, Union, Optional
from keras.models import Sequential
from keras.layers import Flatten
from collections import namedtuple
from keras_implementation.neural_net import dense_model


conv_operation = namedtuple("conv_operation",
                            field_names=[
                                # convolution layer parameters
                                "conv_dim", "filters",
                                "kernel_size", "strides",
                                "padding", "activation",
                                # pooling layer parameters
                                "pool_type", "pool_size"
                            ])


def cnn_model(conv_layout: Union[conv_operation, List[conv_operation]],
              input_shape=None,
              pre_model=None,
              dense_nb_neurons: Optional[List[int]]=None,
              dense_activations=None,
              name: Optional[str]=None,
              *args, **kwargs
              ) -> Sequential:
    """
    cnn model with various number of convolution and pooling layer
    + FFN
    :param conv_layout: list of convolution + pooling layer with relevant parameters
    :param input_shape: shape of input data
    :param pre_model: a keras model to iterate on
    :param dense_nb_neurons: nb of neurons per layer in the dense post-net
    :param dense_activations: type of activation per layer in the dense post-net
    :param name: name of the neural network
    :param args: extra param to dense model
    :param kwargs: extra param to dense model
    :return: keras Sequential model
    """

    model = Sequential(name=name) if not pre_model else pre_model

    if not isinstance(conv_layout, list):
        conv_layout = [conv_layout]

    # first layer
    if input_shape and not pre_model:
        conv_op = conv_layout[0]
        model.add(conv_op.conv_dim(filters=conv_op.filters,
                                   kernel_size=conv_op.kernel_size,
                                   strides=conv_op.strides,
                                   activation=conv_op.activation,
                                   padding=conv_op.padding,
                                   input_shape=input_shape))
        if conv_op.pool_type:
            model.add(conv_op.pool_type(pool_size=conv_op.pool_size))

    # following layers
    for conv, filters, kernel_size, strides, padding, activation, \
            pool_type, pool_size in conv_layout[1:]:
        model.add(conv(filters=filters, strides=strides,
                       kernel_size=kernel_size, padding=padding,
                       activation=activation))
        if pool_type:
            model.add(pool_type(pool_size=pool_size))

    model.add(Flatten())

    return dense_model(nb_units=dense_nb_neurons,
                       activations=dense_activations,
                       pre_model=model,
                       *args, **kwargs)

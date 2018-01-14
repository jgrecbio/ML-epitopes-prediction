from typing import List, Union
from keras.models import Sequential
from keras.layers import Dense


def dense_model(nb_units: Union[List[int], int],
                activations: Union[List, object],
                input_shape=None,
                pre_model=None,
                ) -> Sequential:
    """
    Create a feed-forward network with various number of layers and neurons
    :param input_shape: shape of input data
    :param nb_units: number of neurons per layer or list of number of neurons per layer
    :param activations: activation function to use or list of activation to use per layer
    :param: pre_model: a keras model to iterate on
    :return: keras Sequential model
    """

    if isinstance(nb_units, List) and isinstance(activations, list):
        if len(nb_units) != len(activations):
            raise ValueError("Different numbers of activations and layers")
        nb_layers = max([len(nb_units), len(activations)])
    else:
        nb_layers = 1

    if isinstance(nb_units, int):
        units = [nb_units] * nb_layers
    else:
        units = nb_units
    if not isinstance(activations, list):
        acts = [activations] * nb_layers
    else:
        acts = activations

    model = Sequential() if not pre_model else pre_model
    print(model.summary())
    if input_shape and not pre_model:
        model.add(Dense(input_shape=input_shape, units=units[0], activation=acts[0]))
    else:
        model.add(Dense(units=units[0], activation=acts[0]))

    for nb_neurons, activation in zip(units[1:], acts[1:]):
        model.add(Dense(units=nb_neurons, activation=activation))
    model.summary()
    return model


def compile_model(model: Sequential, optimizer, loss,
                  *args, **kwargs):
    """
    compilation from keras
    :param model: keras Sequential model
    :param optimizer: keras optimizer to use
    :param loss: loss function
    :param args: args to pass to the optimizer
    :param kwargs: kwargs to pass to the optimizer
    :return: compiled sequential model
    """
    mod = model
    mod.compile(optimizer=optimizer(*args, **kwargs), loss=loss)
    return mod

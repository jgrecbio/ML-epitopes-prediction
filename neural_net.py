from typing import List, Union, Optional
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l1, l2, l1_l2


def assert_len(num_or_list: Union[Union[int, float], Union[List[int], List[float]]],
               nb_layers: int,
               type_error_message: Optional[str]=None):
    if isinstance(num_or_list, list):
        if len(num_or_list) != nb_layers:
            raise ValueError(type_error_message)
        else:
            return num_or_list
    elif isinstance(num_or_list, float) or isinstance(num_or_list, int):
        return [num_or_list] * nb_layers
    else:
        return [None] * nb_layers


def set_regularization(l1s: List[float], l2s: List[float]):
    for reg_l1, reg_l2 in zip(l1s, l2s):
        if reg_l1 == 0. and reg_l2 == 0:
            yield None
        elif reg_l1 == 0.:
            yield l2(reg_l2)
        elif reg_l2 == 0.:
            yield l1(reg_l1)
        else:
            yield l1_l2(reg_l1, reg_l2)


def dense_model(nb_units: Union[List[int], int],
                activations: Union[List, object],
                l1_regularization: Optional[Union[List[float], float]]=0.,
                l2_regularization: Optional[Union[List[float], float]]=0.,
                dropout_reg: Optional[Union[List[Union[float, None]], float]]=None,
                input_shape=None,
                pre_model=None,
                ) -> Sequential:
    """
    Create a feed-forward network with various number of layers and neurons
    :param input_shape: shape of input data
    :param nb_units: number of neurons per layer or list of number of neurons per layer
    :param activations: activation function to use or list of activation to use per layer
    :param l1_regularization: l1 reg coef
    :param l2_regularization: l2 reg coef
    :param: dropout_reg: dropout rates
    :param: pre_model: a keras model to iterate on
    :return: keras Sequential model
    """

    if isinstance(nb_units, List) and isinstance(activations, list):
        if len(nb_units) != len(activations):
            raise ValueError("Different numbers of activations and layers")
        nb_layers = max([len(nb_units), len(activations)])
    else:
        nb_layers = 1

    units = assert_len(nb_units, nb_layers, "Not all layers are specified for neurons count")
    acts = assert_len(activations, nb_layers, "Not all layers are specified for activations count")
    l1_reg = assert_len(l1_regularization, nb_layers, "Not all layers are properly regularized")
    l2_reg = assert_len(l2_regularization, nb_layers, "Not all layers are properly regularized")
    dropout = assert_len(dropout_reg, nb_layers, "Not all layers have a proper dropout rate")
    print(dropout)

    reg = set_regularization(l1_reg, l2_reg)

    model = Sequential() if not pre_model else pre_model
    if input_shape and not pre_model:
        model.add(Dense(input_shape=input_shape, units=units[0], activation=acts[0],
                        kernel_regularizer=next(reg)))
    else:
        model.add(Dense(units=units[0], activation=acts[0],
                        kernel_regularizer=next(reg)))
    if dropout[0]:
        model.add(Dropout(dropout[0]))

    for nb_neurons, activation, kernel_reg, dr in zip(units[1:], acts[1:], reg, dropout):
        model.add(Dense(units=nb_neurons, activation=activation,
                        kernel_regularizer=kernel_reg))
        if dr:
            model.add(Dropout(dr))
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

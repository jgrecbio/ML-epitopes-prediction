from typing import List, Union, Optional
from keras.models import Sequential
from keras.optimizers import adam
from keras.losses import mean_squared_error
from keras.layers import Dense, Dropout
from nn_utils import get_optimizer_params, set_regularization, assert_len


def dense_model(
        # nn architecture
        nb_units: Union[List[int], int],
        activations: Union[List, object],
        l1_regularization: Optional[Union[List[float], float]] = 0.,
        l2_regularization: Optional[Union[List[float], float]] = 0.,
        dropout_reg: Optional[Union[List[Union[float, None]], float]] = None,
        input_shape=None,
        pre_model=None,

        # nn hyper parameters
        optimizer=adam,
        loss=mean_squared_error,
        learning_rate: float = 0.01,
        momentum_1: float = 0.95,
        momentum_2: float = 0.999,
        epsilon: float = 1e-8,
        decay: float = 0.,
        rho: float = 0.9,

        # name
        name: Optional[str]=None,
) -> Sequential:
    """
    Generation of fully connected layer
    :param nb_units: nb of units per layer
    :param activations: type of activation per layer
    :param l1_regularization: l1 coef per layer
    :param l2_regularization: l2 coef per layer
    :param dropout_reg: dropout rate per dropout layer
    :param input_shape: shape of input data
    :param pre_model: usage of un-compiled pre-net
    :param optimizer: optimizer to use
    :param loss: loss function
    :param learning_rate: learning rate used with optimizer
    :param momentum_1: if required, beta 1 momentum
    :param momentum_2: if required, beta 2 momentum
    :param epsilon: if required, optimize epsilon
    :param decay: if required, optimizer decay
    :param rho: if required, optimizer rho
    :param name: name of the neural network
    :return: corresponding keras Sequential model
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

    model = Sequential(name=name) if not pre_model else pre_model
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

    compile_params = get_optimizer_params(optimizer, learning_rate, momentum_1, momentum_2, epsilon, decay, rho)
    model.compile(optimizer=optimizer(**compile_params), loss=loss)

    return model

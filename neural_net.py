from toolz import merge
from typing import List, Union, Optional
from keras.models import Sequential
from keras.optimizers import adam, Adam, RMSprop, SGD, Adadelta, Adagrad, Adamax, Nadam, TFOptimizer, sgd
from keras.losses import mean_squared_error
from keras.layers import Dense, Dropout
from keras.regularizers import l1, l2, l1_l2


def assert_len(num_or_list: Union[Union[int, float], Union[List[int], List[float]]],
               nb_layers: int,
               type_error_message: Optional[str] = None):
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


def get_optimizer_params(opt,
                         lr: float, m1: float, m2: float, epsilon: float, decay: float,
                         rho: float):
    opti = opt(lr=lr)
    min_dict = {"lr": lr}
    if isinstance(opti, SGD):
        return min_dict
    elif type(opti) in {Adam, Adamax}:
        return merge(min_dict, {"beta_1": m1, "beta_2": m2,
                                "epsilon": epsilon, "decay": decay})
    elif type(opti) in {RMSprop, Adadelta}:
        return merge(min_dict, {"rho": rho, "epsilon": epsilon,
                                "decay": decay})
    elif isinstance(opti, Adagrad):
        return merge(min_dict, {"epsilon": epsilon, "decay": decay})
    elif isinstance(opti, Nadam):
        return merge(min_dict, {"beta_1": m1, "beta_2": m2, "schedule_decay": decay,
                                "epsilon": epsilon})
    else:
        raise ValueError


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
) -> Sequential:
    """

    :param nb_units:
    :param activations:
    :param l1_regularization:
    :param l2_regularization:
    :param dropout_reg:
    :param input_shape:
    :param pre_model:
    :param optimizer:
    :param loss:
    :param learning_rate:
    :param momentum_1:
    :param momentum_2:
    :param epsilon:
    :param decay:
    :param rho:
    :return:
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

    compile_params = get_optimizer_params(optimizer, learning_rate, momentum_1, momentum_2, epsilon, decay, rho)
    print(compile_params)
    model.compile(optimizer=optimizer(**compile_params), loss=loss)

    return model

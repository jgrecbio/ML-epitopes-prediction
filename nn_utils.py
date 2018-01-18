from typing import Union, List, Optional
from toolz import merge
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam, Adadelta, Adagrad, Adamax, Nadam, SGD, RMSprop


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

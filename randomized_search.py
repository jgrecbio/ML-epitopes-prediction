from typing import Callable
from keras.wrappers.scikit_learn import KerasRegressor


def to_regressor(model_fn: Callable, **kwargs) -> KerasRegressor:
    """
    Given a function that creates a keras model, it yield a wrapped
    scikit-learn compatible model
    :param model_fn: function the creates Sequential keras model
    :param kwargs: parameters of the model
    :return: a wrapped scikit-learn compatible model
    """
    return KerasRegressor(model_fn, **kwargs)

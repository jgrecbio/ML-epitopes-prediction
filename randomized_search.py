from typing import Callable
from keras.wrappers.scikit_learn import KerasRegressor
from neural_net import dense_model, compile_model
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Reshape
from keras.activations import relu, linear
from keras.optimizers import adam
from keras.losses import mean_squared_error
from rnn import rnn_model
from cnn import cnn_model, conv_operation


def to_regressor(model_fn: Callable, **kwargs) -> KerasRegressor:
    """
    Given a function that creates a keras model, it yield a wrapped
    scikit-learn compatible model
    :param model_fn: function the creates Sequential keras model
    :param kwargs: parameters of the model
    :return: a wrapped scikit-learn compatible model
    """
    return KerasRegressor(model_fn, **kwargs)


# fully connected networks
dense1 = dense_model([40, 40, 1], [relu, relu, linear], dropout_reg=0.3, input_shape=(180,))
dense2 = dense_model([80, 80, 1], [relu, relu, linear], dropout_reg=0.3, input_shape=(180,))
dense3 = dense_model([120, 80, 1], [relu, relu, linear], dropout_reg=0.3, input_shape=(180,))
dense4 = dense_model([80, 80, 80, 1], [relu, relu, linear], dropout_reg=0.3, input_shape=(180,))

# cnn networks
reshape_model = Sequential(layers=[Reshape(input_shape=(180,), target_shape=(180, 1))])

cnn1 = cnn_model(conv_layout=conv_operation(Conv1D, 32, 4, 1, "valid", relu, MaxPooling1D, 2),
                 pre_model=reshape_model, dense_nb_neurons=[40, 1], dense_activations=[relu, linear])
cnn2 = cnn_model(conv_layout=conv_operation(Conv1D, 64, 2, 1, "valid", relu, MaxPooling1D, 2),
                 pre_model=reshape_model, dense_nb_neurons=[120, 1], dense_activations=[relu, linear])
cnn3 = cnn_model(conv_layout=conv_operation(Conv1D, 64, 4, 1, "valid", relu, MaxPooling1D, 4),
                 pre_model=reshape_model, dense_nb_neurons=[120, 1], dense_activations=[relu, linear])

# rnn model
rnn1 = rnn_model(nb_rnn_neurons=[32, 32], dense_nb_neurons=[40, 1], dense_activations=[relu, linear])
rnn2 = rnn_model(nb_rnn_neurons=[64, 64], dense_nb_neurons=[40, 1], dense_activations=[relu, linear])
rnn3 = rnn_model(nb_rnn_neurons=[128, 64], dense_nb_neurons=[40, 1], dense_activations=[relu, linear])

nets = [dense1, dense2, dense3, dense4,
        cnn1, cnn2, cnn3,
        rnn1, rnn2, rnn3]
compiler = compile_model(optimizer=adam, loss=mean_squared_error)
c_nets = list()

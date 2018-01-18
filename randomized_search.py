from typing import Callable

# scikit-learn imports
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error

# Keras imports
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Reshape
from keras.activations import relu, linear
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import TensorBoard

# import models
from neural_net import dense_model
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

nns = [dense1, dense2, dense3, dense4,
       cnn1, cnn2, cnn3,
       rnn1, rnn2, rnn3]


def grid_search(x, y, categories, nets):
    scorer = make_scorer(mean_squared_error, greater_is_better=False, )

    for i, net in enumerate(nets):
        cb = TensorBoard(log_dir="./Graph_nn_{}".format(i), histogram_freq=0, write_graph=True, write_images=True)
        inner_cv = StratifiedKFold(categories)
        best_model = GridSearchCV(estimator=net, scoring=scorer, cv=inner_cv, fit_params={"callbacks": [cb]},
                                  param_grid={
                                      "learning_rate": [0.05, 0.01, 0.005, 0.001],
                                      "dropout_reg": [0.5, 0.3, 0.1],
                                      "epochs": [5, 10, 20, 50, 100],
                                      "batch_size": [32, 64, 256, 512]
                                  })
        best_model.fit(x, y)

        best_score = cross_val_score(X=x, y=y, estimator=net, scoring=scorer, groups=categories,
                                     fit_params={"callbacks": [cb]})
        yield best_model.best_params_, best_score

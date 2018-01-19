from typing import Callable

# scikit-learn imports
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error

# Keras imports
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Reshape
from keras.activations import relu, linear
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import TensorBoard, ModelCheckpoint

# import models
from keras_implementation.neural_net import dense_model
from keras_implementation.rnn import rnn_model
from keras_implementation.cnn import cnn_model, conv_operation


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
dense1 = dense_model([40, 40, 1], [relu, relu, linear], dropout_reg=0.3, input_shape=(180,), name="dense1")
dense2 = dense_model([80, 80, 1], [relu, relu, linear], dropout_reg=0.3, input_shape=(180,), name="dense2")
dense3 = dense_model([120, 80, 1], [relu, relu, linear], dropout_reg=0.3, input_shape=(180,), name="dense3")
dense4 = dense_model([80, 80, 80, 1], [relu, relu, linear], dropout_reg=0.3, input_shape=(180,), name="dense4")

# cnn networks
reshape_model = Sequential(layers=[Reshape(input_shape=(180,), target_shape=(180, 1))])

cnn1 = cnn_model(conv_layout=conv_operation(Conv1D, 32, 4, 1, "valid", relu, MaxPooling1D, 2),
                 pre_model=reshape_model, dense_nb_neurons=[40, 1], dense_activations=[relu, linear],
                 name="cnn1")
cnn2 = cnn_model(conv_layout=conv_operation(Conv1D, 64, 2, 1, "valid", relu, MaxPooling1D, 2),
                 pre_model=reshape_model, dense_nb_neurons=[120, 1], dense_activations=[relu, linear],
                 name="cnn2")
cnn3 = cnn_model(conv_layout=conv_operation(Conv1D, 64, 4, 1, "valid", relu, MaxPooling1D, 4),
                 pre_model=reshape_model, dense_nb_neurons=[120, 1], dense_activations=[relu, linear],
                 name="cnn3")

# rnn models
rnn1 = rnn_model(nb_rnn_neurons=[32, 32], dense_nb_neurons=[40, 1], dense_activations=[relu, linear], name="rnn1")
rnn2 = rnn_model(nb_rnn_neurons=[64, 64], dense_nb_neurons=[40, 1], dense_activations=[relu, linear], name="rnn2")
rnn3 = rnn_model(nb_rnn_neurons=[128, 64], dense_nb_neurons=[40, 1], dense_activations=[relu, linear], name="rnn3")

nns = [dense1, dense2, dense3, dense4,
       cnn1, cnn2, cnn3,
       rnn1, rnn2, rnn3]


def grid_search(x_train, y_train, x_val, y_val, categories, nets):
    scorer = make_scorer(mean_squared_error, greater_is_better=False, )

    for net in nets:
        # tensorboard callback
        cb_tb = TensorBoard(log_dir="./Graph_nn_{}".format(net.name),
                            histogram_freq=0, write_graph=True, write_images=True)

        # add validation measure each 5 epochs
        checkpoint = ModelCheckpoint(filepath=".models/{}.net".format(net.name), period=5)
        inner_cv = StratifiedKFold(categories)
        best_model = GridSearchCV(estimator=net, scoring=scorer, cv=inner_cv,
                                  fit_params={"callbacks": [cb_tb]},
                                  param_grid={
                                      "learning_rate": [0.05, 0.01, 0.005, 0.001],
                                      "dropout_reg": [0.5, 0.3, 0.1],
                                      "epochs": [5, 10, 20, 50, 100],
                                      "batch_size": [32, 64, 256, 512]
                                  })
        best_model.fit(x_train, y_train)

        best_score = cross_val_score(X=x_train, y=y_train, estimator=net, scoring=scorer, groups=categories,
                                     fit_params={"callbacks": [cb_tb, checkpoint],
                                                 "validation_data": (x_val, y_val)})
        yield best_model.best_params_, best_score

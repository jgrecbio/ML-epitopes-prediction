import pandas as pd
from typing import Callable, List

# scikit-learn imports
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split, KFold
from sklearn.metrics import make_scorer, mean_squared_error

# Keras imports
from keras.layers import Conv1D, MaxPooling1D
from keras.activations import relu, linear
from keras.wrappers.scikit_learn import KerasRegressor

# import models
from neural_net import dense_model, embed_pre_net, reshape_pre_net
from rnn import rnn_model
from cnn import cnn_model, conv_operation

# data import
from working_data import load_data, get_one_hot_data, get_tokenized_data, meas_discretize


def to_regressor(model_fn: Callable, kwargs) -> KerasRegressor:
    """
    Given a function that creates a keras model, it yield a wrapped
    scikit-learn compatible model
    :param model_fn: function the creates Sequential keras model
    :param kwargs: parameters of the model
    :return: a wrapped scikit-learn compatible model
    """
    return KerasRegressor(model_fn, **kwargs)


# fully connected networks
d1 = (dense_model, {"nb_units": [40, 40, 1], "activations": [relu, relu, linear],
                    "input_shape": (180, ), "name": "dense1"})

d2 = (dense_model, {"nb_units": [80, 80, 1], "activations": [relu, relu, linear],
                    "input_shape": (180, ), "name": "dense2"})

d3 = (dense_model, {"nb_units": [120, 80, 1], "activations": [relu, relu, linear],
                    "input_shape": (180, ), "name": "dense3"})

d4 = (dense_model, {"nb_units": [80, 80, 80, 1], "activations": [relu, relu, relu, linear],
                    "input_shape": (180, ), "name": "dense4"})

# fully connected networks with embed layer
embed = embed_pre_net()
d_em1 = (dense_model, {"nb_units": [40, 40, 1], "activations": [relu, relu, linear],
                       "pre_model": embed, "name": "dense_embed1"})
d_em2 = (dense_model, {"nb_units": [80, 80, 1], "activations": [relu, relu, linear],
                       "pre_model": embed, "name": "dense_embed2"})
d_em3 = (dense_model, {"nb_units": [120, 80, 1], "activations": [relu, relu, linear],
                       "pre_model": embed, "name": "dense_embed3"})
d_em4 = (dense_model, {"nb_units": [80, 80, 1], "activations": [relu, relu, linear],
                       "pre_model": embed, "name": "dense_embed4"})

# cnn networks
c1 = (cnn_model, {"conv_layout": conv_operation(Conv1D, 32, 4, 1, "valid", relu, MaxPooling1D, 2),
                  "pre_model": reshape_pre_net(), "dense_nb_neurons": [40, 1],
                  "dense_activations": [relu, linear], "name": "cnn1"})
c2 = (cnn_model, {"conv_layout": conv_operation(Conv1D, 64, 2, 1, "valid", relu, MaxPooling1D, 2),
                  "pre_model": reshape_pre_net(), "dense_nb_neurons": [120, 1],
                  "dense_activations": [relu, linear], "name": "cnn2"})
c3 = (cnn_model, {"conv_layout": conv_operation(Conv1D, 64, 4, 1, "valid", relu, MaxPooling1D, 4),
                  "pre_model": reshape_pre_net(), "dense_nb_neurons": [120, 1],
                  "dense_activations": [relu, linear], "name": "cnn3"})
c4 = (cnn_model, {"conv_layout": [conv_operation(Conv1D, 64, 4, 1, "valid", relu, MaxPooling1D, 4)]*2,
                  "pre_model": reshape_pre_net(), "dense_nb_neurons": [120, 1],
                  "dense_activations": [relu, linear], "name": "cnn3"})

# rnn models
r1 = (rnn_model, {"nb_rnn_neurons": [32, 32], "dense_nb_neurons": [40, 1],
                  "dense_activations": [relu, linear], "name": "rnn1"})
r2 = (rnn_model, {"nb_rnn_neurons": [64, 64], "dense_nb_neurons": [40, 1],
                  "dense_activations": [relu, linear], "name": "rnn2"})
r3 = (rnn_model, {"nb_rnn_neurons": [128, 64], "dense_nb_neurons": [40, 1],
                  "dense_activations": [relu, linear], "name": "rnn3"})

nns_oh = [d1, d2, d3, d4, c1, c2, c3, c4]
nns_em = [d_em1, d_em2, d_em3, d_em4, r1, r2, r3]


def grid_search(x_train, y_train, nets, categories,
                inner_cv, outer_cv,
                lrs: List[float] = (0.05, 0.01, 0.005, 0.001),
                drs: List[float] = (0.5, 0.3, 0.1),
                eps: List[int] = (5, 10, 20, 50, 100, 300),
                bts: List[int] = (32, 64, 256, 512, 1024),
                summary: bool= False):
    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    for i, (net_fn, kw) in enumerate(nets):
        if summary:
            net_fn(**kw).summary()

        net = to_regressor(net_fn, kw)
        best_model = GridSearchCV(estimator=net, scoring=scorer, cv=inner_cv,
                                  param_grid={
                                      "learning_rate": lrs,
                                      "dropout_reg": drs,
                                      "epochs": eps,
                                      "batch_size": bts,
                                  })
        best_model.fit(x_train, y_train)

        best_score = cross_val_score(X=x_train, y=y_train, estimator=net,
                                     scoring=scorer, cv=outer_cv, groups=categories)
        yield best_model.best_params_, best_score


data = load_data()
_, x_oh, y = get_one_hot_data(data)
x_tok = get_tokenized_data(data)

x_oh_train, x_oh_val, x_tok_train, x_tok_val, y_train, y_val = train_test_split(x_oh, x_tok, y)
categories = meas_discretize(pd.Series(y.reshape(-1)))
inner_cv = KFold(n_splits=10, shuffle=True, random_state=123)
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=321)

results_oh = list(grid_search(x_oh_train, y_train, nets=nns_oh,
                              inner_cv=inner_cv, outer_cv=outer_cv, categories=categories))
results_em = list(grid_search(x_tok_train, y_train, nets=nns_em,
                              inner_cv=inner_cv, outer_cv=outer_cv, categories=categories))

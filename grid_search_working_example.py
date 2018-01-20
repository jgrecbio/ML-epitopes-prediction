import pandas as pd
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from neural_net import dense_model
from keras.activations import relu, linear
from working_data import get_one_hot_data, load_data, meas_discretize
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split


d1 = (dense_model, {"nb_units": [40, 40, 1], "activations": [relu, relu, linear], "dropout_reg": 0.3,
                    "input_shape": (180, ), "name": "dense1"})
reg = KerasRegressor(d1[0], **d1[1])

data = load_data()
print(data.shape)
_, x_oh, y = get_one_hot_data(data)
x_train, x_val, y_train, y_val, data_train, data_val = train_test_split(x_oh, y, data, train_size=0.9)

categories = meas_discretize(pd.Series(y.reshape(-1)))
inner_cv = KFold(n_splits=10, shuffle=True, random_state=123)
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=321)

a = GridSearchCV(reg, {"learning_rate": [0.01, 0.005]},
                 fit_params={"epochs": 10}, n_jobs=2)
a.fit(x_train, y_train)
b = a.predict(x_val)

c = pd.Series(np.exp(b), name="predicted", index=data_val.index)
pd.concat([data_val, c], axis=1).to_csv("test.tsv", sep='\t', index=False)

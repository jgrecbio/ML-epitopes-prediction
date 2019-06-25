import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def cross_validate(model, x, y, nsplit: int = 10, state: int = 42, **fit_params):
    mses = []
    model.save_weights(".model.h5")
    cv = KFold(n_splits=nsplit, shuffle=True, random_state=state)
    for train_index, test_index in cv.split(x):
        model.load_weights(".model.h5")
        model.fit(x[train_index], y[train_index], **fit_params)
        y_pred = model.predict(x[test_index])
        mses.append(mean_squared_error(y[test_index], y_pred))

    return mses

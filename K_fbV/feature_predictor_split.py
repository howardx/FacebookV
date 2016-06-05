# a helper function that splits feature and predictor in training set
import pandas as pd

def feature_predictor_split(data, predictor_col):
    if isinstance(data, pd.DataFrame):
        data = data.as_matrix() # convert to numpy ndarray
    train_X = data[ :, 0:(predictor_col-1) ]
    train_Y = data[ :, predictor_col ]
    return (train_X, train_Y)

import pandas as pd
import numpy as np
import scipy

def factorize_predictor(data, predictor_idx):
    if isinstance(data, pd.DataFrame):
        colname = list(data.columns.values)
        data = data.as_matrix()

    sz = data.shape

    original_class = data[:, predictor_idx] # original class labels before factorization
    data = scipy.delete(data, predictor_idx, 1)  # delete the original class label column - cannot be used by xgboost

    factorized_class, unique_class = pd.factorize(original_class)
    overall_num_of_classes = len(unique_class)

    # reshape so dimension matches for appending to original dataset
    fclass_np_horizontal = np.reshape(factorized_class, (sz[0], 1))

    # add factorized labels as the last column of original matrix
    data = np.hstack((data, fclass_np_horizontal))

    data = pd.DataFrame(data) # convert back to pandas dataframe
    if colname is not None:
        data.columns = colname

    return (data, unique_class)

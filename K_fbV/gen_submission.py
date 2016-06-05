import pandas as pd

def merge_prediction_grids(predict_dict, sort_rowID = False, save_file = False, output_path = None):
    df_finalResult = pd.concat(predict_dict.values(), ignore_index = True)

    if sort_rowID:
        df_finalResult = df_finalResult.sort(['row_id']) # for testing purposes ONLY, not necessary

    if save_file:
        df_finalResult.to_csv(output_path, index = False) # no pandas index in CSV file
    return df_finalResult

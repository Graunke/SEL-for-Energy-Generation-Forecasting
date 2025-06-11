import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from dataHandle import datasetaq
from sklearn.model_selection import KFold

# ------------------------- Settings ------------------------ #

#Sizes
TRAIN_SZ = 0.7
VALID_SZ = 0.15
TEST_SZ = 0.15
FOLD_SZ = 6
LAGS_SZ = 8
H_FORECAST = 1

# ---------------------- Normalization ---------------------- #

#dataframe creation
def train_test_data():
    df = datasetaq()

    #selectiong working features
    df = df[['index', 'month', 'weekday', 'val_geracao', 'val_cargaenergiamwmed']]

    #scaler definition
    scaler = StandardScaler()

    #applying Z-score normalization to the selected features of the df
    df['val_geracao_norm'] = scaler.fit_transform(df[['val_geracao']])
    df['val_cargaenergiamwmed_norm'] = scaler.fit_transform(df[['val_cargaenergiamwmed']])
    df['month_norm'] = scaler.fit_transform(df[['month']])
    df['weekday_norm'] = scaler.fit_transform(df[['weekday']])

    #Ploting the normalized series
    df['val_geracao_norm'].plot()
    df['val_cargaenergiamwmed_norm'].plot()
    df['month_norm'].plot()
    # df['weekday_norm'].plot() #this added to the graph polutes it 
    # plt.show()

    # --------------------- Dataset Division ---------------------- # 

    #creating sequences of data based on PACF analysis for forecasting horizon = 1
    def create_sequences(data, pred, seq_len, n_steps_out):
        X, y = [], []
        data_np = data.to_numpy()
        pred_np = pred.to_numpy()
        n_samples = len(data_np)
        for i in range(n_samples - seq_len - n_steps_out + 1):
            sequence = data_np[i : i + seq_len]
            target = pred_np[i + seq_len : i + seq_len + n_steps_out]
            X.append(sequence)
            y.append(target)
        return X,y

    #predictors = month_num, weekday_num, consumption and generation lagged
    predictors = df[['month_norm','weekday_norm','val_geracao_norm','val_cargaenergiamwmed_norm']]

    #predicted value is the generation value current, not lagged
    predicted  = predictors[['val_geracao_norm']]

    #creating the sequences of values that are going to be used to predict and the corresponding predicted value
    lag_val, pred_val = create_sequences(predictors, predicted,LAGS_SZ,H_FORECAST)
    X = np.array(lag_val)
    y = np.array(pred_val)

    #folding the dataset
    kf = KFold(n_splits = FOLD_SZ, shuffle = False)

    #list of folds
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    # Dataset division iteration
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train.append(X[train_index]), X_test.append(X[test_index])
        y_train.append(y[train_index]), y_test.append(y[test_index])


    return X_train, X_test, y_train, y_test
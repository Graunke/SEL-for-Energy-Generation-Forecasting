import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from xgboost import xgb

# -------------------- Building Models --------------------- #

#Buildintg the LSTM architecture
def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    return model
    
#Buildintg the GRU architecture
def build_gru(input_shape):
    model = Sequential()
    model.add(GRU(64, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    return model

#Buildintg the XGBoost architecture
def build_xgboost(input_shape):
    model = xgb.XGBRegressor()
    print(model.summary())
    return model

#Building a CNN architecture
def build_cnn(input_shape):
    model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    return model
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# -------------------- Building Base Models --------------------- #

#Buildintg the LSTM architecture
def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError()])
    return model
    
#Buildintg the GRU architecture
def build_gru(input_shape):
    model = Sequential()
    model.add(GRU(64, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError()])
    return model

#Buildintg the XGBoost architecture
def build_xgboost():
    model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.001,
    max_depth=6,
    random_state=42,
    )
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
    model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError()])
    return model

# ------------------------ Meta Learner ----------------------- #

def build_rf():
    model = RFE(estimator=RandomForestRegressor(n_estimators=1000, random_state=1),n_features_to_select=4)
    return model
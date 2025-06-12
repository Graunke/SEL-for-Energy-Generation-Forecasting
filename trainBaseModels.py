from preprocessing import train_test_data
from stackingEnsembleArq import build_cnn, build_gru, build_lstm, build_xgboost
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

# ----------------------- Dataset Division ---------------------- # 

X_train, X_test, y_train, y_test = train_test_data()

#list of models
xgb_trained  = []
cnn_trained  = []
gru_trained  = []
lstm_trained = []

# Prediction that are composing the meta learner training set
xgb_preds  = []
cnn_preds  = []
gru_preds  = []
lstm_preds = []

# --------------------- Base Model Training --------------------- #

def train_base_models():
    # CV loop for training
    for i in range(6):
        print(f'fold X {i+1} de treino: {X_train[i].shape} e fold {i+1} X de teste: {X_test[i].shape}')
        print(f'fold y {i+1} de treino: {y_train[i].shape} e fold {i+1} y de teste: {y_test[i].shape}')

        #definindo parametros para a parada antecipada
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        #Best model saving funtion 
        checkpoint = ModelCheckpoint(f'models/lstm_fold_{i+1}.h5', monitor='val_loss', save_best_only=True, verbose = 1)

        #Training the lstm net
        print(f' -------------------------------------- Treinando o fold {i+1} da rede LSTM -------------------------------------- ')
        lstm = build_lstm( X_train[i].shape[1:])
        lstm.fit(X_train[i], y_train[i], validation_data=(X_test[i], y_test[i]), epochs=100, callbacks=[early_stopping, checkpoint])

        #Training the gru net
        print(f' -------------------------------------- Treinando o fold {i+1} da rede GRU -------------------------------------- ')
        gru  = build_gru(  X_train[i].shape[1:])
        checkpoint_gru = ModelCheckpoint(f'models/gru_fold_{i+1}.h5', monitor='val_loss', save_best_only=True, verbose = 1)
        gru.fit(X_train[i], y_train[i], validation_data=(X_test[i], y_test[i]), epochs=100, callbacks=[early_stopping, checkpoint_gru])

        #Training the cnn net
        print(f' -------------------------------------- Treinando o fold {i+1} da rede CNN -------------------------------------- ')
        cnn  = build_cnn(  X_train[i].shape[1:])
        checkpoint_cnn = ModelCheckpoint(f'models/cnn_fold_{i+1}.h5', monitor='val_loss', save_best_only=True, verbose = 1)
        cnn.fit(X_train[i], y_train[i], validation_data=(X_test[i], y_test[i]), epochs=100, callbacks=[early_stopping, checkpoint_cnn])

        #Training the xgboost net
        print(f' -------------------------------------- Treinando o fold {i+1} da rede XGB --------------------------------------')
        xgb = build_xgboost()
        xgb.fit(
            X_train[i].reshape(X_train[i].shape[0], -1), y_train[i].squeeze(),
            eval_set=[(X_test[i].reshape(X_test[i].shape[0], -1), y_test[i].squeeze())],
            verbose=True
        )

        #Saving the models
        lstm_trained.append(lstm)
        gru_trained.append(gru)
        cnn_trained.append(cnn)
        xgb_trained.append(xgb)

        #Appending the predictions to the lists
        lstm_preds.append(lstm.predict(X_test[i]))
        gru_preds.append(gru.predict(X_test[i]))
        cnn_preds.append(cnn.predict(X_test[i]))
        xgb_preds.append(xgb.predict(X_test[i].reshape(X_test[i].shape[0], -1)))

    # Concatenate predictions and actual values from all folds
    y_test_all = np.concatenate(y_test).squeeze()
    lstm_all   = np.concatenate(lstm_preds).squeeze()
    gru_all    = np.concatenate(gru_preds).squeeze()
    cnn_all    = np.concatenate(cnn_preds).squeeze()
    xgb_all    = np.concatenate(xgb_preds).squeeze()

    # Plot all predictions vs actual
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_all, label='Actual', color='black', linewidth=2)
    plt.plot(lstm_all, label='LSTM Prediction', alpha=0.8)
    plt.plot(gru_all, label='GRU Prediction', alpha=0.8)
    plt.plot(cnn_all, label='CNN Prediction', alpha=0.8)
    plt.plot(xgb_all, label='XGBoost Prediction', alpha=0.8)

    plt.title('Model Predictions vs Actual Values (All Folds Combined)')
    plt.xlabel('Time Step')
    plt.ylabel('Target Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    #return the new train data for the meta learner
    return lstm_preds, gru_preds, cnn_preds, xgb_preds








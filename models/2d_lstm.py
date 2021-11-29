import sys
sys.path.insert(0, 'E:\Documents\FRI\predmeti\letnik_01\zimni semester\Strojno učenje\seminarska naloga')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import L1L2
from keras.initializers import GlorotUniform, HeUniform

def lstm_build(n_units=100, n_layers=(4,1), lreg=(0.000001, 0.000001)):
    """
    Train LSTM model on stock time series.

    :param n_units: Number of Neurons per Layer
    :param value: int
    :param n_layers: Number of LSTM & ANN Layers
    :param value: tuple (int, int)
    :param lreg: L1 & L2 Regularization Coeficients
    :param value: (float, float)
    :return model: LSTM Model.
    """
    err_reg = L1L2(lreg[0], lreg[1])

    lstm_initializer = GlorotUniform(seed=17)
    relu_initializer = HeUniform(seed=17)

    model = Sequential()
    model.add(LSTM(units=n_units, return_sequences=True, kernel_regularizer=err_reg,
                                                    input_shape=(feature_set.shape[1], feature_set.shape[2]),
                                                    kernel_initializer=lstm_initializer))
    model.add(Dropout(0.2))
    for i in range(n_layers[0]-2):
        model.add(LSTM(units=n_units, return_sequences=True, kernel_regularizer=err_reg,
                                                        kernel_initializer=lstm_initializer))
        model.add(Dropout(0.2))
    model.add(LSTM(units=n_units, kernel_regularizer=err_reg,
                                                    kernel_initializer=lstm_initializer))
    model.add(Dropout(0.2))

    for i in range(n_layers[1]):
        model.add(Dense(units=n_units, kernel_regularizer=err_reg,
        kernel_initializer=relu_initializer))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    model.compile(optimizer = 'adam', loss = 'mse')

    return model

if __name__ == "__main__":
    import visualization.view_time_series as vts
    import os

    WINDOW_SIZE = 60

    RES_FILE = "../results.csv"

    if os.path.isfile(RES_FILE):
        res = pd.read_csv(RES_FILE)
    else:
        res = pd.DataFrame(columns=["ALGORITHM", "TYPE", "COMPANY", "ERROR"])

    ind = 0
    for stk in ["AAPL", "AMZN", "FB", "GOOG", "MSFT"]:
        aapl = vts.stock_get(stk, "2013-01-01", "2018-01-01")
        aapl_test = vts.stock_get(stk, "2018-01-01", "2019-01-01")

        apple_train = aapl.iloc[:,1:7].values
        num_instances, num_features = aapl.shape
        apple_test = aapl_test.iloc[:, 1:7].values

        feature_set = []
        labels = []
        test_feature_set = []
        test_labels = []

        #scaler = StandardScaler()
        scaler = MinMaxScaler(feature_range=(-1,1))

        apple_train_scaled = scaler.fit_transform(apple_train)
        print(apple_train_scaled)
        apple_test_scaled = scaler.transform(apple_test)

        for i in range(WINDOW_SIZE, apple_train_scaled.shape[0]):
            feature_set.append(apple_train_scaled[i-WINDOW_SIZE:i, :])
            labels.append(apple_train_scaled[i,0])
        for i in range(WINDOW_SIZE, apple_test_scaled.shape[0]):
            test_feature_set.append(apple_test_scaled[i-WINDOW_SIZE:i, :])
            test_labels.append(apple_test_scaled[i,0])

        feature_set, labels = np.array(feature_set), np.array(labels)
        feature_set = np.reshape(feature_set, (feature_set.shape[0], feature_set.shape[1], feature_set.shape[2]))
        print(feature_set.shape)

        test_feature_set, test_labels = np.array(test_feature_set), np.array(test_labels)
        test_feature_set = np.reshape(test_feature_set, (test_feature_set.shape[0], \
                                                        test_feature_set.shape[1], feature_set.shape[2]))

        ### Build model
        model = lstm_build()

        ### Train Model
        stopper = EarlyStopping(monitor='val_loss', patience = 15)
        checkpoint_filepath = stk + '_checkpoint/'
        cacher = ModelCheckpoint(filepath=checkpoint_filepath,
                                save_weights_only=True,
                                monitor='val_loss',
                                mode='min',
                                save_freq='epoch',
                                save_best_only=True)

        history = model.fit(feature_set, labels, epochs = 200, batch_size = 64, \
                               validation_split=0.1,
                               verbose=1, callbacks=[stopper, cacher])
        #print(history.history)
        model.load_weights(checkpoint_filepath)
        model.save(stk + "_model.h5")

        y_hat = model.predict(test_feature_set)

        # Change Scale
        rev_scaler = MinMaxScaler()
        rev_scaler.min_, rev_scaler.scale_ = scaler.min_[0], scaler.scale_[0]

        r_test = rev_scaler.inverse_transform(test_labels.reshape(-1, 1))
        r_y = rev_scaler.inverse_transform(y_hat.reshape(-1, 1))

        print(f"LSTM-RMSE: {mse(r_test, r_y, squared=False)}")
        res = res.append({"ALGORITHM": "LSTM",
                    "TYPE": "RMSE",
                    "COMPANY": stk,
                    "ERROR": mse(r_test, r_y, squared=False)},
                     ignore_index=True)
        print(f"LSTM-MAPE: {mape(r_test, r_y)}")
        res = res.append({"ALGORITHM": "LSTM",
                    "TYPE": "MAPE",
                    "COMPANY": stk,
                    "ERROR": mape(r_test, r_y)},
                    ignore_index=True)

        ### Visualize Results
        ax = plt.subplot2grid((2,3), (int(ind/3),ind%3))
        rdf = pd.DataFrame()
        rdf["Open"] = r_y.flatten()
        rdf["Date"] = np.array(aapl_test["Date"][WINDOW_SIZE:])
        tdf = pd.DataFrame()
        tdf["Open"] = r_test.flatten()
        tdf["Date"] = np.array(aapl_test["Date"][WINDOW_SIZE:])
        vts.stock_view(rdf, wtitle=stk, axes=ax)
        vts.stock_view(tdf, wtitle=stk, axes=ax)
        #vts.stock_view(aapl)
        #plt.legend()
        ind += 1
    plt.show()

    res.to_csv(RES_FILE, index=False)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import L1L2
from keras.initializers import GlorotUniform, HeUniform

AAPL = pd.read_csv('AAPL.csv')
AAPL_test = pd.read_csv('AAPL_test.csv')

#print(AAPL)

#AAPL['Open'].plot(x=AAPL['Date'], title="Open")
#plt.show()

apple_train = AAPL.iloc[:, 1:2].values
#print(apple_train)
apple_test = AAPL_test.iloc[:, 1:2].values

scaler = MinMaxScaler(feature_range=(-1,1))

apple_train_scaled = scaler.fit_transform(apple_train)
apple_test_scaled = scaler.transform(apple_test)
#print(apple_train_scaled)
fig, ax = plt.subplots()
#ax.title("Open")
pd.DataFrame(apple_train_scaled).plot(ax=ax)
pd.DataFrame(apple_test_scaled, index=np.arange(1260,1260+apple_test_scaled.shape[0])).plot(ax=ax)
plt.xticks(ticks=np.arange(0, 1260, step=365), labels=AAPL["Date"].iloc[np.arange(0, 1260, step=365)])
#plt.show()

feature_set = []
labels = []
test_feature_set = []
test_labels = []

for i in range(60, 1260):
    feature_set.append(apple_train_scaled[i-60:i, 0])
    labels.append(apple_train_scaled[i,0])
for i in range(60, apple_test_scaled.shape[0]):
    test_feature_set.append(apple_test_scaled[i-60:i, 0])
    test_labels.append(apple_test_scaled[i,0])

#print(feature_set)

feature_set, labels = np.array(feature_set), np.array(labels)
feature_set = np.reshape(feature_set, (feature_set.shape[0], feature_set.shape[1], 1))

test_feature_set, test_labels = np.array(test_feature_set), np.array(test_labels)
test_feature_set = np.reshape(test_feature_set, (test_feature_set.shape[0], test_feature_set.shape[1], 1))

lstm_initializer = GlorotUniform(seed=17)
relu_initializer = HeUniform(seed=17)

model = Sequential()
model.add(LSTM(units=100, return_sequences=True, kernel_regularizer=L1L2(0.0000, 0.00001),
                                                input_shape=(feature_set.shape[1], 1),
                                                kernel_initializer=lstm_initializer))
model.add(Dropout(0.2))
model.add(LSTM(units=150, return_sequences=True, kernel_regularizer=L1L2(0.0000, 0.00001),
                                                kernel_initializer=lstm_initializer))
model.add(Dropout(0.2))
model.add(LSTM(units=150, return_sequences=True, kernel_regularizer=L1L2(0.0000, 0.00001),
                                                kernel_initializer=lstm_initializer))
model.add(Dropout(0.2))
model.add(LSTM(units=100, kernel_regularizer=L1L2(0.0000, 0.00001),
                                                kernel_initializer=lstm_initializer))
model.add(Dropout(0.2))
#model.add(BatchNormalization())
model.add(Dense(units=100, kernel_regularizer=L1L2(0.0000, 0.00001),
kernel_initializer=relu_initializer))
model.add(Dropout(0.2))
model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

stopper = EarlyStopping(monitor='loss', patience = 20)
checkpoint_filepath = 'checkpoints/'
cacher = ModelCheckpoint(filepath=checkpoint_filepath,
                        save_weights_only=True,
                        validation_split=0.2,
                        monitor='loss',
                        mode='min',
                        save_best_only=True)

model.fit(feature_set, labels, epochs = 300, batch_size = 64, \
                       verbose=1, callbacks=[stopper, cacher])
model.load_weights(checkpoint_filepath)
model.save("model.h5")

y_hat = model.predict(test_feature_set)
print(mse(test_labels, y_hat))
fig, ax = plt.subplots()
ax.plot(test_labels)
ax.plot(y_hat)
plt.show()

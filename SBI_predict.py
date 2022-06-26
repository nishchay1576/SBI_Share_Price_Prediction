import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

dataset  = yf.download('SBIN.NS','2012-01-03','2020-06-15')
trainset = dataset.iloc[:,2:3].values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
trainset = scaler.fit_transform(trainset)

X_train = []
y_train = []

for i in range(60,2074):
    
    X_train.append(trainset[i-60:i,0])
    y_train.append(trainset[i,0])
    
X_train,y_train = np.array(X_train),np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_train,y_train, test_size = 0.25, random_state = 0)

X_train_m = X_train[:1500]
X_test_m = X_train[1500:]
y_train_m = y_train[:1500]
y_test_m = y_train[1500:]
# Part 2 - Building the RNN


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 70, return_sequences = True, input_shape = (X_train_m.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 70  , return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 70, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 70))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train_m, y_train_m, epochs = 100, batch_size = 32)



predicted = regressor.predict(X_test_m)
predicted_price = scaler.inverse_transform(predicted)

# Visualising the results
plt.plot(scaler.inverse_transform(y_test_m.reshape(-1,1)), color = 'red', label = 'Real SBI share price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted SBI Stock Price')
plt.title('SBI Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('SBI Stock Price')
plt.legend()
plt.show()


#179 open #171 min

    
todayset = dataset.iloc[:,2:3].values
df = pd.DataFrame({})

todayset = np.append(todayset,[172])

todayset = np.append(todayset,[169.4])
today = todayset[todayset.shape[0] - 60:]

today =  scaler.transform(today.reshape(-1,1))
X = []
for i in range(len(today)):
    
    X.append(today[i,0])
    
X = np.array(X)
X = X.reshape(60,1)
X = np.reshape(X, (X.shape[1], X.shape[0], 1))



predicted_today = regressor.predict(X)
predicted_price_today = scaler.inverse_transform(predicted_today)
    
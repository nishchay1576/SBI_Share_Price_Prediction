# Stock Price Prediction of State Bank Of India  Using  LSTM Recurrent Neural Network


# Dataset:  
The dataset is taken from yahoo finace's website in CSV format. The dataset consists of Open, High, Low and Closing Prices of sbi stocks from 3rd january 2012 to 15th June 2020 - total 2074 rows.

# Price Indicator:  
Stock traders mainly use three indicators for prediction: OHLC average (average of Open, High, Low and Closing Prices), HLC average (average of High, Low and Closing Prices) and Closing price, In this project, Closing price  has been used.

# Data Pre-processing:<br />
After Fetching closing data, it becomes one column data. This has been converted into  time series data

# Model:<br />
One sequential LSTM layers  and one dense layer is used to build the RNN model using Keras deep learning library. Since this is a regression task, 'linear' activation has been used in final layer.

# Version:<br />
Python 3.5 and latest versions of all libraries including deep learning library Keras and Tensorflow.

# Training:<br />
75% data is used for training. Adagrad (adaptive gradient algorithm) optimizer is used for faster convergence. After training starts it will look like:


# Test:<br />
Test accuracy metric is root mean square error (RMSE).

# Results:<br />

![Image of Prediction](https://github.com/Meet0067/SBI-share-price-prediction-using-LSTM/blob/master/sbi.PNG)


Finally, this work can greatly help the quantitative traders to take decisions.

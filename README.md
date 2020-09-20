# Deep-Learning
LSTM Stock Predictor

![deep-learning.jpg](Images/deep-learning.jpg)

Due to the volatility of cryptocurrency speculation, investors will often try to incorporate sentiment from social media and news articles to help guide their trading strategies. One such indicator is the [Crypto Fear and Greed Index (FNG)](https://alternative.me/crypto/fear-and-greed-index/) which attempts to use a variety of data sources to produce a daily FNG value for cryptocurrency. This project tries to help build and evaluate deep learning models using both the FNG values and simple closing prices to determine if the FNG indicator provides a better signal for cryptocurrencies than the normal closing price data.

This project uses deep learning recurrent neural networks to model bitcoin closing prices. One model will use the FNG indicators to predict the closing price while the second model will use a window of closing prices to predict the nth closing price.

The following steps were covered:

1. [Prepare the data for training and testing](#prepare-the-data-for-training-and-testing)
2. [Build and train custom LSTM RNNs](#build-and-train-custom-lstm-rnns)
3. [Evaluate the performance of each model](#evaluate-the-performance-of-each-model)

### Files

[Closing Prices Starter Notebook](Starter_Code/lstm_stock_predictor_closing.ipynb)

[FNG Starter Notebook](Starter_Code/lstm_stock_predictor_fng.ipynb)

- - -

### Prepare the data for training and testing


For the Fear and Greed model,  the FNG values were used to try and predict the closing price. 

For the closing price model, the previous closing prices were used to try and predict the next closing price. 

Each model used 70% of the data for training and 30% of the data for testing.

The MinMaxScaler was applied to the X and y values to scale the data for the model.

Finally, the X_train and X_test values were reshaped to fit the model's requirement of samples, time steps, and features. (*example:* `X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))`)

### Build and train custom LSTM RNNs

In each Jupyter Notebook, the same custom LSTM RNN architecture was created. In one notebook, the data was fitted using the FNG values. In the second notebook, the data was fitted  using only closing prices.

The same parameters and training steps were used for each model. This is necessary to compare each model accurately.

### Evaluate the performance of each model

Finally, use the testing data to evaluate each model and the performance was compared.

Using the above the following observations were made:

> Which model has a lower loss? The LSTM model using the closing prices has lower loss.
>
> Which model tracks the actual values better over time?
>
> Which window size works best for the model?

- - -
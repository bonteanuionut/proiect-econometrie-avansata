import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
#from tqdm import tqdm_notebook
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

def split_data(dataframe: pd.DataFrame):
    '''
    DOCSTRING

    This function splits the data into training data and test data.

    dataframe -> the dataset you're using

    Returns tuple of DataFrames.
    '''
    nrows = int(round(len(dataframe) * 0.75, 0))
    dataframe_train = dataframe.sort_index().iloc[:nrows, :]
    dataframe_test = dataframe.sort_index().iloc[nrows:, :]
    assert len(dataframe_train) + len(dataframe_test) == len(dataframe), 'Not the same number of rows!'
    return (dataframe_train, dataframe_test)

def plot_pacf_acf(dataframe: pd.DataFrame, y: str):
    """
    DOCSTRING

    This function plots the ACF(Autocorrelation Function) and PACF(Partial Autocorrelation Function)

    dataframe -> the dataset you're using
    y -> the dependent variable
    """

    plot_pacf(dataframe[y])
    plot_acf(dataframe[y])

def adfuller_and_diff(dataframe: pd.DataFrame, y: str):
    """
    DOCSTRING

    This function checks for stationarity and if not, it will differentiate our Y until it's stationary.

    dataframe -> the dataset you're using
    y -> the dependent variable
    """
    result = adfuller(dataframe[y])
    i = 1
    if result[1] > 0.05:
        print('Not stationary!')
        while True:
            dataframe[f'{y}_diff_{i}'] = dataframe[y].diff(periods=i)
            dataframe.dropna(inplace=True)
            result = adfuller(dataframe[f'{y}_diff_{i}'])
            if result[1] > 0.05:
                print('Still not stationary!')
                i += 1
                continue
            else:
                print('Stationary!')
                print(f'ADF: {result[0]}')
                print(f'p-value: {result[1]}')
                break
    print(f'Diff: {i}')

def train_predict_plot(df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, y: str, order: tuple):
    """
    DOCSTRING

    This function computes the predictions and will plot the actual values, the train values and the test values (out-of sample).
    df -> the dataset you're using
    train_df -> the train dataset
    test_df -> the test dataaset
    y -> the dependent variable
    order -> a parameter that's passed to ARIMA model (p, d, q)

    This function will also print the RMSE.

    """
    model = ARIMA(train_df[y], order = order)
    model_fit = model.fit()
    label = f"ARIMA({order[0]},{order[1]},{order[2]})"
    length_test = len(test_df)
    train = model_fit.fittedvalues.rename(label+'_train')
    test = model_fit.forecast(length_test).rename(label+'_test')
    test.index = test_df.index
    df = df.merge(right=train,
                  how='left',
                  left_index=True,
                  right_index=True)
    df = df.merge(right=test,
                  how='left',
                  left_index=True,
                  right_index=True)
    plt.plot(df.index, df[y], label='Actual')
    plt.plot(df.index, df[label+'_train'], label='Train', alpha=0.3)
    plt.plot(df.index, df[label+'_test'], label='Test', alpha=0.3)
    plt.legend()
    plt.show()
    rmse_train = sqrt(mean_squared_error(df.iloc[:len(train_df)][y], df.iloc[:len(train_df)][label+'_train']))
    rmse_test = sqrt(mean_squared_error(df.iloc[len(train_df):][y], df.iloc[len(train_df):][label+'_test']))
    print(f'Root mean squared error (RMSE) training: {rmse_train}')
    print(f'Root mean squared error (RMSE) training: {rmse_test}')
import sys
sys.path.insert(0, 'E:\Documents\FRI\predmeti\letnik_01\zimni semester\Strojno učenje\seminarska naloga')

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse

def arima_build(df, aargs):
    """
    Remove trend from stock time series with ARIMA.

    :param df: Stock dataframe
    :param value: pandas.core.data.DataFrame
    :param aargs: statsmodels.tsa.arima.model.ARIMA hyperparameters
    :value: tuple (int, int, int)
    :return amodels: fitted models
    :rtype: dict
    """

    amodels = {}

    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        model = ARIMA(np.array(df[col]), order=aargs)
        amodels[col] = model.fit()

    return amodels

def arima_smooth(df, amodels):
    """
    Smooth all columns of time series with already fitted ARIMA models.
    """
    rdf = pd.DataFrame()
    tdf = pd.DataFrame()

    rdf["Date"] = df["Date"]
    tdf["Date"] = df["Date"]

    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        history = df[col]
        tdf[col] = amodels[col].apply(np.array(df[col])).fittedvalues
        rdf[col] = df[col] - tdf[col]

    return tdf, rdf

def arima_forecast(df, aargs, window_size):
    """
    Predict time series with ARIMA model.

    :param df: Stock data
    :param value: pandas.core.data.DataFrame
    :param amodels: dict of ARIMA models
    :param window_size: model starts predicting after window_size steps
    :param value: int
    """
    tdf = pd.DataFrame()

    tdf["Date"] = np.array(df["Date"])

    for col in ["Open"]:
        res = df[col][:window_size].copy().values.tolist()
        #print(res)
        for lag in range(window_size, len(df[col])):
            model = ARIMA(np.array(df[col][:lag]), order=aargs)
            model_fit = model.fit()
            y = model_fit.forecast()
            #print(y[0])
            res.append(y[0])
        tdf[col] = pd.Series(res)
    #print(pd.Series(res))
    #print(len(res))
    #print(len(tdf["Date"]))
    return tdf


if __name__ == "__main__":
    import visualization.view_time_series as vts
    import os

    WINDOW_SIZE = 60
    RES_FILE = "../arima_results.csv"

    if os.path.isfile(RES_FILE):
        res = pd.read_csv(RES_FILE)
    else:
        res = pd.DataFrame(columns=["ALGORITHM", "START", "END","TYPE", "COMPANY", "ERROR"])

    ind = 0
    for stk in ["AAPL", "AMZN", "FB", "GOOG", "MSFT"]:
        for i in range(9,10):
            # Set dates for 10-fold cross validation
            start_date = "20"+ str(int(i/2)+10) +"-0"+ str(int(i%2)*5+1) +"-01"
            end_date = "20"+ str(int(i/2)+15) +"-0"+ str(int(i%2)*5+1) +"-01"
            start_date_test = end_date
            end_date_test = "20"+ str(int(i/2)+16) +"-0"+ str(int(i%2)*5+1) +"-01"

            aapl_test = vts.stock_get(stk, start_date_test, end_date_test)
            aapl_pred = arima_forecast(aapl_test, (1,1,1), WINDOW_SIZE)
            print(f"RMSE of ARIMA: {mse(np.array(aapl_test['Open']), np.array(aapl_pred['Open']), squared=False)}")
            res = res.append({"ALGORITHM": "ARIMA",
                        "TYPE": "RMSE",
                        "START": start_date,
                        "END": end_date,
                        "COMPANY": stk,
                        "ERROR": mse(np.array(aapl_test['Open']), np.array(aapl_pred['Open']), squared=False)},
                        ignore_index=True)
            print(f"MAPE of ARIMA: {mape(np.array(aapl_test['Open']), np.array(aapl_pred['Open']))}")
            res = res.append({"ALGORITHM": "ARIMA",
                        "TYPE": "MAPE",
                        "START": start_date,
                        "END": end_date,
                        "COMPANY": stk,
                        "ERROR": mape(np.array(aapl_test['Open']), np.array(aapl_pred['Open']))},
                        ignore_index=True)

            res.to_csv(RES_FILE, index=False)

            #ax = plt.subplot2grid((1,), (int(ind/3),ind%3))
            #aapl_full = vts.stock_get("AAPL", "2013-01-01", "2019-01-01")
            #vts.stock_view(aapl_full)
            ax = plt.subplot2grid((2,3), (int(ind/3),ind%3))

            vts.stock_view(aapl_test, wtitle=stk, axes=ax)
            #vts.stock_view(aapl_test.iloc[WINDOW_SIZE:], wtitle=stk, axes=ax)
            vts.stock_view(aapl_pred.tail(len(aapl_pred)-WINDOW_SIZE), wtitle=stk, axes=ax)
            #print(aapl_test_pred.head())
            #vts.stock_view(aapl_test_pred)
            #stock_view(stk, "2013-01-01", "2018-01-01", ax)
            ind += 1
    plt.show()

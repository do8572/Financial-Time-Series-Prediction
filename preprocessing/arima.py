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

def arima_forecast(df, aargs, window_size, cols=["Open"]):
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

    for col in cols:
        res = df[col][:window_size].copy().values.tolist()
        #print(res)
        for lag in range(window_size, len(df[col])):
            model = ARIMA(np.array(df[col][:lag]), order=aargs)
            model_fit = model.fit()
            y = model_fit.forecast()
            print(y[0])
            res.append(y[0])
        tdf[col] = pd.Series(res)
    #print(pd.Series(res))
    #print(len(res))
    #print(len(tdf["Date"]))
    return tdf

def arima_residuals(df, aargs, window_size):
    return arima_forecast(df, aargs, window_size, cols=["Open", "High", "Low", "Close", "Adj Close", "Volume"])


if __name__ == "__main__":
    import visualization.view_time_series as vts
    import os

    WINDOW_SIZE = 60
    RES_FILE = "../arima_results.csv"

    res = pd.DataFrame(columns=["ALGORITHM", "START", "END","TYPE", "COMPANY", "ERROR"])

    ind = 0
    for stk in ["AAPL", "AMZN", "FB", "GOOG", "MSFT"]:
        PRED_FILE = "./predictions/arima_prediction_"+ stk +".csv"
        predfile = pd.DataFrame(columns=["Date", "Real", "Pred"])
        for i in range(1,10):
            # Set dates for 10-fold cross validation
            start_date = "20"+ str(int(i/2)+10) +"-0"+ str(int(i%2)*5+1) +"-01"
            end_date = "20"+ str(int(i/2)+15) +"-0"+ str(int(i%2)*5+1) +"-01"
            start_date_test = end_date
            end_date_test = "20"+ str(int(i/2)+16) +"-0"+ str(int(i%2)*5+1) +"-01"

            print(f"Stock {stk}, train time frame: {start_date}-{end_date}")
            print(f"Stock {stk}, test time frame: {start_date_test}-{end_date_test}")

            stk_test = vts.stock_get(stk, start_date_test, end_date_test)
            tmp_test = stk_test.copy()

            stk_pred = arima_forecast(stk_test, (1,1,1), WINDOW_SIZE)

            print(stk_pred["Open"])

            tmp_test = tmp_test.reset_index()
            for i in range(len(stk_pred)):
                predfile = predfile.append({"Date": stk_pred.at[i, "Date"],
                                 "Real": np.array(stk_test['Open'])[i],
                                 "Pred": np.array(stk_pred['Open'])[i]}, ignore_index=True)

            print(f"RMSE of ARIMA: {mse(np.array(stk_test['Open']), np.array(stk_pred['Open']), squared=False)}")
            res = res.append({"ALGORITHM": "ARIMA",
                        "TYPE": "RMSE",
                        "START": start_date,
                        "END": end_date,
                        "COMPANY": stk,
                        "ERROR": mse(np.array(stk_test['Open']), np.array(stk_pred['Open']), squared=False)},
                        ignore_index=True)
            print(f"MAPE of ARIMA: {mape(np.array(stk_test['Open']), np.array(stk_pred['Open']))}")
            res = res.append({"ALGORITHM": "ARIMA",
                        "TYPE": "MAPE",
                        "START": start_date,
                        "END": end_date,
                        "COMPANY": stk,
                        "ERROR": mape(np.array(stk_test['Open']), np.array(stk_pred['Open']))},
                        ignore_index=True)

            res.to_csv(RES_FILE, index=False)

        ### Visualize Results
        ax = plt.subplot2grid((2,3), (int(ind/3),ind%3))
        rdf = pd.DataFrame()
        rdf["Open"] = predfile["Real"]
        rdf["Date"] = predfile["Date"]
        tdf = pd.DataFrame()
        tdf["Open"] = predfile["Pred"]
        tdf["Date"] = predfile["Date"]
        """
        rdf["Open"] = r_y.flatten()
        rdf["Date"] = np.array(stk_test["Date"][WINDOW_SIZE:])
        tdf = pd.DataFrame()
        tdf["Open"] = r_test.flatten()
        tdf["Date"] = np.array(stk_test["Date"][WINDOW_SIZE:])
        """
        print(rdf)
        vts.stock_view(rdf, wtitle=stk, axes=ax)
        vts.stock_view(tdf, wtitle=stk, axes=ax)
        #vts.stock_view(aapl)
        #plt.legend()
        ind += 1
    plt.show()

import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

def stock_get(stock, fromDate=None, toDate=None):
    """
    Reads stock data to dataframe.

    :param stock: stock name
    :param value: str
    :param fromDate: from date
    :param value: str
    :param toDate: to date
    :param value: str
    :return: Stock data in specified timeframe
    :rtype: pandas.core.frame.DataFrame
    """
    df = pd.read_csv("../data/" + stock + ".csv")
    df["Date"] = pd.to_datetime(df["Date"])
    if fromDate == None:
        fromDate = df["Date"].min()
    else:
        fromData= datetime.strptime(fromDate, "%Y-%m-%d")
    if toDate == None:
        toDate = df["Date"].max()
    else:
        toDate = datetime.strptime(toDate, "%Y-%m-%d")
    df = df[(fromDate <= df["Date"]) & (df["Date"] < toDate)]
    return df

def stock_view(filename, fromDate=None, toDate=None, axes=None):
    """
    Visualize stock data.

    :param filename: stock name
    :param value: str
    :param fromDate: from date
    :param value: str
    :param toDate: to date
    :param value: str
    :return fig: matplotlib axes containing figure
    :rtype: matplotlib.axes.Axes
    """
    df = stock_get(filename, fromDate, toDate)
    fig = sns.lineplot(x="Date", y="Open", data=df, ax=axes)
    fig.set(title = filename, ylabel="Open ($)")
    return fig

if __name__ == "__main__":
    ind = 0
    for stk in ["AAPL", "AMZN", "FB", "GOOG", "MSFT"]:
        ax = plt.subplot2grid((2,3), (int(ind/3),ind%3))
        stock_view(stk, "2013-01-01", "2018-01-01", ax)
        stock_view(stk, "2018-01-01", "2019-01-01", ax)
        ind += 1
    plt.show()

import sys
sys.path.insert(0, "E:\Documents\FRI\predmeti\letnik_01\zimni semester\Strojno učenje\seminarska naloga")

from preprocessing.arima import arima_residuals
import visualization.view_time_series as vts
import os

if __name__ == "__main__":
    for stk in ["AAPL", "AMZN", "FB", "GOOG", "MSFT"]:
        stk_df = vts.stock_get(stk, "2010-01-01", "2030-01-01")
        stk_rdf = arima_residuals(stk_df, (1,1,1), 60) 
        stk_df = stk_df.reset_index()
        print(stk_rdf)
        print(stk_df)
        for colname in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            stk_rdf[colname] = stk_df[colname] - stk_rdf[colname]
        stk_rdf.to_csv("../data/" + stk + "_residuals.csv", index=False)
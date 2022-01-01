# https://github.com/aj-cloete/pssa/blob/master/Singular%20Spectrum%20Analysis%20Example.ipynb
import sys
sys.path.insert(0, 'E:\Documents\FRI\predmeti\letnik_01\zimni semester\Strojno učenje\seminarska naloga')

from mySSA import mySSA
import visualization.view_time_series as vts

import pandas as pd
import numpy as np

def ssa_residuals(stockname: str, suspected_seasonality: int, embedding_dimension: int):
    df = vts.stock_get(stockname, "2000-01-01", "2030-01-01")
    for i in range(1,8):
        ssa = mySSA(df)
        dft = df.iloc[:,i]
    

aapl = vts.stock_get("AMZN", "2013-01-01", "2018-01-01")
aapl = aapl.iloc[:,1]

ssa = mySSA(aapl)
suspected_seasonality = 30
ssa.embed(embedding_dimension=200, suspected_frequency=suspected_seasonality, verbose=True)
ssa.decompose(verbose=True)

from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt 
rcParams['figure.figsize'] = 11, 4

ssa.view_s_contributions()
plt.show()

rcParams['figure.figsize'] = 11, 2
for i in range(10):
    ssa.view_reconstruction(ssa.Xs[i], names=i, symmetric_plots=i!=0)
rcParams['figure.figsize'] = 11, 4
plt.show()

ssa.ts.plot(title='Original Time Series'); # This is the original series for comparison
streams5 = [i for i in range(8)]
var = ssa.forecast_recurrent(steps_ahead=1, singular_values=streams5, plot=False)
print(var)
reconstructed5 = ssa.view_reconstruction(*[ssa.Xs[i] for i in streams5], names=streams5, return_df=True)
plt.show()

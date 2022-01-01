import sys
sys.path.insert(0, 'E:\Documents\FRI\predmeti\letnik_01\zimni semester\Strojno učenje\seminarska naloga')

import pandas as pd
import visualization.view_time_series as vts
import numpy as np
import matplotlib.pyplot as plt
from pyts.decomposition import SingularSpectrumAnalysis
import matplotlib.pyplot as plt
from plotnine import *

WINDOW_SIZE = 10

def build_timeseries(df, window_size=WINDOW_SIZE):
    df = df.iloc[:,1:7].values
    X = []
    for i in range(window_size, df.shape[0]):
            X.append(df[i-window_size:i, :])
    
    return np.array(X)

def ssa_smoothing(df, cols, window_size=WINDOW_SIZE):
    X = build_timeseries(df)
    print(X.shape)
    ssa = SingularSpectrumAnalysis(window_size=window_size, groups=10)
    print(X[:,:,1])
    X_ssa = ssa.fit_transform(X[:,:,1])

    return X_ssa

if __name__ == "__main__":
    aapl = vts.stock_get("AAPL", "2013-01-01", "2018-01-01")
    X_ssa = ssa_smoothing(aapl, None, window_size=WINDOW_SIZE)
    print(X_ssa[0,1])
    print(X_ssa.shape)

    aapl = aapl.iloc[WINDOW_SIZE:, :]

    for i in range(X_ssa.shape[1]):
        aapl["vawe" + str(i)] = X_ssa[:,i,0]

    print(aapl["vawe1"])

    g = (
        ggplot(aapl, aes(x="Date", y="Open")) +
        geom_line() +
        geom_line(aes(y="vawe1"))
    )

    print(g)
"""
# Parameters
n_samples, n_timestamps = 100, 48

# Toy dataset
rng = np.random.RandomState(41)
X = rng.randn(n_samples, n_timestamps)
print(X.shape)
print(X[1,:])
print(X[2,:])

# We decompose the time series into three subseries
groups = [np.arange(i, i + 5) for i in range(0, 11, 5)]

# Singular Spectrum Analysis
ssa = SingularSpectrumAnalysis(window_size=10)
X_ssa = ssa.fit_transform(X)
print(ssa.get_params())

# Show the results for the first time series and its subseries
plt.figure(figsize=(16, 6))

ax1 = plt.subplot(121)
ax1.plot(X[0], 'o-', label='Original')
ax1.legend(loc='best', fontsize=14)

ax2 = plt.subplot(122)
print(X_ssa[0,1])
X_sum = np.zeros(X_ssa[0, 0].shape)
for i in range(1, len(groups)):
    X_sum += X_ssa[0, i]
ax2.plot(X_sum, 'o--', label='SSA {0}'.format(10))
for i in range(len(groups)):
    ax2.plot(X_ssa[0, i], 'o--', label='SSA {0}'.format(i + 1))

ax2.legend(loc='best', fontsize=14)

plt.suptitle('Singular Spectrum Analysis', fontsize=20)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
print(aapl.head())
plt.show()

# The first subseries consists of the trend of the original time series.
# The second and third subseries consist of noise.

"""

import pandas as pd

Dataset = pd.read_csv('data/archive/dow_historic_2000_2020.csv')

print(Dataset['stock'].unique())
print(len(Dataset['stock'].unique()))

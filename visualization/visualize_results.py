import pandas as pd

res = pd.read_csv("../results.csv")
res.set_index("ALGORITHM")
res.sort_values(by="COMPANY")

res_rmse = res[res["TYPE"] == "RMSE"]
res_rmse = res_rmse[["ALGORITHM", "COMPANY", "ERROR"]]
res_mape = res[res["TYPE"] == "MAPE"]
res_mape = res_mape[["ALGORITHM", "COMPANY", "ERROR"]]

print(res_rmse.sort_values(by="COMPANY"))
print(res_mape.sort_values(by="COMPANY"))

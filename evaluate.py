from pipeFord import pipeFord
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
from sklearn.model_selection import cross_val_score
import pandas as pd

def evaluate(x_train, x_valid, y_train, y_valid):
    pipe_xgb = pipeFord()
    pipe_xgb.fit(x_train,y_train)
    xgb_pred = pipe_xgb.predict(x_valid)
    xgb_pred_train = pipe_xgb.predict(x_train)
    scores_xgb = cross_val_score(pipe_xgb, x_train, y_train, cv=5)
    MAE = mean_absolute_error(y_valid, xgb_pred)
    MAE = float('{:.2f}'.format(MAE))
    MSE = mean_squared_error(y_valid, xgb_pred)
    MSE = float('{:.2f}'.format(MSE))
    RMSE = mean_squared_error(y_valid, xgb_pred, squared=False)
    RMSE = float('{:.2f}'.format(RMSE))
    MAE_treino = mean_absolute_error(y_train, xgb_pred_train)
    MAE_treino = float('{:.2f}'.format(MAE_treino))
    MSE_treino = mean_squared_error(y_train, xgb_pred_train)
    MSE_treino = float('{:.2f}'.format(MSE_treino))
    RMSE_treino = mean_squared_error(y_train, xgb_pred_train, squared=False)
    RMSE_treino = float('{:.2f}'.format(RMSE_treino))
    Colunas = ["MAE", "MSE", "RMSE"]
    Index = ['Teste', 'Treino']
    comparativo = pd.DataFrame(columns=Colunas, index=Index)
    comparativo['MAE'] = [MAE, MAE_treino]
    comparativo['MSE'] = [MSE, MSE_treino]
    comparativo['RMSE'] = [RMSE, RMSE_treino]
    return (print(np.mean(scores_xgb)),comparativo)


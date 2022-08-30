# Importando bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb

#Carregando o Data Frame
df = pd.read_csv("Projeto_Ford.csv")

#Treino, Teste e Validação

y = df.price
x = df.drop(columns=['price'])

# Separando treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

# Separando teste e validação
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.15)

#Pipeline
# Variaveis One Hot Enconder
cat_columns = ['model','transmission','fuelType']
#Variaveis Min e Max
num_columns = ['year','mileage','mpg','engineSize','Km/L']

#Definindo os pipelines das variaveis categoricas e numericas
pipe_cat = make_pipeline(OneHotEncoder(handle_unknown = 'ignore'))
pipe_num = make_pipeline(MinMaxScaler())

#Pipeline completo
pipe_full =ColumnTransformer([
    ("num",pipe_num,num_columns),
    ("cat",pipe_cat,cat_columns)
     ])

pipe_xgb =  make_pipeline(pipe_full, xgb.XGBRegressor(random_state=7))

pipe_xgb.fit(x_train,y_train)
xgb_pred = pipe_xgb.predict(x_valid)
xgb_pred_x = pipe_xgb.predict(x_train)



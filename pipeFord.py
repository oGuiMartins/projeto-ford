from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

def pipeFord():
    # Pipeline
    # Variaveis One Hot Enconder
    cat_columns = ['model', 'transmission', 'fuelType']
    # Variaveis Min e Max
    num_columns = ['year', 'mileage', 'mpg', 'engineSize']

    # Definindo os pipelines das variaveis categoricas e numericas
    pipe_cat = make_pipeline(OneHotEncoder(handle_unknown='ignore'))
    pipe_num = make_pipeline(MinMaxScaler())

    # Pipeline completo
    pipe_full = ColumnTransformer([
        ("num", pipe_num, num_columns),
        ("cat", pipe_cat, cat_columns)
    ])

    pipe_xgb = make_pipeline(pipe_full, xgb.XGBRegressor(random_state=0,
                                                            learning_rate = 0.01,
                                                            n_estimators = 1000))

    return pipe_xgb


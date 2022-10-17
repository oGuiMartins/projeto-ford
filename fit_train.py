import joblib as jb
from pipeFord import pipeFord
from choosing_car_model import choosing_car_model
from train_test_valid_split import train_test_valid_split

def fit_train(df,car_model):
    #Carregando DF por modelo do carro
    df = choosing_car_model(df,car_model)

    #Selecionando toda a base
    x = train_test_valid_split(df,'all')[6]
    y = train_test_valid_split(df,'all')[7]

    #Pipeline
    pipe_xgb = pipeFord()

    #Fit e Predict
    pipe_xgb.fit(x,y)

    jb.dump(pipe_xgb, f"{car_model}_xgb_pred.pkl.z")



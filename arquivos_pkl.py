from fit_train import fit_train
from choosing_car_model import cars_with_most_sales
import pandas as pd

df = pd.read_csv("Projeto_Ford.csv")

cars_with_most_sales = cars_with_most_sales(df)

for car in cars_with_most_sales:
    fit_train(df,car)

fit_train(df,'others')
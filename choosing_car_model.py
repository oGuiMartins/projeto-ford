import pandas as pd

limit = 199

def choosing_car_model(df,car_model):
    df = df

    qtd_carros = df.model.value_counts()

    cars_with_most_sales = pd.DataFrame((qtd_carros >= limit)).query('model==True').index

    if car_model in cars_with_most_sales:
        return df[df.model == car_model]
    else:
        return df[~df.model.isin(cars_with_most_sales)]


def cars_with_most_sales(df):
    qtd_carros = df.model.value_counts()

    cars_with_most_sales = pd.DataFrame((qtd_carros >= limit)).query('model==True').index

    return cars_with_most_sales

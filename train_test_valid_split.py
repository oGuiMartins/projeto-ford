from sklearn.model_selection import train_test_split

def train_test_valid_split(df,base):

    if base == "train-test-valid":
        # Treino, Teste e Validação

        y = df.price
        x = df.drop(columns=['price'])

        # Separando treino e teste
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=7)

        # Separando teste e validação
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.15, random_state=7)

        return [x_test,y_test,x_train, x_valid, y_train, y_valid]

    elif base == "train-test":
        y = df.price
        x = df.drop(columns=['price'])

        # Separando treino e teste
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=7)
        return [x_test, y_test, x_train, y_train]

    else:
        # Treino, Teste e Validação

        y = df.price
        x = df.drop(columns=['price'])

        # Separando treino e teste
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=7)

        # Separando teste e validação
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.15, random_state=7)

        return [x_test, y_test, x_train, x_valid, y_train, y_valid, x, y]



#retornar separado vario return
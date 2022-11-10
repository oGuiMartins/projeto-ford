from flask import Flask, request, render_template
import joblib as jb
import pandas as pd
import numpy as np
import json

app = Flask(__name__,template_folder='Arsha',static_folder='Arsha/assets')

#carregar um modelo por vez
xgb_fiesta = jb.load("fiesta_xgb_pred.pkl.z")
xgb_focus = jb.load("focus_xgb_pred.pkl.z")
xgb_kuga = jb.load("kuga_xgb_pred.pkl.z")
xgb_ecosport = jb.load("ecosport_xgb_pred.pkl.z")
xgb_cmax = jb.load("c-max_xgb_pred.pkl.z")
xgb_ka_plus = jb.load("ka+_xgb_pred.pkl.z")
xgb_mondeo = jb.load("mondeo_xgb_pred.pkl.z")
xgb_bmax = jb.load("b-max_xgb_pred.pkl.z")
xgb_smax = jb.load("s-max_xgb_pred.pkl.z")
xgb_grand_cmax = jb.load("grand c-max_xgb_pred.pkl.z")
xgb_galaxy = jb.load("galaxy_xgb_pred.pkl.z")
xgb_edge = jb.load("edge_xgb_pred.pkl.z")
xgb_ka = jb.load("ka_xgb_pred.pkl.z")
xgb_others = jb.load("others_xgb_pred.pkl.z")

@app.route('/')
def home():
    return render_template("index.html")

def get_data():
    model = request.form.get('Model')
    year = request.form.get('Year')
    transmission = request.form.get('Transmission')
    mileage = request.form.get('Mileage')
    fueltype = request.form.get('fuelType')
    enginesize = request.form.get('engineSize')
    mpg = request.form.get('mpg')

    d_dict = {'model': [model], 'year':[year], 'transmission':[transmission], 'mileage':[mileage],
    'fuelType':[fueltype], 'mpg':[mpg], 'engineSize':[enginesize]}

    return pd.DataFrame.from_dict(d_dict,orient='columns')

@app.route('/send', methods=['POST'])
def show_data():
    df = get_data()
    df = df[['model', 'year', 'transmission', 'mileage', 'fuelType', 'mpg', 'engineSize']]
    df.model = df.model.str.lower()
    df.transmission = df.transmission.str.lower()
    df.fuelType = df.fuelType.str.lower()
    df.engineSize =df.engineSize.str.lower()
    #selecionando o modelo
    if df.model[0] == 'fiesta':
        prediction = xgb_fiesta.predict(df)
        prediction = '{:0,.2f}'.format(prediction[0])
        print('fiesta')
    elif df.model[0] == 'focus':
        prediction = xgb_focus.predict(df)
        prediction = '{:0,.2f}'.format(prediction[0])
        print('focus')
    elif df.model[0] == 'kuga':
        prediction = xgb_kuga.predict(df)
        prediction = '{:0,.2f}'.format(prediction[0])
    elif df.model[0] == 'ecosport':
        prediction = xgb_ecosport.predict(df)
        prediction = '{:0,.2f}'.format(prediction[0])
    elif df.model[0] == 'c-max':
        prediction = xgb_cmax.predict(df)
        prediction = '{:0,.2f}'.format(prediction[0])
    elif df.model[0] == 'ka+':
        prediction = xgb_ka_plus.predict(df)
        prediction = '{:0,.2f}'.format(prediction[0])
    elif df.model[0] == 'mondeo':
        prediction = xgb_mondeo.predict(df)
        prediction = '{:0,.2f}'.format(prediction[0])
    elif df.model[0] == 'b-max':
        prediction = xgb_bmax.predict(df)
        prediction = '{:0,.2f}'.format(prediction[0])
    elif df.model[0] == 's-max':
        prediction = xgb_smax.predict(df)
        prediction = '{:0,.2f}'.format(prediction[0])
    elif df.model[0] == 'grand c-max':
        prediction = xgb_grand_cmax.predict(df)
        prediction = '{:0,.2f}'.format(prediction[0])
    elif df.model[0] == 'galaxy':
        prediction = xgb_galaxy.predict(df)
        prediction = '{:0,.2f}'.format(prediction[0])
    elif df.model[0] == 'edge':
        prediction = xgb_edge.predict(df)
        prediction = '{:0,.2f}'.format(prediction[0])
    elif df.model[0] == 'ka':
        prediction = xgb_edge.predict(df)
        prediction = '{:0,.2f}'.format(prediction[0])
    else:
        prediction = xgb_others.predict(df)
        prediction = '{:0,.2f}'.format(prediction[0])
        print('outros')


    return render_template('inner-page.html', result=prediction, model=df.model[0], year=df.year[0],
                           transmission=df.transmission[0], mileage=df.mileage[0], fueltype=df.fuelType[0],mpg=df.mpg[0],
                           enginesize=df.engineSize[0])



@app.route('/api/<val>')
def api(val):
    consult_df = pd.DataFrame(columns=['model', 'year', 'transmission', 'mileage', 'fuelType', 'mpg', 'engineSize'])
    consult_list = val.strip('][').lower().split(',')
    consult_df.loc[0] = {'model':consult_list[0],
                        'year':int(consult_list[1]),
                        'transmission':consult_list[2],
                        'mileage':int(consult_list[3]),
                        'fuelType':consult_list[4],
                        'mpg':np.float(consult_list[5]),
                        'engineSize':np.float(consult_list[6])}
    consult_df = consult_df[['model', 'year', 'transmission', 'mileage', 'fuelType', 'mpg', 'engineSize']]

    # selecionando o modelo
    if consult_df.model[0] == 'fiesta':
        prediction = xgb_fiesta.predict(consult_df)
    elif consult_df.model[0] == 'focus':
        prediction = xgb_focus.predict(consult_df)
    elif consult_df.model[0] == 'kuga':
        prediction = xgb_kuga.predict(consult_df)
    elif consult_df.model[0] == 'ecosport':
        prediction = xgb_ecosport.predict(consult_df)
    elif consult_df.model[0] == 'c-max':
        prediction = xgb_cmax.predict(consult_df)
    elif consult_df.model[0] == 'ka+':
        prediction = xgb_ka_plus.predict(consult_df)
    elif consult_df.model[0] == 'mondeo':
        prediction = xgb_mondeo.predict(consult_df)
    elif consult_df.model[0] == 'b-max':
        prediction = xgb_bmax.predict(consult_df)
    elif consult_df.model[0] == 's-max':
        prediction = xgb_smax.predict(consult_df)
    elif consult_df.model[0] == 'grand c-max':
        prediction = xgb_grand_cmax.predict(consult_df)
    elif consult_df.model[0] == 'galaxy':
        prediction = xgb_galaxy.predict(consult_df)
    elif consult_df.model[0] == 'edge':
        prediction = xgb_edge.predict(consult_df)
    elif consult_df.model[0] == 'ka':
        prediction = xgb_edge.predict(consult_df)
    else:
        prediction = xgb_others.predict(consult_df)

    res = {"Valor":int(prediction),'Modelo':consult_df.model[0],'Ano':consult_df.year[0],
           "Tipo de transmissao": consult_df.transmission[0], 'Milhas': consult_df.mileage[0],
           'fuelType':consult_df.fuelType[0],'Consumo':consult_df.mpg[0], 'Potencia do Motor':consult_df.engineSize[0]}


    return json.dumps(res)

if __name__=='__main__': 
    app.run(host='0.0.0.0', port=8080)

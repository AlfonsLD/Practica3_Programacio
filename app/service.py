import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

species = ['Adelie', 'Chinstrap', 'Gentoo']

categorical = ['island', 'sex']
numerical = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']  


def predict_single(penguin, dv, sc, model):
    penguin_df = pd.DataFrame([penguin])
    
    penguin_cat = dv.transform(penguin_df[categorical].to_dict(orient='records'))
    penguin_num = sc.transform(penguin_df[numerical])
    penguin_std = np.hstack([penguin_cat, penguin_num])
    
    y_pred = model.predict(penguin_std)[0]
    y_prob = model.predict_proba(penguin_std)[0][y_pred]
    return (y_pred, y_prob)

def predict(dv, sc, model):
    penguin = request.get_json()
    especie, probabilitat = predict_single(penguin, dv, sc, model)
   
    result = {
        'penguin': species[especie],
        'probability': float(probabilitat)
    }
    return jsonify(result)

app = Flask('penguins')


@app.route('/predict_lr', methods=['POST'])
def predict_lr():
    with open('models/lr.pck', 'rb') as f:
        dv, sc, model = pickle.load(f)
    return predict(dv, sc, model)

@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    with open('models/svm.pck', 'rb') as f:
        dv, sc, model = pickle.load(f)
    return predict(dv, sc, model)

@app.route('/predict_dt', methods=['POST'])
def predict_dt():
    with open('models/dt.pck', 'rb') as f:
        dv, sc, model = pickle.load(f)
    return predict(dv, sc, model)

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    with open('models/knn.pck', 'rb') as f:
        dv, sc, model = pickle.load(f)
    return predict(dv, sc, model)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
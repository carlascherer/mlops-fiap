import os
import logging
import json
import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, jsonify

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

app = Flask(__name__)
app.json_encoder = NpEncoder

def init():
    global encoder
    global scaler
    global kmeans
    global stats_label
    
    encoder = pickle.load(open('docker/artifacts/encoder_categorias.pkl', 'rb'))
    scaler = pickle.load(open('docker/artifacts/normalizador.pkl', 'rb'))
    kmeans = pickle.load(open('docker/artifacts/modelo_kmeans.pkl', 'rb'))
    stats_label = pd.read_csv('docker/stats_label.csv')

    logging.info("Init complete")

def get_score_description(score):
    if score < 600:
        return "baixa"
    elif score < 700:
        return "media"
    elif score >= 700:
        return "alta"

def get_persona(row):
    return f"Uma pessoa com pontuação {get_score_description(row['PontuacaoCredito'])}, vivendo na {row['Regiao']}, com aproximadamente {row['Idade']} anos."

@app.route("/predict", methods=['POST'])
def call_predict():
    data = pd.DataFrame(request.json, index=[0])

    # Transformar dados com o encoder
    data = encoder.transform(data[['loan_purpose', 'Security_Type', 'Region', 'age']])

    # Escalar os dados
    data_scaled = scaler.transform(data)

    # Predição
    prediction = kmeans.predict(data_scaled)[0]
    stats_label_selected = stats_label.query("label == @prediction")

    persona = get_persona(data.iloc[0])

    ret = json.dumps({
        'prediction_label': prediction,
        'defaulted_ratio': stats_label_selected.iloc[0]['defaulted_ratio'],
        'mean_credit_score_defaulted': stats_label_selected.iloc[0]['mean_credit_score_defaulted'],
        'persona': persona
    }, cls=NpEncoder)

    return jsonify(ret)

@app.route("/", methods=['GET'])
def call_home():
    return "SERVER IS RUNNING!"

if __name__ == '_main_':
    init()
    app.run(port=8080, host='0.0.0.0')
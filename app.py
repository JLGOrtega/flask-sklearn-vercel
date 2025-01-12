from flask import Flask, request, jsonify, render_template
import datetime
import pickle
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
from sqlalchemy import create_engine 


app = Flask(__name__)

# Cargar modelo
with open("titanic_model.pkl", "rb") as f:
   modelo = pickle.load(f)

def get_ts():
    timestamp = datetime.datetime.now().isoformat()
    return timestamp[0:19]

churro = "postgresql://postgres:postgres@35.233.106.171:5432/postgres"
engine = create_engine(churro)

@app.route('/')
def formulario():
    return render_template('formulario.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    # Obtener los datos del formulario
    caracteristicas = [float(x) for x in request.form.values()]
    print(caracteristicas)
    entrada = [caracteristicas]

    # Realizar la predicción
    prediccion = modelo.predict(entrada)
    df_logs = pd.DataFrame({"inputs":[entrada], "predictions": [prediccion[0]], "timestamp": get_ts()})
    df_logs.to_sql("logs", con=engine, index=None, if_exists="append")
    print(df_logs)
    all_logs = pd.read_sql("""SELECT * FROM logs""", con=engine)

    fig = plt.figure()
    all_logs.predictions.value_counts().plot(kind="bar");
    plt.title(f"PREDICTION BALANCE up to: {all_logs.timestamp.max()}")


    # Guardar gráfica en memoria como bytes
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)

    # Convertir la gráfica a una base64 para incrustarla en HTML
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return render_template('resultado.html', prediccion=prediccion[0], grafica=img_base64)
    

if __name__ == '__main__':
    app.run(debug=True)
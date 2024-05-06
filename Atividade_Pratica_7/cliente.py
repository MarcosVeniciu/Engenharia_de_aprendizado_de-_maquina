from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import Pyro5.api
import json


def main():
    # Conecte-se ao servidor remoto
    #uri = Pyro5.api.locate_ns().lookup("servidor")
    uri = Pyro5.api.locate_ns(host="\"3.86.82.136\"", port=9090).lookup("servidor")

    proxy = Pyro5.api.Proxy(uri)

    # O dataset no csv ja esta processado
    input_data = pd.read_csv("df_teste.csv")
    input_data = input_data.drop("Unnamed: 0", axis=1)
    
    X = input_data.drop("Exited", axis=1)  # Remove da conjunto de teino o atributo Exited, pois ele sera usado para classificar os dados
    y = input_data["Exited"]

    json_data = X.to_json(orient='records')
    # Envie os dados para o servidor e receba a classificação
    prediction = proxy.classificar(json_data)
    prediction = json.loads(prediction)
    actual_labels = np.array(y)

    print('Accuracy: ', round(float(accuracy_score(actual_labels, prediction))*100, 4), '%')
    print('Accuracy Balanced: ', round(float(balanced_accuracy_score(actual_labels, prediction))*100, 4), '%')
    print('\nClassification Stats:')
    print(classification_report(actual_labels, prediction))


if __name__ == "__main__":
    main()

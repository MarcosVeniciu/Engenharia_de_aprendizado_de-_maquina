from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from joblib import dump
import pandas as pd
import numpy as np



df = pd.read_csv("Churn_Bank.csv")
print("Quantidade de registros: " + str(len(df)))

df = df.drop("Unnamed: 0", axis=1)
df = df.drop("RowNumber", axis=1) # numero da linha
df = df.drop("CustomerId", axis=1) # id do cliente
df = df.drop("Surname", axis=1) # Sobrenome dos clientes

le = LabelEncoder()
Geography = df['Geography']
encoded = le.fit_transform(df["Geography"])
df = df.drop("Geography", axis=1)
df["Geography"] = encoded

Gender = df['Gender']
encoded = le.fit_transform(df["Gender"])
df = df.drop("Gender", axis=1) # sobre nome
df["Gender"] = encoded

scaler = MinMaxScaler()
atributos_floats = ["Balance", "EstimatedSalary"]
df[atributos_floats] = scaler.fit_transform(df[atributos_floats])

idade_intervalo = [0, 20, 30, 40, 50, 60, 70, 80 ,90 ,100 ]
df["Age"] = pd.cut(df["Age"], idade_intervalo, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9])

credito_intervalo = [300, 400, 500, 600, 700, 800, 900 ]
nomes = [1, 2, 3, 4, 5, 6]
df["CreditScore"] = pd.cut(df["CreditScore"], credito_intervalo, labels=nomes)

df.head(5)



X = df.drop("Exited", axis=1)  # Remove da conjunto de teino o atributo Exited, pois ele sera usado para classificar os dados
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42) # separa 10% dos dados para teste
print("Dados para Treino: " + str(len(X_train)))
print("Dados para Teste: " + str(len(X_test)))


modelos = RandomForestClassifier(n_estimators=86, random_state=42)

modelos.fit(X_train, y_train)

pred_labels = modelos.predict(X_test)
actual_labels = np.array(y_test)

print('Accuracy: ', round(float(accuracy_score(actual_labels, pred_labels))*100, 4), '%')
print('Accuracy Balanced: ', round(float(balanced_accuracy_score(actual_labels, pred_labels))*100, 4), '%')
print('\nClassification Stats:')
print(classification_report(actual_labels, pred_labels))
dump(modelos, 'modelo_salvo.joblib')
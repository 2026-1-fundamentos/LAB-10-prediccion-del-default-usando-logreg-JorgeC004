# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# ------------------------------------------------------------------------------
# Rutas
# ------------------------------------------------------------------------------
INPUT_TRAIN = "files/input/train_data.csv.zip"
INPUT_TEST = "files/input/test_data.csv.zip"
MODEL_PATH = "files/models/model.pkl.gz"
METRICS_PATH = "files/output/metrics.json"


# ------------------------------------------------------------------------------
# Paso 1: limpieza
# ------------------------------------------------------------------------------
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])
    df = df.dropna()

    # Filtrar registros con "no disponible" codificados como 0 en
    # EDUCATION o MARRIAGE (según el diccionario del dataset).
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]

    # Agrupar EDUCATION > 4 en la categoría "others" (4).
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    return df


def load_and_clean_data():
    train_df = pd.read_csv(INPUT_TRAIN, index_col=False, compression="zip")
    test_df = pd.read_csv(INPUT_TEST, index_col=False, compression="zip")

    train_df = clean_dataset(train_df)
    test_df = clean_dataset(test_df)

    return train_df, test_df


# ------------------------------------------------------------------------------
# Paso 2: split x/y
# ------------------------------------------------------------------------------
def split_data(df: pd.DataFrame):
    x = df.drop(columns=["default"])
    y = df["default"]
    return x, y


# ------------------------------------------------------------------------------
# Paso 3: pipeline
# ------------------------------------------------------------------------------
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]


def make_pipeline(x_train: pd.DataFrame) -> Pipeline:
    numerical_features = [
        col for col in x_train.columns if col not in CATEGORICAL_FEATURES
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", MinMaxScaler(), numerical_features),
        ],
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_classif)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ],
    )

    return pipeline


# ------------------------------------------------------------------------------
# Paso 4: optimización de hiperparámetros
# ------------------------------------------------------------------------------
def make_grid_search(pipeline: Pipeline) -> GridSearchCV:
    param_grid = {
        "feature_selection__k": range(1, 25),
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__solver": ["liblinear", "lbfgs"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )

    return grid_search


# ------------------------------------------------------------------------------
# Paso 5: guardar el modelo
# ------------------------------------------------------------------------------
def save_model(model):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with gzip.open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)


# ------------------------------------------------------------------------------
# Paso 6 y 7: métricas y matrices de confusión
# ------------------------------------------------------------------------------
def compute_metrics(dataset_name: str, y_true, y_pred) -> dict:
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def compute_confusion_matrix(dataset_name: str, y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {
            "predicted_0": int(cm[0][0]),
            "predicted_1": int(cm[0][1]),
        },
        "true_1": {
            "predicted_0": int(cm[1][0]),
            "predicted_1": int(cm[1][1]),
        },
    }


def save_metrics(records: list):
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    train_df, test_df = load_and_clean_data()

    x_train, y_train = split_data(train_df)
    x_test, y_test = split_data(test_df)

    pipeline = make_pipeline(x_train)
    model = make_grid_search(pipeline)
    model.fit(x_train, y_train)

    save_model(model)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    records = [
        compute_metrics("train", y_train, y_train_pred),
        compute_metrics("test", y_test, y_test_pred),
        compute_confusion_matrix("train", y_train, y_train_pred),
        compute_confusion_matrix("test", y_test, y_test_pred),
    ]

    save_metrics(records)


if __name__ == "__main__":
    main()

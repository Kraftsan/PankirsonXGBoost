'''
Задача 2. Обнаружение болезни паркинсона с помощью XGBoost
Твоя задача с помощью Data Science предсказать заболевание
паркинсона на ранней стадии, используя алгоритм машинного
обучения XGBoost и библиотеку sklearn для нормализации признаков.
Как это сделать? Тебя  придется самостоятельно изучить данный вопрос.

Используй следующий датасет UCI ML Parkinsons. Описание признаков и
меток датасета представлены здесь. От тебя  требуется помимо создания
самой модели получить ее точность на тестовой выборке. Выборки делить в
соотношении 80% обучающая, 20% - тестовая.

Дополнительные баллы ты получишь, если сможешь получить точность более 95%.

'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
def load_data():
    df = pd.read_csv('parkinsons.data')
    return df

def preprocessData(df):
    print(df.head())
    print(df.describe())
    print(df.info())
    print(df.shape)
    print(df.isnull().sum())

    # разделение на признаки
    X = df.drop(['status', 'name'], axis=1)  # удаляем целевую переменную и имя пациента
    y = df['status']  # целевая переменная

    return X, y

def main():
    df = load_data()

    X, y = preprocessData(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nРазмер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    return X, y
if __name__ == "__main__":

    model, accuracy = main()
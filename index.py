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

def trainModel(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Создание модели XGBoost
    model = xgb.XGBClassifier(
        learning_rate=0.01,
        n_estimators=100,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
        verbose=True
    )
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    score = accuracy_score(y_test, y_pred)
    print(score)
    return model, scaler, y_pred, y_pred_proba


def evaluteModel(y_test, y_pred):
    # Точность
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Точность модели {accuracy * 100:.2f}%')

    print("\nОтчет классификации:")
    print(classification_report(y_test, y_pred))

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Матрица ошибок')
    plt.ylabel('Истинные значения')
    plt.xlabel('Предсказанные значения')
    plt.show()

    return accuracy

def plotFeatureImport(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 6))
    plt.title('Важность признаков')
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()

    # Вывод топ 10
    print('ТОП 10 самый важных признаков')
    for i in range(min(10, len(importances))):
        print(f'{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}%')

def improveModel(X_train, X_test, y_train, y_test, scaler):
    # настройка гиперпараметров
    from sklearn.model_selection import GridSearchCV

    # Маштабируем данные
    X_train_scaler = scaler.fit_transform(X_train)
    X_test_scaler = scaler.transform(X_test)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }

    model = xgb.XGBClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator = model,
        param_grid = param_grid,
        cv = 5,
        scoring = 'accuracy',
        n_jobs = -1,
        verbose = 1,
    )

    grid_search.fit(X_train_scaler, y_train)

    print(f'Лучшие параметры {grid_search.best_params_}')
    print(f'Лучшая точность {grid_search.best_score_:.4f}')

    return grid_search.best_estimator_

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

    # Обучение модели
    model, scaler, y_pred, y_pred_proba = trainModel(X_train, X_test, y_train, y_test)

    accuracy = evaluteModel(y_test, y_pred)

    # Важность признаков
    plotFeatureImport(model, X.columns.tolist())

    # Улучшение модели
    improved_model = improveModel(X_train, X_test, y_train, y_test, scaler)

    return model, accuracy

if __name__ == "__main__":
    model, accuracy = main()
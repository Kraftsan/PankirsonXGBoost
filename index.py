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
from numpy.random.mtrand import gamma
from sklearn.model_selection import StratifiedKFold
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

    model = xgb.XGBClassifier(
        learning_rate=0.05,
        n_estimators=300,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=50
    )

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    score = accuracy_score(y_test, y_pred)
    print(f"Точность базовой модели: {score * 100:.2f}%")
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
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
    }

    model = xgb.XGBClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train_scaled, y_train)

    print(f'Лучшие параметры: {grid_search.best_params_}')
    print(f'Лучшая точность на кросс-валидации: {grid_search.best_score_ * 100:.2f}%')

    # оценим на тестовых данных
    best_model = grid_search.best_estimator_
    y_pred_improved = best_model.predict(X_test_scaled)
    improved_accuracy = accuracy_score(y_test, y_pred_improved)
    print(f'Точность улучшенной модели на тесте: {improved_accuracy * 100:.2f}%')

    return best_model

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

    X_test_scaled = scaler.transform(X_test)
    y_pred_improved = improved_model.predict(X_test_scaled)
    improved_accuracy = accuracy_score(y_test, y_pred_improved)
    print(f'Точность улучшенной модели {improved_accuracy*100:.2f}%')

    return model, accuracy

if __name__ == "__main__":
    model, accuracy = main()
"""Пример анализа набора данных Glass Identification."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import kagglehub


# Шаг 1. Загрузка датасета с Kaggle.
# kagglehub сам управляет кэшем и не требует учётных данных.
data_path = Path(kagglehub.dataset_download("uciml/glass"))
# В наборе только один CSV-файл, находим его автоматически.
csv_path = next(data_path.glob("*.csv"))

# Шаг 2. Чтение CSV в DataFrame и настройка типов.
# Все признаки числовые, столбец "Type" трактуем как категориальный код (целые числа).
dtype_map = {
    "RI": "float64",
    "Na": "float64",
    "Mg": "float64",
    "Al": "float64",
    "Si": "float64",
    "K": "float64",
    "Ca": "float64",
    "Ba": "float64",
    "Fe": "float64",
    "Type": "int64",
}
df = pd.read_csv(csv_path, dtype=dtype_map)

print("Данные загружены из файла:", csv_path)
print("Размерность датафрейма:", df.shape)
print("Первые строки данных:")
print(df.head())

# Отделяем признаки и целевую переменную.
feature_columns = [col for col in df.columns if col != "Type"]
X_raw = df[feature_columns]
y = df["Type"]

# Шаг 3. Стандартизация.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
print("Стандартизация выполнена. Средние значения по признакам (должны быть около 0):")
print(X_scaled.mean(axis=0))

# Шаг 4. Разделение на обучающую и тестовую выборки.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.4, random_state=42, stratify=y
)
print("Размер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)

# Шаг 5. Перебор параметров дерева решений.
alpha_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.2, 0.8]
criteria = ["entropy", "gini"]
results = []

for criterion in criteria:
    for alpha in alpha_values:
        clf = DecisionTreeClassifier(
            criterion=criterion,
            ccp_alpha=alpha,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        results.append(
            {
                "criterion": criterion,
                "alpha": alpha,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "depth": clf.get_depth(),
                "leaves": clf.get_n_leaves(),
            }
        )

results_df = pd.DataFrame(results)
print("Результаты подбора гиперпараметров (отсортировано по точности на тесте):")
print(results_df.sort_values(by=["test_accuracy", "train_accuracy"], ascending=False))

best_row = results_df.sort_values(by=["test_accuracy", "train_accuracy"], ascending=False).iloc[0]
print("Лучшее дерево без снижения размерности:")
print(best_row)

# Шаг 6. PCA и построение дерева в пространстве главных компонент.
pca = PCA()
pca.fit(X_train)
explained = np.cumsum(pca.explained_variance_ratio_)
# Выбираем минимальное число компонент, объясняющих не менее 95% дисперсии.
n_components = int(np.searchsorted(explained, 0.95) + 1)
print("Кумулятивная объяснённая дисперсия:")
print(explained)
print("Выбрано компонент:", n_components)

pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

pca_results = []
for criterion in criteria:
    for alpha in alpha_values:
        clf = DecisionTreeClassifier(
            criterion=criterion,
            ccp_alpha=alpha,
            random_state=42,
        )
        clf.fit(X_train_pca, y_train)
        train_acc = accuracy_score(y_train, clf.predict(X_train_pca))
        test_acc = accuracy_score(y_test, clf.predict(X_test_pca))
        pca_results.append(
            {
                "criterion": criterion,
                "alpha": alpha,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "depth": clf.get_depth(),
                "leaves": clf.get_n_leaves(),
            }
        )

pca_results_df = pd.DataFrame(pca_results)
print("Результаты после PCA (отсортировано по точности на тесте):")
print(pca_results_df.sort_values(by=["test_accuracy", "train_accuracy"], ascending=False))

best_pca_row = pca_results_df.sort_values(by=["test_accuracy", "train_accuracy"], ascending=False).iloc[0]
print("Лучшее дерево после PCA:")
print(best_pca_row)

# Шаг 7. Сравнение результатов.
print("Сравнение лучших моделей:")
comparison = pd.DataFrame(
    [
        {"Модель": "Без PCA", **best_row.to_dict()},
        {"Модель": "С PCA", **best_pca_row.to_dict()},
    ]
)
print(comparison)

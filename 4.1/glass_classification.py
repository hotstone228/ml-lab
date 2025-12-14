"""Пример анализа набора данных Glass Identification."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt

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

# === Новые расчёты по качеству модели ===
# Обучаем лучшее дерево на исходных признаках и сразу получаем прогнозы для метрик.
best_clf = DecisionTreeClassifier(
    criterion=best_row["criterion"],
    ccp_alpha=best_row["alpha"],
    random_state=42,
)
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)
print("F1 (macro) на тесте:", f1_score(y_test, y_pred, average="macro"))
print("F1 (micro) на тесте:", f1_score(y_test, y_pred, average="micro"))

# Готовим данные для много-классовой ROC-кривой (подход One-vs-Rest).
classes = np.unique(y_train)
y_test_bin = label_binarize(y_test, classes=classes)
y_score = best_clf.predict_proba(X_test)

fpr = {}
tpr = {}
roc_auc = {}
for i, cls in enumerate(classes):
    fpr[cls], tpr[cls], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[cls] = auc(fpr[cls], tpr[cls])

print("ROC-AUC по классам:")
for cls in classes:
    print(f"Класс {cls}: AUC = {roc_auc[cls]:.3f}")

# === Новые графики ===
# 1) Много-классовые ROC-кривые по подходу One-vs-Rest.
plt.figure(figsize=(8, 6))
for cls in classes:
    plt.plot(fpr[cls], tpr[cls], label=f"Класс {cls} (AUC = {roc_auc[cls]:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("Доля ложных срабатываний")
plt.ylabel("Доля истинных срабатываний")
plt.title("ROC-кривые для классов стекла")
plt.legend()
roc_path = Path("roc_curves.png")
plt.tight_layout()
plt.savefig(roc_path)
print("ROC-кривые сохранены в файл:", roc_path)

# 2) Визуализация дерева решений для лучшей модели без PCA.
plt.figure(figsize=(14, 10))
plot_tree(
    best_clf,
    filled=True,
    feature_names=feature_columns,
    class_names=[str(cls) for cls in classes],
    max_depth=4,
    fontsize=8,
)
plt.title("Фрагмент дерева решений (первые уровни)")
tree_path = Path("decision_tree.png")
plt.tight_layout()
plt.savefig(tree_path)
print("Визуализация дерева сохранена в файл:", tree_path)

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

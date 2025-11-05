"""Простая демонстрация приёмов классификации для лабораторной работы 4.2."""

from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# Шаг 1. Загрузка данных
# ---------------------------------------------------------------------------
print("Загружаю набор данных о вине с Kaggle...")
dataset_dir = Path(
    kagglehub.dataset_download("ehsanesmaeili/red-and-white-wine-quality-merged")
)
csv_path = dataset_dir / "wine_quality_merged.csv"

print("Файл с данными:", csv_path)

# ---------------------------------------------------------------------------
# Шаг 2. Создание DataFrame и проверка типов столбцов
# ---------------------------------------------------------------------------
df_raw = pd.read_csv(csv_path)

float_columns = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]

df_raw[float_columns] = df_raw[float_columns].astype("float64")
df_raw["type"] = df_raw["type"].astype("category")

print("Всего записей:", len(df_raw))
print("Доступные признаки:", list(df_raw.columns))
print("Метка класса — столбец 'type' (красное или белое вино).")

# ---------------------------------------------------------------------------
# Шаг 3. Стандартизация числовых признаков
# ---------------------------------------------------------------------------
features_df = df_raw.drop(columns=["type"]).copy()
labels_series = df_raw["type"].map({"red": 0, "white": 1}).astype("int64")

scaler_full = StandardScaler()
features_scaled = scaler_full.fit_transform(features_df.to_numpy(dtype=np.float64))
features_scaled_df = pd.DataFrame(
    features_scaled,
    columns=features_df.columns,
)

print("Стандартизация выполнена. Среднее по признакам (первые 5 значений):")
print(features_scaled_df.mean().head())
print("Стандартное отклонение (первые 5 значений):")
print(features_scaled_df.std(ddof=0).head())

# ---------------------------------------------------------------------------
# Шаг 4. Разделение на обучающую, тестовую и валидационную выборки (5/3/2)
# ---------------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    features_scaled_df.to_numpy(dtype=np.float64),
    labels_series.to_numpy(),
    test_size=0.5,
    random_state=42,
    stratify=labels_series,
)

X_test, X_valid, y_test, y_valid = train_test_split(
    X_temp,
    y_temp,
    test_size=0.4,
    random_state=42,
    stratify=y_temp,
)

print("Размеры выборок (обучение / тест / валидация):", X_train.shape, X_test.shape, X_valid.shape)

# ---------------------------------------------------------------------------
# Вспомогательная функция для оценки качества классификаторов
# ---------------------------------------------------------------------------
def evaluate_classifier(model, x_train, y_train, x_test, y_test, title):
    """Обучает модель, оценивает Accuracy, Precision, Recall и ROC-AUC."""

    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    if hasattr(model, "predict_proba"):
        test_scores = model.predict_proba(x_test)[:, 1]
    else:
        test_scores = model.decision_function(x_test)

    metrics = {
        "train_accuracy": accuracy_score(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_precision": precision_score(y_test, test_pred, zero_division=0),
        "test_recall": recall_score(y_test, test_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, test_scores),
    }

    print(title)
    print(
        "  Точность на обучении: {:.4f}, на тесте: {:.4f}".format(
            metrics["train_accuracy"], metrics["test_accuracy"]
        )
    )
    print(
        "  Precision: {:.4f}, Recall: {:.4f}, ROC-AUC: {:.4f}".format(
            metrics["test_precision"], metrics["test_recall"], metrics["test_roc_auc"]
        )
    )
    return metrics


# ---------------------------------------------------------------------------
# Шаг 5. Поиск лучшего дерева решений
# ---------------------------------------------------------------------------
criterion_options = ["entropy", "gini"]
alpha_options = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.2, 0.8]

best_tree = None
best_tree_params = None
best_tree_metrics = None

for criterion in criterion_options:
    for alpha in alpha_options:
        tree_model = DecisionTreeClassifier(
            criterion=criterion,
            ccp_alpha=alpha,
            random_state=42,
        )
        metrics = evaluate_classifier(
            tree_model,
            X_train,
            y_train,
            X_test,
            y_test,
            f"Дерево решений (criterion={criterion}, ccp_alpha={alpha})",
        )
        if (best_tree_metrics is None) or (
            metrics["test_roc_auc"] > best_tree_metrics["test_roc_auc"]
        ):
            best_tree = tree_model
            best_tree_params = {"criterion": criterion, "ccp_alpha": alpha}
            best_tree_metrics = metrics

print("Лучшее дерево по ROC-AUC на тестовой выборке:", best_tree_params)

# Дополнительное обучение лучшего дерева для последующих шагов
best_tree.fit(np.vstack([X_train, X_test]), np.concatenate([y_train, y_test]))

# ---------------------------------------------------------------------------
# Шаг 6. Поиск лучшего SVM-классификатора
# ---------------------------------------------------------------------------
svm_options = []

# Линейное ядро
for c_value in [0.5, 1.0, 2.0, 5.0]:
    svm_options.append(
        {
            "kernel": "linear",
            "C": c_value,
        }
    )

# Полиномиальное ядро
for c_value in [1.0, 2.0, 5.0]:
    for degree in [2, 3, 4]:
        for coef0 in [0.0, 0.5, 1.0]:
            svm_options.append(
                {
                    "kernel": "poly",
                    "C": c_value,
                    "degree": degree,
                    "coef0": coef0,
                    "gamma": "scale",
                }
            )

# Радиально-базисное ядро
for c_value in [0.5, 1.0, 2.0, 5.0]:
    for gamma in ["scale", "auto", 0.01, 0.1]:
        svm_options.append(
            {
                "kernel": "rbf",
                "C": c_value,
                "gamma": gamma,
            }
        )

# Сигмоидальное ядро
for c_value in [0.5, 1.0, 2.0, 5.0]:
    for gamma in ["scale", "auto", 0.01, 0.1]:
        for coef0 in [0.0, 0.5, 1.0]:
            svm_options.append(
                {
                    "kernel": "sigmoid",
                    "C": c_value,
                    "gamma": gamma,
                    "coef0": coef0,
                }
            )

best_svm = None
best_svm_params = None
best_svm_metrics = None

for params in svm_options:
    svm_model = SVC(probability=True, random_state=42, **params)
    metrics = evaluate_classifier(
        svm_model,
        X_train,
        y_train,
        X_test,
        y_test,
        "SVM (параметры: {})".format(params),
    )
    if (best_svm_metrics is None) or (
        metrics["test_roc_auc"] > best_svm_metrics["test_roc_auc"]
    ):
        best_svm = svm_model
        best_svm_params = params
        best_svm_metrics = metrics

print("Лучший SVM по ROC-AUC на тестовой выборке:", best_svm_params)

# Переобучаем лучший SVM на объединённых обучающей и тестовой выборках
best_svm.fit(np.vstack([X_train, X_test]), np.concatenate([y_train, y_test]))

# ---------------------------------------------------------------------------
# Шаг 7. Анализ с помощью PCA
# ---------------------------------------------------------------------------
# Выполняем PCA по обучающей выборке, чтобы объяснить 95% дисперсии
pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("PCA: число главных компонент, объясняющих 95% дисперсии:", pca.n_components_)

# Обучаем дерево и SVM с теми же лучшими параметрами, но на PCA-признаках
pca_tree = DecisionTreeClassifier(
    criterion=best_tree_params["criterion"],
    ccp_alpha=best_tree_params["ccp_alpha"],
    random_state=42,
)
print("Качество дерева после PCA:")
pca_tree_metrics = evaluate_classifier(
    pca_tree,
    X_train_pca,
    y_train,
    X_test_pca,
    y_test,
    "Дерево решений (PCA)",
)

pca_svm = SVC(probability=True, random_state=42, **best_svm_params)
print("Качество SVM после PCA:")
pca_svm_metrics = evaluate_classifier(
    pca_svm,
    X_train_pca,
    y_train,
    X_test_pca,
    y_test,
    "SVM (PCA)",
)

# ---------------------------------------------------------------------------
# Шаг 8. Финальная оценка на валидационной выборке
# ---------------------------------------------------------------------------
X_valid_pca = pca.transform(X_valid)

valid_tree_pred = best_tree.predict(X_valid)
valid_tree_proba = best_tree.predict_proba(X_valid)[:, 1]

tree_valid_metrics = {
    "accuracy": accuracy_score(y_valid, valid_tree_pred),
    "precision": precision_score(y_valid, valid_tree_pred, zero_division=0),
    "recall": recall_score(y_valid, valid_tree_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_valid, valid_tree_proba),
}

valid_svm_pred = best_svm.predict(X_valid)
valid_svm_proba = best_svm.predict_proba(X_valid)[:, 1]

svm_valid_metrics = {
    "accuracy": accuracy_score(y_valid, valid_svm_pred),
    "precision": precision_score(y_valid, valid_svm_pred, zero_division=0),
    "recall": recall_score(y_valid, valid_svm_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_valid, valid_svm_proba),
}

print("\nКачество лучших моделей на валидационной выборке:")
print("  Дерево решений:", tree_valid_metrics)
print("  SVM:", svm_valid_metrics)

if svm_valid_metrics["roc_auc"] > tree_valid_metrics["roc_auc"]:
    print(
        "SVM показывает более высокое качество по ROC-AUC на валидации,"
        " что полезно при необходимости минимизировать ошибки в определении типа вина."
    )
else:
    print(
        "Дерево решений выигрывает по ROC-AUC на валидации,"
        " поэтому его стоит предпочесть при текущем разделении данных."
    )

print(
    "Выбор между моделями зависит от требований:"
    " дерево решений проще интерпретировать,"
    " а SVM обычно даёт более высокое качество при сложных границах раздела."
)


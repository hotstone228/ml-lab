"""Лабораторная работа №3: классификация качества вина при помощи SVM."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import kagglehub


# ---------------------------------------------------------------------------
# Шаг 1. Загрузка данных с Kaggle
# ---------------------------------------------------------------------------
dataset_dir = Path(
    kagglehub.dataset_download("ehsanesmaeili/red-and-white-wine-quality-merged")
)
csv_path = dataset_dir / "wine_quality_merged.csv"

# ---------------------------------------------------------------------------
# Шаг 2. Создание DataFrame и задание корректных типов столбцов
# ---------------------------------------------------------------------------
df_raw = pd.read_csv(csv_path)
df_wine = df_raw.copy()

# Перечень признаков с вещественными значениями.
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

# Гарантируем корректные типы данных.
df_wine[float_columns] = df_wine[float_columns].astype("float64")
df_wine["type"] = df_wine["type"].astype("category")

print("Шаг 2: исходные данные загружены, форма:", df_wine.shape)
print("Доступные столбцы:", list(df_wine.columns))
print("Метка класса — столбец 'quality' (оценка качества вина).")

# ---------------------------------------------------------------------------
# Шаг 3. Подготовка признаков и стандартизация
# ---------------------------------------------------------------------------
# Для алгоритмов машинного обучения требуется числовое представление категориальных признаков.
# Кодируем тип вина (красное/белое) целочисленным значением и работаем с исходными признаками.
df_features = df_wine.drop(columns=["quality"]).copy()
df_features["type"] = df_features["type"].map({"red": 0, "white": 1}).astype("float64")

X_full = df_features.to_numpy(dtype=np.float64)
y_full = df_wine["quality"].to_numpy()

print(
    "Шаг 3: признаки подготовлены, стандартизация будет выполняться внутри конвейера",
    "по обучающим данным.",
)

# ---------------------------------------------------------------------------
# Шаг 4. Разделение на обучающую и тестовую выборки (60% / 40%)
# ---------------------------------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_full,
    y_full,
    test_size=0.4,
    random_state=42,
    stratify=y_full,
)
print(
    "Шаг 4: обучающая выборка:",
    X_train_raw.shape,
    "тестовая выборка:",
    X_test_raw.shape,
)

# ---------------------------------------------------------------------------
# Шаг 5. Обучение SVM и подбор гиперпараметров
# ---------------------------------------------------------------------------
n_features = X_train_raw.shape[1]
base_gamma = 1.0 / n_features

# Подбираем параметры kernel, gamma, degree, C при помощи GridSearchCV.
svm_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "svc",
            SVC(cache_size=2000),
        ),
    ]
)

param_grid = [
    {
        "svc__C": [np.float64(4.6415888336127775)],
        "svc__class_weight": ["balanced"],
        "svc__gamma": ["scale"],
        "svc__kernel": ["rbf"],
        "svc__probability": [True],
    },
]

scoring = {
    "bal_acc": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "roc_auc_ovr": "roc_auc_ovr",  # для многокласса тоже работает (One-vs-Rest)
}

cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
search = GridSearchCV(
    svm_pipeline,
    param_grid=param_grid,
    scoring=scoring,
    refit="bal_acc",
    n_jobs=-1,
)
search.fit(X_train_raw, y_train)
best_svm = search.best_estimator_

train_pred = best_svm.predict(X_train_raw)
test_pred = best_svm.predict(X_test_raw)
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)
train_balanced_accuracy = balanced_accuracy_score(y_train, train_pred)
test_balanced_accuracy = balanced_accuracy_score(y_test, test_pred)

print("Шаг 5: лучшие параметры SVM:", search.best_params_)
print("Точность (Accuracy) на обучающей выборке:", round(train_accuracy, 4))
print(
    "Сбалансированная точность на обучающей выборке:", round(train_balanced_accuracy, 4)
)
print("Точность (Accuracy) на тестовой выборке:", round(test_accuracy, 4))
print(
    "Сбалансированная точность на тестовой выборке:", round(test_balanced_accuracy, 4)
)

# ---------------------------------------------------------------------------
# Шаг 6. Применение PCA и повторное обучение SVM
# ---------------------------------------------------------------------------
pca_scaler = StandardScaler()
X_train_scaled = pca_scaler.fit_transform(X_train_raw)

pca_full = PCA(random_state=42)
pca_full.fit(X_train_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
required_components = 0.80

pca_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=42)),
        ("svc", SVC(cache_size=2000)),
    ]
)

pca_param_grid = [
    {
        "pca__n_components": [0.99],
        "svc__C": [np.float64(4.6415888336127775)],
        "svc__class_weight": ["balanced"],
        "svc__gamma": [np.float64(0.08333333333333333)],
        "svc__kernel": ["rbf"],
        "svc__probability": [True],
    }
]

search_pca = GridSearchCV(
    pca_pipeline,
    param_grid=pca_param_grid,
    scoring=scoring,
    refit="bal_acc",  # выбираем лучшую модель по balanced accuracy
    n_jobs=-1,
)
search_pca.fit(X_train_raw, y_train)
best_svm_pca = search_pca.best_estimator_

train_pred_pca = best_svm_pca.predict(X_train_raw)
test_pred_pca = best_svm_pca.predict(X_test_raw)
train_accuracy_pca = accuracy_score(y_train, train_pred_pca)
test_accuracy_pca = accuracy_score(y_test, test_pred_pca)
train_balanced_accuracy_pca = balanced_accuracy_score(y_train, train_pred_pca)
test_balanced_accuracy_pca = balanced_accuracy_score(y_test, test_pred_pca)

# достать реальный PCA объект
pca_step = best_svm_pca.named_steps["pca"]

print("Фактическое число главных компонент:", pca_step.n_components_)

print("Шаг 6: количество главных компонент:", required_components)
print("Лучшие параметры SVM после PCA:", search_pca.best_params_)
print("Точность (Accuracy) на обучающей выборке (PCA):", round(train_accuracy_pca, 4))
print(
    "Сбалансированная точность на обучающей выборке (PCA):",
    round(train_balanced_accuracy_pca, 4),
)
print("Точность (Accuracy) на тестовой выборке (PCA):", round(test_accuracy_pca, 4))
print(
    "Сбалансированная точность на тестовой выборке (PCA):",
    round(test_balanced_accuracy_pca, 4),
)


# ---------------------------------------------------------------------------
# Шаг 7. Построение предиктивного эллипса в пространстве двух главных компонент
# ---------------------------------------------------------------------------
def plot_predictive_ellipse(
    X_train_2d: np.ndarray,
    X_test_2d: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    """Строит диаграмму рассеяния и эллипс, охватывающий 95% распределения обучающих точек."""

    # Среднее и ковариационная матрица по обучающему набору.
    mean_vector = X_train_2d.mean(axis=0)
    covariance_matrix = np.cov(X_train_2d, rowvar=False)

    # Вычисляем собственные значения/векторы для получения параметров эллипса.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Коэффициент масштаба для 95%-ного доверительного интервала (хи-квадрат для 2 переменных).
    chi_square_scale = np.sqrt(5.991)
    width = 2 * chi_square_scale * np.sqrt(eigenvalues[0])
    height = 2 * chi_square_scale * np.sqrt(eigenvalues[1])
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_test_2d[:, 0],
        X_test_2d[:, 1],
        c=predictions,
        cmap="viridis",
        s=25,
        alpha=0.8,
        edgecolors="none",
    )

    ellipse = Ellipse(
        xy=mean_vector,
        width=width,
        height=height,
        angle=angle,
        edgecolor="red",
        facecolor="none",
        linewidth=2,
        label="Предиктивный эллипс (95%)",
    )
    ax.add_patch(ellipse)

    ax.set_title("Проекция на две главные компоненты и предиктивный эллипс")
    ax.set_xlabel("Главная компонента 1")
    ax.set_ylabel("Главная компонента 2")
    colorbar = plt.colorbar(scatter, ax=ax)
    colorbar.set_label("Предсказанный класс качества")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)


scaler_step = best_svm_pca.named_steps["scaler"]
pca_step = best_svm_pca.named_steps["pca"]

X_train_pca = pca_step.transform(scaler_step.transform(X_train_raw))
X_test_pca = pca_step.transform(scaler_step.transform(X_test_raw))

X_train_2d = X_train_pca[:, :2]
X_test_2d = X_test_pca[:, :2]
ellipse_output_path = Path("3/plots/lab3_predictive_ellipse.png")
plot_predictive_ellipse(X_train_2d, X_test_2d, test_pred_pca, ellipse_output_path)
print("Шаг 7: график с предиктивным эллипсом сохранён по пути:", ellipse_output_path)

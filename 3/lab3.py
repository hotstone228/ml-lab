"""Лабораторная работа №3: классификация качества вина при помощи SVM."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import kagglehub


# ---------------------------------------------------------------------------
# Шаг 1. Загрузка данных с Kaggle
# ---------------------------------------------------------------------------
# Скачиваем набор данных с помощью kagglehub (аналогично примеру из lab1).
dataset_dir = Path(kagglehub.dataset_download("ehsanesmaeili/red-and-white-wine-quality-merged"))
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
]

# Гарантируем корректные типы данных.
df_wine[float_columns] = df_wine[float_columns].astype("float64")
df_wine["quality"] = df_wine["quality"].astype("int64")
df_wine["type"] = df_wine["type"].astype("category")

print("Шаг 2: исходные данные загружены, форма:", df_wine.shape)
print("Доступные столбцы:", list(df_wine.columns))
print("Метка класса — столбец 'quality' (оценка качества вина).")

# ---------------------------------------------------------------------------
# Шаг 3. Стандартизация признаков
# ---------------------------------------------------------------------------
# Для алгоритмов машинного обучения требуется числовое представление категориальных признаков.
# Кодируем тип вина (красное/белое) целочисленным значением.
df_features = df_wine.drop(columns=["quality"]).copy()
df_features["type"] = df_features["type"].map({"red": 0, "white": 1}).astype("float64")

scaler = StandardScaler()
scaled_values = scaler.fit_transform(df_features)
df_scaled = pd.DataFrame(scaled_values, columns=df_features.columns)

print("Шаг 3: стандартизация завершена, форма матрицы признаков:", df_scaled.shape)

# ---------------------------------------------------------------------------
# Шаг 4. Разделение на обучающую и тестовую выборки (60% / 40%)
# ---------------------------------------------------------------------------
y = df_wine["quality"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(
    df_scaled,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y,
)
print("Шаг 4: обучающая выборка:", X_train.shape, "тестовая выборка:", X_test.shape)

# ---------------------------------------------------------------------------
# Шаг 5. Обучение SVM и подбор гиперпараметров
# ---------------------------------------------------------------------------
# Подбираем параметры kernel, gamma, coef0, degree, C при помощи GridSearchCV.
param_grid = [
    {"kernel": ["linear"], "C": [0.5, 1.0, 2.0, 5.0]},
    {
        "kernel": ["rbf"],
        "C": [1.0, 5.0, 10.0],
        "gamma": ["scale", 0.1, 0.01],
    },
    {
        "kernel": ["poly"],
        "C": [1.0, 5.0],
        "gamma": ["scale", 0.1],
        "degree": [2, 3],
        "coef0": [0.0, 0.5],
    },
    {
        "kernel": ["sigmoid"],
        "C": [1.0, 5.0],
        "gamma": ["scale", 0.1],
        "coef0": [0.0, 0.5],
    },
]

svm = SVC()
search = GridSearchCV(svm, param_grid=param_grid, cv=5, n_jobs=-1)
search.fit(X_train, y_train)
best_svm = search.best_estimator_

train_pred = best_svm.predict(X_train)
test_pred = best_svm.predict(X_test)
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print("Шаг 5: лучшие параметры SVM:", search.best_params_)
print("Точность на обучающей выборке:", round(train_accuracy, 4))
print("Точность на тестовой выборке:", round(test_accuracy, 4))

# ---------------------------------------------------------------------------
# Шаг 6. Применение PCA и повторное обучение SVM
# ---------------------------------------------------------------------------
pca_full = PCA(random_state=42)
pca_full.fit(X_train)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
required_components = int(np.searchsorted(cumulative_variance, 0.95) + 1)
# Гарантируем минимум два компонента для последующей визуализации эллипса.
required_components = max(2, min(required_components, X_train.shape[1]))

pca = PCA(n_components=required_components, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

search_pca = GridSearchCV(SVC(), param_grid=param_grid, cv=5, n_jobs=-1)
search_pca.fit(X_train_pca, y_train)
best_svm_pca = search_pca.best_estimator_

train_pred_pca = best_svm_pca.predict(X_train_pca)
test_pred_pca = best_svm_pca.predict(X_test_pca)
train_accuracy_pca = accuracy_score(y_train, train_pred_pca)
test_accuracy_pca = accuracy_score(y_test, test_pred_pca)

print("Шаг 6: количество главных компонент:", required_components)
print("Лучшие параметры SVM после PCA:", search_pca.best_params_)
print("Точность на обучающей выборке (PCA):", round(train_accuracy_pca, 4))
print("Точность на тестовой выборке (PCA):", round(test_accuracy_pca, 4))

# ---------------------------------------------------------------------------
# Шаг 7. Построение предиктивного эллипса в пространстве двух главных компонент
# ---------------------------------------------------------------------------
def plot_predictive_ellipse(X_train_2d: np.ndarray, X_test_2d: np.ndarray, predictions: np.ndarray, output_path: Path) -> None:
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


X_train_2d = X_train_pca[:, :2]
X_test_2d = X_test_pca[:, :2]
ellipse_output_path = Path("3/plots/lab3_predictive_ellipse.png")
plot_predictive_ellipse(X_train_2d, X_test_2d, test_pred_pca, ellipse_output_path)
print("Шаг 7: график с предиктивным эллипсом сохранён по пути:", ellipse_output_path)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import kagglehub

# ---------------------------------------------------------------------------
# Задание 1 — загрузка
# ---------------------------------------------------------------------------
"""На основе загруженного CSV-файла создайте Pandas DataFrame, 
гарантировав правильные типы данных, переменных (например, правильную загрузку дат). 
Назовите переменные lat, ing, desc, zip, title, accident_time, town, address, e соответственно."""
dataset_path = Path(kagglehub.dataset_download("mchirico/montcoalert"))
csv_path = dataset_path / "911.csv"
dtype_map = {"zip": "Int64", "e": "Int64"}
df_raw = pd.read_csv(csv_path, dtype=dtype_map, parse_dates=["timeStamp"])

rename_map = {
    "lat": "lat",
    "lng": "ing",
    "desc": "desc",
    "zip": "zip",
    "title": "title",
    "timeStamp": "accident_time",
    "twp": "town",
    "addr": "address",
    "e": "e",
}

df_task_1 = df_raw.rename(columns=rename_map)
print("Задание 1:", df_task_1.shape)

# ---------------------------------------------------------------------------
# Задание 2 — удаление лишних столбцов
# ---------------------------------------------------------------------------
"""Измените полученный в задании 1 Pandas DataFrame, исключив из исходного набора 
переменные desc, zip, address, e."""
df_task_2 = df_task_1.drop(columns=["desc", "zip", "address", "e"]).copy()
print("Задание 2:", df_task_2.shape)

# ---------------------------------------------------------------------------
# Задание 3 — сортировка данных
# ---------------------------------------------------------------------------
"""Измените полученный в задании 1 Pandas DataFrame, отсортировав набор, 
полученный в задании 2 в следующем порядке: town lat ing accident_time title."""
sort_columns = ["town", "lat", "ing", "accident_time", "title"]
df_task_3 = df_task_2.sort_values(by=sort_columns, ascending=True)
print("Задание 3:")
print(sort_columns)

# ---------------------------------------------------------------------------
# Задание 4 — частоты по городам
# ---------------------------------------------------------------------------
"""Выполните простейший количественный анализ по переменной town Pandas 
DataFrame, полученного в задании 3, 
отсортировав при этом результаты в порядке возрастания количества появлений городов среди наблюдений. 
Сохраните результаты в новый Pandas DataFrame."""
df_task_4 = (
    df_task_3["town"]
    .value_counts()  # Считаем, сколько раз встречается каждое значение.
    .reset_index(name="count")  # Преобразует Series в DataFrame.
    .sort_values(by="count", ascending=True)  # Сортирует по количеству
)
print("Задание 4:")
print(df_task_4)

# ---------------------------------------------------------------------------
# Задание 5 — четыре экстремальных города
# ---------------------------------------------------------------------------
"""Сформируйте новый Pandas DataFrame, в котором останутся только 4 названия городов: 
два наиболее часто встречающихся и два наименее часто встречающихся среди наблюдений исходного массива."""
df_task_5 = pd.concat([df_task_4.head(2), df_task_4.tail(2)])
print("Задание 5:")
print(df_task_5)

# ---------------------------------------------------------------------------
# Задание 6 — фильтрация по городам и добавление часа вызова
# ---------------------------------------------------------------------------
"""Сформируйте новый Pandas DataFrame, исключив из Pandas DataFrame, 
полученного в задании 3 все наблюдения, относящиеся к 4-м городам, 
полученным в задании 5, а также все наблюдения, где город не указан. 
Кроме того, добавьте в полученный набор новую переменную hour, 
содержащую время суток в часах, в которое произошёл инцидент. """
excluded_towns = df_task_5["town"].dropna().tolist()
mask_valid_town = df_task_3["town"].notna() & ~df_task_3["town"].isin(excluded_towns)
# mask_valid_town = булев фильтр: True только там, где город указан и не входит в исключённые.
df_task_6 = df_task_3.loc[mask_valid_town].copy()
# Берёт из df_task_3 только строки, где mask_valid_town == True
df_task_6["hour"] = df_task_6["accident_time"].dt.hour
print("Задание 6:")
print(df_task_6)

# ---------------------------------------------------------------------------
# Задание 7 — частоты по часам суток
# ---------------------------------------------------------------------------
"""Выполните простейший количественный анализ по переменной hour Pandas DataFrame, 
полученного в задании 6, отсортировав при этом результаты 
в порядке убывания количества появлений данного часа среди наблюдений. 
Сохраните результаты в новый Pandas DataFrame."""
df_task_7 = (
    df_task_6["hour"]
    .value_counts()
    .reset_index(name="count")
    .sort_values(by="count", ascending=False)
)
print("Задание 7:")
print(df_task_7)

# ---------------------------------------------------------------------------
# Задание 8 — нормализация счётчиков вызовов
# ---------------------------------------------------------------------------
"""Выполните нормализацию набора данных, полученного в пункте 7, по переменной 
count."""
df_task_8 = df_task_7.copy()
count_min = df_task_8["count"].min()
count_max = df_task_8["count"].max()
df_task_8["count_normalized"] = (df_task_8["count"] - count_min) / (
    count_max - count_min
)  # по формуле от дженни
print("Задание 8:")

# ---------------------------------------------------------------------------
# Задание 9 — графическое сравнение распределений
# ---------------------------------------------------------------------------
"""Постройте гистограмму и кривую распределения для переменной count. Сравните 
кривую распределения графически с кривой нормального распределения. Сделайте выводы."""
plots_dir = Path("1/plots")
plots_dir.mkdir(parents=True, exist_ok=True)
values = df_task_7["count"].to_numpy()
# Берём столбец count и превращаем в массив numpy для удобных численных операций.
x_range = np.linspace(values.min(), values.max(), 200)
# По этим X будет строиться теоретическая кривая нормального распределения.
mean = values.mean()  # среднее
std = values.std(ddof=0)  # стандартное отклонение

normal_curve = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(
    -((x_range - mean) ** 2) / (2 * std**2)
)
# Формула плотности нормального распределения

plt.figure(figsize=(10, 6))
plt.hist(
    values,
    bins=10,
    density=True,  # density=True нормирует её, чтобы площадь ≈ 1 (как у функции плотности).
    alpha=0.6,
    color="skyblue",
    edgecolor="black",
    label="Наблюдаемая плотность",
)
# голубые столбики — реальные данные (count по часам);
plt.plot(
    x_range, normal_curve, color="red", linewidth=2, label="Нормальное распределение"
)
# красная кривая — теоретическая нормальная кривая.
plt.title("Распределение количества вызовов по часам")
plt.xlabel("Количество вызовов")
plt.ylabel("Плотность")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plot_distribution_path = plots_dir / "task_09_count_distribution.png"
plt.savefig(plot_distribution_path)
plt.close()
print("Задание 9:", plot_distribution_path)

# ---------------------------------------------------------------------------
# Задание 10 — линейная регрессия количества вызовов от часа суток
# ---------------------------------------------------------------------------
"""Постройте график линейной регрессии для зависимости переменной count от 
переменной hour. Сделайте выводы об их взаимосвязи."""
hours = df_task_7["hour"].to_numpy()
counts = df_task_7["count"].to_numpy()

slope, intercept = np.polyfit(hours, counts, 1)
# для подбора прямой линии первого порядка (1),
# которая наилучшим образом аппроксимирует зависимость count = f(hour)
regression_line = slope * hours + intercept
# Вычисляем предсказанные значения по найденной модели — координаты линии регрессии.

plt.figure(figsize=(10, 6))
plt.scatter(hours, counts, color="navy", alpha=0.7, label="Наблюдения")
# Рисует точки (реальные данные) — зависимость количества вызовов от часа.
plt.plot(
    hours, regression_line, color="orange", linewidth=2, label="Линейная регрессия"
)
# Добавляет прямую линию линейной регрессии поверх точек.
plt.title("Зависимость количества вызовов от часа суток")
plt.xlabel("Час суток")
plt.ylabel("Количество вызовов")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plot_regression_path = plots_dir / "task_10_regression.png"
plt.savefig(plot_regression_path)
plt.close()

print("Задание 10: график линейной регрессии сохранён по пути", plot_regression_path)

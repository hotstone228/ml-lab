"""Полное решение лабораторной работы 1 по анализу данных 911.

Скрипт выполняет все шаги последовательно, сохраняя результаты каждого
задания в отдельных переменных. Это позволяет не перезаписывать
полученные ранее данные и облегчает проверку каждого этапа. Комментарии
и выводы оформлены на русском языке согласно требованиям методических
указаний.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import kagglehub  # type: ignore
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Модуль kagglehub не найден. Установите пакет командой 'pip install kagglehub'"
        " перед запуском скрипта."
    ) from error


# ---------------------------------------------------------------------------
# Подготовка исходных данных
# ---------------------------------------------------------------------------
# Загружаем свежую версию набора с Kaggle и читаем CSV в DataFrame.
dataset_path = Path(kagglehub.dataset_download("mchirico/montcoalert"))
csv_path = dataset_path / "911.csv"

# Столбцы с индексами содержат пропуски, поэтому используем тип Int64 с поддержкой NA.
dtype_map = {"zip": "Int64", "e": "Int64"}

# Парсим временную метку сразу как datetime, чтобы удобнее было извлекать часы вызовов.
df_raw = pd.read_csv(csv_path, dtype=dtype_map, parse_dates=["timeStamp"])

# Переименовываем столбцы в соответствии с формулировкой заданий лабораторной работы.
rename_map = {
    "lat": "lat",
    "lng": "ing",  # намеренно оставляем опечатку из условия
    "desc": "desc",
    "zip": "zip",
    "title": "title",
    "timeStamp": "accident_time",
    "twp": "town",
    "addr": "address",
    "e": "e",
}

# Оставляем только необходимые столбцы и задаём единый строковый тип для текстовых полей.
df_task_1 = df_raw[list(rename_map.keys())].rename(columns=rename_map)
for column in ["desc", "title", "town", "address"]:
    df_task_1[column] = df_task_1[column].astype("string")

print("Задание 1: исходный датафрейм сформирован. Размер:", df_task_1.shape)

# ---------------------------------------------------------------------------
# Задание 2 — удаление лишних столбцов
# ---------------------------------------------------------------------------
df_task_2 = df_task_1.drop(columns=["desc", "zip", "address", "e"]).copy()
print("Задание 2: удалены столбцы desc, zip, address, e. Новый размер:", df_task_2.shape)

# ---------------------------------------------------------------------------
# Задание 3 — сортировка данных
# ---------------------------------------------------------------------------
sort_columns = ["town", "lat", "ing", "accident_time", "title"]
df_task_3 = df_task_2.sort_values(by=sort_columns, ascending=True).reset_index(drop=True)
print("Задание 3: данные отсортированы по столбцам", sort_columns)

# ---------------------------------------------------------------------------
# Задание 4 — частоты по городам
# ---------------------------------------------------------------------------
df_task_4 = (
    df_task_3["town"]
    .value_counts(dropna=False)
    .rename_axis("town")
    .reset_index(name="count")
    .sort_values(by="count", ascending=True)
    .reset_index(drop=True)
)
print("Задание 4: рассчитаны количества вызовов по городам. Всего городов:", len(df_task_4))

# ---------------------------------------------------------------------------
# Задание 5 — четыре экстремальных города
# ---------------------------------------------------------------------------
# Используем объединение head и tail; если набор городов пересекается, удаляем дубликаты.
df_task_5 = (
    pd.concat([df_task_4.head(2), df_task_4.tail(2)], ignore_index=True)
    .drop_duplicates(subset=["town"], keep="first")
    .reset_index(drop=True)
)
print("Задание 5: выбраны четыре города (два частых и два редких):")
print(df_task_5)

# ---------------------------------------------------------------------------
# Задание 6 — фильтрация по городам и добавление часа вызова
# ---------------------------------------------------------------------------
excluded_towns = df_task_5["town"].dropna().tolist()
mask_valid_town = df_task_3["town"].notna() & ~df_task_3["town"].isin(excluded_towns)
df_task_6 = df_task_3.loc[mask_valid_town].copy()
df_task_6["hour"] = df_task_6["accident_time"].dt.hour
print("Задание 6: сформирован датафрейм без выбранных городов. Размер:", df_task_6.shape)

# ---------------------------------------------------------------------------
# Задание 7 — частоты по часам суток
# ---------------------------------------------------------------------------
df_task_7 = (
    df_task_6["hour"]
    .value_counts()
    .rename_axis("hour")
    .reset_index(name="count")
    .sort_values(by="count", ascending=False)
    .reset_index(drop=True)
)
print("Задание 7: рассчитаны частоты по часам суток. Всего часов:", len(df_task_7))

# ---------------------------------------------------------------------------
# Задание 8 — нормализация счётчиков вызовов
# ---------------------------------------------------------------------------
df_task_8 = df_task_7.copy()
count_min = df_task_8["count"].min()
count_max = df_task_8["count"].max()
if count_max == count_min:
    df_task_8["count_normalized"] = 0.0
else:
    df_task_8["count_normalized"] = (df_task_8["count"] - count_min) / (count_max - count_min)
print("Задание 8: выполнена min-max нормализация счётчиков вызовов.")

# ---------------------------------------------------------------------------
# Задание 9 — графическое сравнение распределений
# ---------------------------------------------------------------------------
plots_dir = Path(__file__).resolve().parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

values = df_task_7["count"].to_numpy(dtype=float)
if len(values) == 0:
    raise ValueError("Не найдено значений для построения распределения по часам.")

x_range = np.linspace(values.min(), values.max(), 200)
mean = values.mean()
std = values.std(ddof=0)
if std == 0:
    normal_curve = np.zeros_like(x_range)
else:
    normal_curve = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x_range - mean) ** 2) / (2 * std**2))

plt.figure(figsize=(10, 6))
plt.hist(values, bins=10, density=True, alpha=0.6, color="skyblue", edgecolor="black", label="Наблюдаемая плотность")
plt.plot(x_range, normal_curve, color="red", linewidth=2, label="Нормальное распределение")
plt.title("Распределение количества вызовов по часам")
plt.xlabel("Количество вызовов")
plt.ylabel("Плотность")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plot_distribution_path = plots_dir / "task_09_count_distribution.png"
plt.savefig(plot_distribution_path)
plt.close()
print("Задание 9: сохранена гистограмма распределения по пути", plot_distribution_path)

# ---------------------------------------------------------------------------
# Задание 10 — линейная регрессия количества вызовов от часа суток
# ---------------------------------------------------------------------------
hours = df_task_7["hour"].to_numpy(dtype=float)
counts = df_task_7["count"].to_numpy(dtype=float)

if len(hours) > 1:
    slope, intercept = np.polyfit(hours, counts, 1)
    regression_line = slope * hours + intercept
else:
    slope = 0.0
    intercept = counts[0] if len(counts) == 1 else 0.0
    regression_line = np.full_like(hours, intercept)

plt.figure(figsize=(10, 6))
plt.scatter(hours, counts, color="navy", alpha=0.7, label="Наблюдения")
plt.plot(hours, regression_line, color="orange", linewidth=2, label="Линейная регрессия")
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
print("Лабораторная работа 1 выполнена успешно.")

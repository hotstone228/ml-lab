"""Простое решение для второй лабораторной работы."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

# KaggleHub использую так же, как в первой лабораторной работе.
import kagglehub


DATASET_SLUG = "freecodecamp/2016-new-coder-survey-"
CSV_FILENAME = "2016-FCC-New-Coders-Survey-Data.csv"


def download_dataset() -> Path:
    """Скачиваю датасет через kagglehub и возвращаю путь к CSV."""
    dataset_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    csv_path = dataset_path / CSV_FILENAME
    return csv_path


def load_data(csv_path: Path) -> pd.DataFrame:
    """Читаю только нужные столбцы и задаю типы."""
    usecols = [
        "EmploymentField",
        "EmploymentStatus",
        "Gender",
        "JobPref",
        "JobWherePref",
        "MaritalStatus",
        "Income",
    ]
    dtype_map = {col: "string" for col in usecols}
    dtype_map["Income"] = "float64"
    df = pd.read_csv(csv_path, usecols=usecols, dtype=dtype_map, low_memory=False)
    for col in usecols:
        if col == "Income":
            continue
        df[col] = df[col].astype("string")
    return df


def filter_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляю только male/female и наблюдения без пропусков."""
    mask_gender = df["Gender"].isin(["male", "female"])
    df_valid = df.loc[mask_gender].dropna()
    return df_valid


def expected_from_counts(table: pd.DataFrame) -> np.ndarray:
    """Вычисляет ожидаемые частоты из правил независимости."""
    counts = table.to_numpy(dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)
    col_sums = counts.sum(axis=0, keepdims=True)
    total = counts.sum()
    return row_sums @ col_sums / total


def analyze_pair(df: pd.DataFrame, col_a: str, col_b: str) -> None:
    """Строю таблицы и печатаю выбранный тест."""
    pair_df = df[[col_a, col_b]].dropna()
    observed = pd.crosstab(pair_df[col_a], pair_df[col_b])
    total = observed.values.sum()
    print("\n================================================================")
    print(f"Пара переменных: {col_a} — {col_b}")
    print("Таблица сопряжённости:")
    print(observed)

    rows, cols = observed.shape
    if rows == 2 and cols == 2:
        expected = expected_from_counts(observed)
        if (expected < 5).any():
            print("Использую точный критерий Фишера (малые ожидаемые частоты).")
            stat, p_value = fisher_exact(observed)
            dof = 1
        elif total < 40:
            print("Использую хи-квадрат Пирсона с поправкой Йейтса (малый объём).")
            stat, p_value, dof, expected = chi2_contingency(observed, correction=True)
        else:
            print("Использую хи-квадрат Пирсона (достаточный объём).")
            stat, p_value, dof, expected = chi2_contingency(observed, correction=False)
    else:
        stat, p_value, dof, expected = chi2_contingency(observed, correction=False)
        print("Использую хи-квадрат Пирсона (таблица больше 2x2).")

    expected_df = pd.DataFrame(expected, index=observed.index, columns=observed.columns)
    print("Таблица ожидаемых значений:")
    print(expected_df.round(2))
    print(f"Статистика: {stat:.4f}, степени свободы: {dof}, p-value: {p_value:.4f}")


def main() -> None:
    """Главная функция: скачивание, подготовка и анализ."""
    csv_path = download_dataset()
    print(f"CSV-файл: {csv_path}")
    df_raw = load_data(csv_path)
    print("Размер исходного набора:", df_raw.shape)
    df_filtered = filter_sample(df_raw)
    print("Размер очищенного набора:", df_filtered.shape)

    pairs = [
        ("Gender", "JobPref"),
        ("Gender", "JobWherePref"),
        ("JobWherePref", "MaritalStatus"),
        ("EmploymentField", "JobWherePref"),
        ("EmploymentStatus", "JobWherePref"),
    ]

    for col_a, col_b in pairs:
        analyze_pair(df_filtered, col_a, col_b)


if __name__ == "__main__":
    main()

from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------
PLOTS_DIR = Path("2/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_currency(name: str) -> pd.DataFrame:
    """Загружаем Excel-файл и приводим даты к типу datetime."""
    file_path = Path("2") / f"{name}.xlsx"
    df = pd.read_excel(file_path)
    df = df.copy()
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.dropna(subset=["data"]).sort_values("data")
    df["curs"] = df["curs"].astype(float)
    return df


def make_pp_plot(values: pd.Series, title: str, suffix: str) -> Path:
    """Строим P-P график для сравнения эмпирической и теоретической нормальных CDF."""
    sorted_values = np.sort(values.to_numpy())
    mean = sorted_values.mean()
    std = sorted_values.std(ddof=0)
    if std == 0:
        std = 1e-9  # Защита от деления на ноль при константном ряду.
    # Вычисляем квантильные уровни для эмпирических данных.
    empirical_probs = np.linspace(1 / (len(sorted_values) + 1), len(sorted_values) / (len(sorted_values) + 1), len(sorted_values))
    # Теоретические вероятности нормального распределения на тех же значениях.
    standardized = (sorted_values - mean) / std
    theoretical_probs = stats.norm.cdf(standardized)

    plt.figure(figsize=(8, 6))
    plt.scatter(theoretical_probs, empirical_probs, color="teal", s=20)
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.title(f"P-P график: {title} ({suffix})")
    plt.xlabel("Теоретическая CDF")
    plt.ylabel("Эмпирическая CDF")
    plt.grid(alpha=0.3)
    plot_path = PLOTS_DIR / f"pp_{title}_{suffix}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def make_qq_plot(values: pd.Series, title: str, suffix: str) -> Path:
    """Строим Q-Q график с помощью scipy.stats.probplot."""
    plt.figure(figsize=(8, 6))
    stats.probplot(values, dist="norm", plot=plt)
    plt.title(f"Q-Q график: {title} ({suffix})")
    plot_path = PLOTS_DIR / f"qq_{title}_{suffix}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def plot_hist_with_statistics(values: pd.Series, title: str, suffix: str) -> tuple[Path, float, float, float]:
    """Строим гистограмму и отмечаем моду, медиану и среднее."""
    mean = values.mean()
    median = values.median()
    mode = values.mode().iloc[0]

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=10, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axvline(mode, color="purple", linestyle="-", linewidth=2, label=f"Мода: {mode:.4f}")
    plt.axvline(median, color="green", linestyle="--", linewidth=2, label=f"Медиана: {median:.4f}")
    plt.axvline(mean, color="red", linestyle=":", linewidth=2, label=f"Среднее: {mean:.4f}")
    plt.title(f"Гистограмма курса: {title} ({suffix})")
    plt.xlabel("Курс")
    plt.ylabel("Частота")
    plt.legend()
    plt.grid(alpha=0.3)
    plot_path = PLOTS_DIR / f"hist_{title}_{suffix}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path, mean, median, mode


def run_normality_tests(values: pd.Series) -> dict:
    """Запускаем три классических критерия нормальности."""
    result = {}
    shapiro_stat, shapiro_p = stats.shapiro(values)
    result["shapiro"] = (shapiro_stat, shapiro_p)

    anderson_res = stats.anderson(values, dist="norm")
    result["anderson"] = (anderson_res.statistic, list(zip(anderson_res.significance_level, anderson_res.critical_values)))

    mean = values.mean()
    std = values.std(ddof=0)
    if std == 0:
        std = 1e-9
    ks_stat, ks_p = stats.kstest(values, "norm", args=(mean, std))
    result["kolmogorov"] = (ks_stat, ks_p)
    return result


def remove_outliers_iqr(values: pd.Series) -> pd.Series:
    """Убираем выбросы с помощью межквартильного размаха."""
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = values[(values >= lower) & (values <= upper)]
    return filtered


def analyze_currency(df: pd.DataFrame, name: str) -> pd.Series:
    """Полный цикл анализа одной валюты."""
    values = df.set_index("data")["curs"]

    print(f"\n=== Анализ валюты {name} (исходные данные) ===")
    hist_path, mean, median, mode = plot_hist_with_statistics(values, name, "raw")
    print(f"Гистограмма сохранена: {hist_path}")
    print(f"Среднее: {mean:.4f}, медиана: {median:.4f}, мода: {mode:.4f}")

    pp_path = make_pp_plot(values, name, "raw")
    qq_path = make_qq_plot(values, name, "raw")
    print(f"P-P график: {pp_path}")
    print(f"Q-Q график: {qq_path}")

    tests = run_normality_tests(values)
    shapiro_stat, shapiro_p = tests["shapiro"]
    print(f"Критерий Шапиро-Уилка: статистика={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
    anderson_stat, anderson_table = tests["anderson"]
    print(f"Критерий Андерсона-Дарлинга: статистика={anderson_stat:.4f}")
    for level, critical in anderson_table:
        print(f"  Уровень значимости {level:.1f}%: критическое значение {critical:.4f}")
    ks_stat, ks_p = tests["kolmogorov"]
    print(f"Критерий Колмогорова-Смирнова: статистика={ks_stat:.4f}, p-value={ks_p:.4f}")

    cleaned_values = remove_outliers_iqr(values)
    removed = len(values) - len(cleaned_values)
    print(f"Удалено выбросов: {removed}")

    hist_path_clean, mean_c, median_c, mode_c = plot_hist_with_statistics(cleaned_values, name, "clean")
    print(f"Гистограмма (без выбросов) сохранена: {hist_path_clean}")
    print(f"Среднее: {mean_c:.4f}, медиана: {median_c:.4f}, мода: {mode_c:.4f}")

    pp_path_clean = make_pp_plot(cleaned_values, name, "clean")
    qq_path_clean = make_qq_plot(cleaned_values, name, "clean")
    print(f"P-P график (без выбросов): {pp_path_clean}")
    print(f"Q-Q график (без выбросов): {qq_path_clean}")

    tests_clean = run_normality_tests(cleaned_values)
    shapiro_stat, shapiro_p = tests_clean["shapiro"]
    print(f"Критерий Шапиро-Уилка (без выбросов): статистика={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
    anderson_stat, anderson_table = tests_clean["anderson"]
    print(f"Критерий Андерсона-Дарлинга (без выбросов): статистика={anderson_stat:.4f}")
    for level, critical in anderson_table:
        print(f"  Уровень значимости {level:.1f}%: критическое значение {critical:.4f}")
    ks_stat, ks_p = tests_clean["kolmogorov"]
    print(f"Критерий Колмогорова-Смирнова (без выбросов): статистика={ks_stat:.4f}, p-value={ks_p:.4f}")

    return cleaned_values


def analyze_correlations(data: pd.DataFrame) -> None:
    """Исследуем пары валют и сохраняем графики корреляции."""
    for left, right in combinations(data.columns, 2):
        plt.figure(figsize=(8, 6))
        plt.scatter(data[left], data[right], color="darkorange", edgecolor="black", alpha=0.7)
        plt.title(f"Диаграмма рассеяния: {left} vs {right}")
        plt.xlabel(left)
        plt.ylabel(right)
        plt.grid(alpha=0.3)
        path = PLOTS_DIR / f"scatter_{left}_{right}.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"График корреляции {left}-{right}: {path}")

        pearson_r, pearson_p = stats.pearsonr(data[left], data[right])
        spearman_r, spearman_p = stats.spearmanr(data[left], data[right])
        kendall_tau, kendall_p = stats.kendalltau(data[left], data[right])
        print(
            f"Коэффициенты корреляции для пары {left}-{right}:\n"
            f"  Пирсон: r={pearson_r:.4f}, p-value={pearson_p:.4f}\n"
            f"  Спирмен: r={spearman_r:.4f}, p-value={spearman_p:.4f}\n"
            f"  Кендалл: tau={kendall_tau:.4f}, p-value={kendall_p:.4f}"
        )


def main() -> None:
    """Основная функция лабораторной работы."""
    global USD, CAD, EUR
    USD = load_currency("USD")
    CAD = load_currency("CAD")
    EUR = load_currency("EUR")
    print("Данные загружены: USD, CAD, EUR")

    cleaned_series = {}
    for name, df in zip(["USD", "CAD", "EUR"], [USD, CAD, EUR]):
        cleaned_series[name] = analyze_currency(df, name)

    combined = pd.DataFrame({name: series for name, series in cleaned_series.items()})
    combined = combined.dropna()
    print("\n=== Корреляционный анализ (после удаления выбросов) ===")
    analyze_correlations(combined)


if __name__ == "__main__":
    main()

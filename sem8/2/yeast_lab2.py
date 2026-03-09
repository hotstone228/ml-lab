import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# -----------------------------------------------------------------------------
# 1. Загрузка данных из предыдущей лабораторной работы
# -----------------------------------------------------------------------------
# файл в формате ARFF-подобном: строки с "@" содержат метаинформацию, их
# нужно пропустить. Оставшиеся строки — таблица, разделённая запятыми.
cols = ["Mcg", "Gvh", "Alm", "Mit", "Erl", "Pox", "Vac", "Nuc", "Class"]
df = pd.read_csv("1\\yeast1.dat", comment="@", header=None, names=cols)

# класс-метка хранится строкой, сделаем числа 0/1
# positive -> 1, negative -> 0

df["Class"] = df["Class"].str.strip().map({"positive": 1, "negative": 0})

# -----------------------------------------------------------------------------
# 2. Разбиение на обучающую и тестовую выборки (60/40)
# -----------------------------------------------------------------------------
# перед этим стандартизируем признаки: многие методы бустинга работают лучше,
# когда все признаки находятся в одной шкале.
features = cols[:-1]
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

X = df[features].values
y = df["Class"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
print(f"train {len(X_train)}  test {len(X_test)}")

# вспомогательная функция для печати метрик


def report(clf, name):
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    print(f"\n{name}")
    print(" accuracy ", accuracy_score(y_test, preds))
    print(" precision", precision_score(y_test, preds, average="macro"))
    print(" recall   ", recall_score(y_test, preds, average="macro"))
    print(" roc auc  ", roc_auc_score(y_test, probs, average="macro"))


# -----------------------------------------------------------------------------
# 3. Классическое задание: два алгоритма бустинга на основе "леса"
# -----------------------------------------------------------------------------
# AdaBoost использует простые деревья (stumps) как базовые классификаторы.
ada = AdaBoostClassifier(random_state=42)
report(ada, "AdaBoost")

# градиентный бустинг, реализованный в sklearn
gb = GradientBoostingClassifier(random_state=42)
report(gb, "GradientBoosting")

# -----------------------------------------------------------------------------
# 4. Творческая часть: два любых алгоритма бустинга. Первый обязательно
#    на основе градиентного спуска, второй любой другой.
# -----------------------------------------------------------------------------
# в качестве первого возьмём более новый класс из sklearn, который
# реализован через оптимизацию по градиенту и работает быстрее на больших
# данных.
hgb = HistGradientBoostingClassifier(random_state=42)
report(hgb, "HistGradientBoosting (градиентный спуск)")

# второй выбранный бустинг — возьмём CatBoost, который считается простым в
# использовании и даёт хорошую производительность на табличных данных.
# хотя у нас все признаки числовые, библиотека всё равно работает нормально.
cb = CatBoostClassifier(verbose=False, random_state=0)
report(cb, "CatBoost (альтернативный бустинг)")


# Результаты
"""train 890  test 594

AdaBoost
 accuracy  0.7609427609427609
 precision 0.7116228070175439
 recall    0.6595392924060399
 roc auc   0.7858205665160366

GradientBoosting
 accuracy  0.7744107744107744
 precision 0.7274172615184944
 recall    0.6931279620853081
 roc auc   0.7915449685881185

HistGradientBoosting (градиентный спуск)
 accuracy  0.7626262626262627
 precision 0.7099607709363807
 recall    0.6865562658437121
 roc auc   0.7844428524192659

CatBoost (альтернативный бустинг)
 accuracy  0.7777777777777778
 precision 0.7349004804392587
 recall    0.6886090598479003
 roc auc   0.8097376832359748"""

# CatBoost лучше по ROC AUC

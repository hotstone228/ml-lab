import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# 1. Загрузка и первичный осмотр данных
# Данные поставляются в файле, похожем на ARFF (широкий формат, с описанием атрибутов в начале).
# Строки, начинающиеся с символа '@', содержат метаинформацию и не являются частью таблицы.
# Поэтому мы передаём параметр comment='@' в pandas, чтобы такие строки игнорировались.
# Остальные строки -- просто числа, разделённые запятыми, их и считываем как CSV.
file_path = "1\\yeast1.dat"

# список названий колонок. Они идут в том же порядке, что и в файле.
# Количество признаков и их названия указаны в описании набора данных.
columns = ["Mcg", "Gvh", "Alm", "Mit", "Erl", "Pox", "Vac", "Nuc", "Class"]

df = pd.read_csv(file_path, comment="@", header=None, names=columns)

print("Размер данных:", df.shape)
print(df.head())

# После загрузки колонка с метками может содержать пробелы вокруг слов,
# это легко исправить с помощью str.strip().
# Затем мы переводим текстовые метки в числа: положительный класс = 1, отрицательный = 0.
# Это упрощает работу алгоритмов sklearn, которые ожидают числовые метки.
df["Class"] = df["Class"].str.strip().map({"positive": 1, "negative": 0})

# 2. Убедимся, что каждый столбец имеет подходящий тип данных.
# Признаки измеряются числами с плавающей точкой, а метка - целое число.
# Это важно, потому что sklearn не умеет работать с объектами или строками.
for col in columns[:-1]:
    df[col] = df[col].astype(float)
df["Class"] = df["Class"].astype(int)

print("\nТипы данных:")
print(df.dtypes)

# 3. Приведение признаков к единой шкале (стандартизация).
# Многие алгоритмы чувствительны к масштабу признаков, поэтому вычитаем среднее и
# делим на стандартное отклонение, получая нулевое среднее и единичную дисперсию.
scaler = StandardScaler()
features = columns[:-1]
X_scaled = scaler.fit_transform(df[features])
df_scaled = pd.DataFrame(X_scaled, columns=features)
df_scaled["Class"] = df["Class"]

# 4. Разбиение выборки на части для обучения и проверки.
# Мы хотим 50% примеров на обучение, 30% на тестовую проверку гиперпараметров
# и 20% на финальную валидацию. Это достигается последовательным разбиением:
# сперва делим пополам, потом одну из половин ещё раз (40%/60%).
# random_state фиксирует случайность для воспроизводимости.
train, temp = train_test_split(df_scaled, test_size=0.5, random_state=42)
test, val = train_test_split(temp, test_size=0.4, random_state=42)

X_train = train[features].values
y_train = train["Class"].values
X_test = test[features].values
y_test = test["Class"].values
X_val = val[features].values
y_val = val["Class"].values

print("\nРазбиение: train", len(train), "test", len(test), "val", len(val))

# Вспомогательная функция, которая обучает классификатор, делает предсказания
# и вычисляет основные метрики: accuracy, precision, recall и ROC-AUC.
# Параметр description служит для вывода информации о текущей модели.


def evaluate(clf, Xtr, ytr, Xte, yte, description):
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xte)
    probs = clf.predict_proba(Xte)[:, 1]
    acc = accuracy_score(yte, preds)
    prec = precision_score(yte, preds, average="macro")
    rec = recall_score(yte, preds, average="macro")
    roc = roc_auc_score(yte, probs, average="macro")
    print(
        f"{description}: acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}, roc_auc={roc:.3f}"
    )
    return roc


# 5. Поиск оптимального дерева решений.
# Будем перебирать различные значения параметра ccp_alpha (для обрезки дерева)
# и критерий качества (энтропия или индекс Джини). Для каждой комбинации
# оцениваем метрику ROC-AUC на тестовом наборе и запоминаем лучшую.

print("\nНастройка DecisionTreeClassifier")
best_dt = None
best_dt_score = -1

alphas = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.2, 0.8]
criteria = ["entropy", "gini"]

for alpha in alphas:
    for crit in criteria:
        desc = f"DT alpha={alpha} crit={crit}"
        clf = DecisionTreeClassifier(ccp_alpha=alpha, criterion=crit, random_state=42)
        roc = evaluate(clf, X_train, y_train, X_test, y_test, desc)
        if roc > best_dt_score:
            best_dt_score = roc
            best_dt = (alpha, crit, clf)

print("\nЛучшие параметры DT:", best_dt[0], best_dt[1], "ROC-AUC", best_dt_score)

# 6. Настройка SVM-классификатора.
# Параметры kernel, C, gamma, степень полинома и сдвиг coef0 влияют на гибкость
# и форму разделяющей поверхности. Перебираем возможные комбинации
# и выбираем ту, что даёт наилучшую ROC-AUC на тестовом наборе.
best_svm = None
best_svm_score = -1

kernels = ["linear", "poly", "rbf", "sigmoid"]
C_values = [0.1, 1, 10]
gammas = ["scale", "auto"]
degrees = [2, 3]
coef0s = [0.0, 1.0]

for kernel in kernels:
    for C in C_values:
        for gamma in gammas:
            if kernel == "poly":
                for degree in degrees:
                    for coef0 in coef0s:
                        desc = f"SVM ker={kernel} C={C} gamma={gamma} deg={degree} coef0={coef0}"
                        clf = SVC(
                            kernel=kernel,
                            C=C,
                            gamma=gamma,
                            degree=degree,
                            coef0=coef0,
                            probability=True,
                            random_state=42,
                        )
                        roc = evaluate(clf, X_train, y_train, X_test, y_test, desc)
                        if roc > best_svm_score:
                            best_svm_score = roc
                            best_svm = (kernel, C, gamma, degree, coef0)
            else:
                # degree (степень) и coef0 (сдвиг) релевантны только при использовании
                # полиномиального или сигмоидального ядра, поэтому остальные ядра их не задают.
                desc = f"SVM ker={kernel} C={C} gamma={gamma}"
                clf = SVC(
                    kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42
                )
                roc = evaluate(clf, X_train, y_train, X_test, y_test, desc)
                if roc > best_svm_score:
                    best_svm_score = roc
                    best_svm = (kernel, C, gamma)

print("\nЛучшие параметры SVM:", best_svm, "ROC-AUC", best_svm_score)

# 7. Поиск хороших параметров для случайного леса.
# Критерий разбиения (джини/энтропия), количество деревьев и число признаков,
# выбираемых в каждом узле, сильно влияют на производительность.
# Перебираем варианты и сохраняем лучшую ROC-AUC.
print("\nНастройка RandomForestClassifier")
best_rf = None
best_rf_score = -1

crit = ["entropy", "gini"]
n_estimators = [10, 50, 100]
max_feats = [None, "sqrt", "log2", 4]

for c in crit:
    for n in n_estimators:
        for mf in max_feats:
            desc = f"RF crit={c} trees={n} max_features={mf}"
            clf = RandomForestClassifier(
                criterion=c, n_estimators=n, max_features=mf, random_state=42
            )
            roc = evaluate(clf, X_train, y_train, X_test, y_test, desc)
            if roc > best_rf_score:
                best_rf_score = roc
                best_rf = (c, n, mf)

print("\nЛучшие параметры RF:", best_rf, "ROC-AUC", best_rf_score)

# 8. Обогащение признаков и повтор тех же шагов с учётом стратификации.
# Обогащение означает создание полиномиальных комбинаций признаков второй степени,
# что даёт новые признаки, возможно улучшающие модель.
# Стратификация при разбиении гарантирует, что доля положительных/отрицательных
# примеров одинакова во всех подвыборках, что важно для несбалансированного
# набора данных.
print("\nОбогащение данных полиномиальными признаками (степень 2)")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[features])
poly_features = poly.get_feature_names_out(features)
df_enriched = pd.DataFrame(X_poly, columns=poly_features)
df_enriched["Class"] = df["Class"]

# После генерации новых признаков снова масштабируем их
# (полиномиальные признаки обычно имеют другой масштаб).
scaler2 = StandardScaler()
X_poly_scaled = scaler2.fit_transform(df_enriched[poly_features])
df_enriched[poly_features] = X_poly_scaled

# Разбиваем обогащённый набор на train/test/val так же, как раньше,
# но передаём аргумент stratify, чтобы классы были равномерно распределены.
train_e, temp_e = train_test_split(
    df_enriched, test_size=0.5, random_state=42, stratify=df_enriched["Class"]
)
test_e, val_e = train_test_split(
    temp_e, test_size=0.4, random_state=42, stratify=temp_e["Class"]
)

print("Обогащённое разбиение:", len(train_e), len(test_e), len(val_e))

X_tr_e = train_e[poly_features].values
y_tr_e = train_e["Class"].values
X_te_e = test_e[poly_features].values
y_te_e = test_e["Class"].values
X_val_e = val_e[poly_features].values
y_val_e = val_e["Class"].values

# Обычно мы повторяем предыдущие циклы перебора параметров
# с теми же значениями, но теперь на обогащённых данных.
# Для экономии места на экране вывод сокращён, но по сути выполняются те же операции.

print("\n-- Дерево решений на обогащённых данных")
best_dt_e_score = -1
best_dt_e = None
for alpha in alphas:
    for crit in criteria:
        clf = DecisionTreeClassifier(ccp_alpha=alpha, criterion=crit, random_state=42)
        roc = evaluate(
            clf, X_tr_e, y_tr_e, X_te_e, y_te_e, f"DT_e alpha={alpha} crit={crit}"
        )
        if roc > best_dt_e_score:
            best_dt_e_score = roc
            best_dt_e = (alpha, crit)
print("Лучшее обогащённое DT:", best_dt_e, best_dt_e_score)

print("\n-- SVM на обогащённых данных")
best_svm_e_score = -1
best_svm_e = None
for kernel in kernels:
    for C in C_values:
        for gamma in gammas:
            if kernel == "poly":
                for degree in degrees:
                    for coef0 in coef0s:
                        clf = SVC(
                            kernel=kernel,
                            C=C,
                            gamma=gamma,
                            degree=degree,
                            coef0=coef0,
                            probability=True,
                            random_state=42,
                        )
                        roc = evaluate(
                            clf,
                            X_tr_e,
                            y_tr_e,
                            X_te_e,
                            y_te_e,
                            f"SVM_e ker={kernel} C={C} gamma={gamma} deg={degree} coef0={coef0}",
                        )
                        if roc > best_svm_e_score:
                            best_svm_e_score = roc
                            best_svm_e = (kernel, C, gamma, degree, coef0)
            else:
                clf = SVC(
                    kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42
                )
                roc = evaluate(
                    clf,
                    X_tr_e,
                    y_tr_e,
                    X_te_e,
                    y_te_e,
                    f"SVM_e ker={kernel} C={C} gamma={gamma}",
                )
                if roc > best_svm_e_score:
                    best_svm_e_score = roc
                    best_svm_e = (kernel, C, gamma)
print("Лучший обогащённый SVM:", best_svm_e, best_svm_e_score)

print("\n-- RandomForest на обогащённых данных")
# заново определяем список критериев в явном виде, чтобы ничего не подсосало лишнего
rf_crit = ["entropy", "gini"]
best_rf_e_score = -1
best_rf_e = None
for c in rf_crit:
    for n in n_estimators:
        for mf in max_feats:
            clf = RandomForestClassifier(
                criterion=c, n_estimators=n, max_features=mf, random_state=42
            )
            roc = evaluate(
                clf, X_tr_e, y_tr_e, X_te_e, y_te_e, f"RF_e crit={c} trees={n} mf={mf}"
            )
            if roc > best_rf_e_score:
                best_rf_e_score = roc
                best_rf_e = (c, n, mf)
print("Лучший обогащённый RF:", best_rf_e, best_rf_e_score)

# На заключительном этапе берём лучшие модели, найденные на тестовых данных,
# и оцениваем их на отложенной валидационной выборке. Это позволяет получить
# оценки качества и выбрать итоговый алгоритм.
print("\nФинальная оценка на валидационной выборке")

# original best models
best_dt_model = DecisionTreeClassifier(
    ccp_alpha=best_dt[0], criterion=best_dt[1], random_state=42
)
best_dt_model.fit(X_train, y_train)
print("Оригинальное лучшее DT на валид.: ")
evaluate(best_dt_model, X_train, y_train, X_val, y_val, "DT_val")

# Затем SVM: параметры могут быть с разной длиной в кортеже,
# поэтому восстанавливаем модель условно и снова обучаем на всем train.
best_svm_model = None
if len(best_svm) == 3:
    best_svm_model = SVC(
        kernel=best_svm[0],
        C=best_svm[1],
        gamma=best_svm[2],
        probability=True,
        random_state=42,
    )
else:
    best_svm_model = SVC(
        kernel=best_svm[0],
        C=best_svm[1],
        gamma=best_svm[2],
        degree=best_svm[3],
        coef0=best_svm[4],
        probability=True,
        random_state=42,
    )
best_svm_model.fit(X_train, y_train)
print("Оригинальное лучшее SVM на валид.: ")
evaluate(best_svm_model, X_train, y_train, X_val, y_val, "SVM_val")

# И случайный лес: тот же подход, строим с найденными ранее гиперпараметрами.
best_rf_model = RandomForestClassifier(
    criterion=best_rf[0],
    n_estimators=best_rf[1],
    max_features=best_rf[2],
    random_state=42,
)
best_rf_model.fit(X_train, y_train)
print("Оригинальное лучшее RF на валид.: ")
evaluate(best_rf_model, X_train, y_train, X_val, y_val, "RF_val")

# Теперь повторяем тот же финальный анализ, но для моделей, обученных
# на обогащённых данных. Это позволяет оценить, принесло ли добавление новых
# признаков реальную пользу. Снова начинаем с дерева решений.
print("Обогащённое лучшее DT на валид.: ")
best_dt_en_model = DecisionTreeClassifier(
    ccp_alpha=best_dt_e[0], criterion=best_dt_e[1], random_state=42
)
best_dt_en_model.fit(X_tr_e, y_tr_e)
evaluate(best_dt_en_model, X_tr_e, y_tr_e, X_val_e, y_val_e, "DT_e_val")

print("Обогащённое лучшее SVM на валид.: ")
best_svm_en_model = None
if len(best_svm_e) == 3:
    best_svm_en_model = SVC(
        kernel=best_svm_e[0],
        C=best_svm_e[1],
        gamma=best_svm_e[2],
        probability=True,
        random_state=42,
    )
else:
    best_svm_en_model = SVC(
        kernel=best_svm_e[0],
        C=best_svm_e[1],
        gamma=best_svm_e[2],
        degree=best_svm_e[3],
        coef0=best_svm_e[4],
        probability=True,
        random_state=42,
    )
best_svm_en_model.fit(X_tr_e, y_tr_e)
evaluate(best_svm_en_model, X_tr_e, y_tr_e, X_val_e, y_val_e, "SVM_e_val")

print("Обогащённое лучшее RF на валид.: ")
best_rf_en_model = RandomForestClassifier(
    criterion=best_rf_e[0],
    n_estimators=best_rf_e[1],
    max_features=best_rf_e[2],
    random_state=42,
)
best_rf_en_model.fit(X_tr_e, y_tr_e)
evaluate(best_rf_en_model, X_tr_e, y_tr_e, X_val_e, y_val_e, "RF_e_val")

print("\nГотово.")


# Результаты
"""Финальная оценка на валидационной выборке
Оригинальное лучшее DT на валид.:
DT_val: acc=0.717, prec=0.669, rec=0.616, roc_auc=0.727
Оригинальное лучшее SVM на валид.: 
SVM_val: acc=0.737, prec=0.743, rec=0.608, roc_auc=0.787
Оригинальное лучшее RF на валид.: 
RF_val: acc=0.781, prec=0.768, rec=0.697, roc_auc=0.799
Обогащённое лучшее DT на валид.: 
DT_e_val: acc=0.741, prec=0.680, rec=0.625, roc_auc=0.731
Обогащённое лучшее SVM на валид.: 
SVM_e_val: acc=0.758, prec=0.735, rec=0.616, roc_auc=0.752
Обогащённое лучшее RF на валид.: 
RF_e_val: acc=0.764, prec=0.712, rec=0.689, roc_auc=0.758"""

# Выводы и замечания
# На отложенной валидации лучше всех показал себя случайный лес (RF) на
# оригинальных данных: roc_auc ~0.80. Это совпадает с результатом на тестовом
# наборе, что означает, что оптимизация гиперпараметров не привела к сильному
# переобучению.
# SVM занял второе место с roc_auc ~0.79, дерево решений было заметно ниже.
# Обогащение признаков полиномиальными комбинациями дало небольшое улучшение
# для DT и SVM, но ухудшило финальный RF; в целом выигрыш оказался незначительным,
# поэтому с точки зрения простоты лучше оставить исходные признаки.

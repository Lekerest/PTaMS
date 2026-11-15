# УСЛОВИЕ: практическое задание 5. Генераторы и случайные блуждания.
# Разрешён только готовый генератор равномерных чисел из [0, 1).
# 1. Написать генераторы ±1, показательного, нормального, Парето и Коши.
#    Вывести квантильные формулы; для нормального использовать Бокса–Мюллера.
#    Сравнить частоты с вероятностями, гистограммы — с плотностями.
#    Для Парето и Коши нормировать по всей выборке, включая значения вне окна.
# 2. Методом отбраковки получить плотность f(x) = (1+x+sin²(3πx))/2 на (0,1).
#    Проверить её интеграл и границы. Показать пробные точки и гистограмму.
#    Сравнить долю принятых точек с теоретической.
# 3. Смоделировать N независимых блужданий по n шагов: ±1 при p=0.5 и 0.55,
#    показательные при λ=1, нормальные и с плотностью задачи 2.
#    Найти μ и σ²; показать траектории, среднее и kμ, отдельно разброс и σ√k.
#    Для тех же показательных шагов сравнить ξ и ξ−1.
#    Численно сравнить среднее и стандартное отклонение при 100 и 10000 шагах.
# 4. Для плотности задачи 2 сравнить стандартизованные суммы с нормальной
#    плотностью при n=1,2,10,100, одинаковых N, интервалах и масштабах осей.
#    При фиксированном n увеличить N. Отдельно проверить нормальные шаги, n=1.
# 5. Показать блуждания Парето и Коши при том же n, сравнить с задачей 3.
#    Объяснить большие скачки и применимость формул среднего и дисперсии.

import numpy as np  # Массивы и математические вычисления.
import matplotlib.pyplot as plt  # Построение графиков.

####################################################
# Выбор задачи и параметры
####################################################
TASK = 5  # Номер задачи: 1, 2, 3, 4 или 5.
size = 10000  # Объём выборки для проверки генераторов в задачах 1 и 2.
p = 0.5  # Вероятность +1 в задаче 1; должно быть 0 < p < 1.
lam = 1  # Параметр показательного распределения в задаче 1; lam > 0.
n = 10000  # Число шагов на графиках блужданий в задачах 3 и 5.
N = 2000  # Число независимых траекторий / сумм в задачах 3 и 4.
np.random.seed(42)  # Один раз фиксируем генератор для повторяемости.

####################################################
# Задача 1. Генераторы только из равномерных чисел
####################################################
def open_uniform(size):
    u = np.random.random(size)  # Независимые равномерные числа из [0, 1).
    while np.any(u == 0):  # В логарифм и квантиль Коши нельзя передавать ноль.
        u[u == 0] = np.random.random(np.sum(u == 0))  # Перегенерируем только нули.
    return u  # Теперь каждое число строго между 0 и 1.


def signs(size, p=0.5):
    return np.where(np.random.random(size) < p, 1, -1)  # Длины частей [0,1): p и 1−p.


def exponential(size, lam=1):
    return -np.log(1 - open_uniform(size)) / lam  # Обратная к F(x)=1−exp(−λx).


def normal(size):
    u = open_uniform(size)  # Первая независимая равномерная выборка.
    v = np.random.random(size)  # Вторая независимая равномерная выборка.
    return np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v)  # Бокс–Мюллер: берём Z1.


def pareto(size):
    return (1 - open_uniform(size)) ** (-2 / 3)  # Обратная к F(x)=1−x^(−3/2).


def cauchy(size):
    return np.tan(np.pi * (open_uniform(size) - 0.5))  # Обратная к F(x)=1/2+arctan(x)/π.

####################################################
# Задача 2. Метод отбраковки
####################################################
def density(x):
    return np.where((x > 0) & (x < 1), (1 + x + np.sin(3 * np.pi * x) ** 2) / 2, 0)


def rejection(size, return_trials=False):
    sample = np.empty(size)  # Здесь будут только принятые значения.
    filled = 0  # Сколько значений уже получили.
    trial_x, trial_y = [], []  # Пробные точки нужны только для рисунка задачи 2.
    while filled < size:
        remaining = size - filled  # Сколько значений ещё требуется.
        u = np.random.random(remaining)  # Горизонтальные координаты в [0,1).
        v = 1.5 * np.random.random(remaining)  # Вертикальные координаты в [0,1.5).
        accepted = u[v < density(u)]  # Оставляем первые координаты точек под графиком.
        sample[filled:filled + len(accepted)] = accepted  # Записываем принятые значения.
        filled += len(accepted)  # Увеличиваем число заполненных элементов.
        if return_trials:
            trial_x.extend(u)  # Учитываем все попытки, включая отброшенные.
            trial_y.extend(v)
    if return_trials:
        return sample, np.array(trial_x), np.array(trial_y)
    return sample


# Моменты плотности задачи 2 получены интегрированием; вывод есть в README.
mu_f = 13 / 24  # Интеграл x*f(x) на (0,1).
var_f = 47 / 576 - 1 / (72 * np.pi ** 2)  # Интеграл x²*f(x) минус квадрат среднего.

####################################################
# Задача 3. Блуждание и его характеристики
####################################################
def random_walk(generator, n, N):
    positions = np.zeros((N, n + 1))  # Строка — траектория, столбец — момент k; S0=0.
    for j in range(N):
        positions[j, 1:] = np.cumsum(generator(n))  # Своя независимая последовательность шагов.
    return positions


def draw_walks(axis, positions, mu, title):
    k = np.arange(positions.shape[1])  # Моменты 0,1,...,n.
    axis.plot(k, positions[:3].T, alpha=0.6)  # Первые три траектории.
    axis.plot(k, positions.mean(axis=0), color="red", label="Выборочное среднее")
    axis.plot(k, k * mu, "k--", label="Теория kμ")
    axis.set(title=title, xlabel="Число шагов k", ylabel="Положение Sₖ")


def print_comparison(positions, mu, variance, title):
    print(f"\n{title}: μ = {mu:.6f}, σ² = {variance:.6f}")
    print("Шаги | Среднее: теория / выборка | Ст. отклонение: теория / выборка")
    for steps in [100, 10000]:
        values = positions[:, steps]  # Положения всех N траекторий в один момент.
        print(f"{steps:5d} | {steps * mu:10.3f} / {values.mean():10.3f} | "
              f"{np.sqrt(steps * variance):10.3f} / {values.std():10.3f}")


# В задачах 3 и 5 исследуем именно параметры, перечисленные в ТЗ.
walk_types = [
    ("±1, p = 0.5", lambda size: signs(size, 0.5), 0, 1),
    ("±1, p = 0.55", lambda size: signs(size, 0.55), 0.1, 0.99),
    ("Показательные, λ = 1", exponential, 1, 1),
    ("Стандартные нормальные", normal, 0, 1),
    ("Плотность задачи 2", rejection, mu_f, var_f),
]

####################################################
# Общие функции построения графиков
####################################################
def histogram(axis, sample, bins, label="Выборка"):
    counts, edges = np.histogram(sample, bins=bins)  # Числа наблюдений по интервалам.
    heights = counts / (len(sample) * np.diff(edges))  # Делим на ВСЮ выборку и ширину.
    axis.stairs(heights, edges, fill=True, alpha=0.4, label=label)
    axis.set(xlabel="Значение", ylabel="Плотность")


def finish_figure(fig):
    for axis in fig.axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()

####################################################
# Запуск выбранной задачи
####################################################
if TASK == 1:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Задача 1. Генераторы: объём выборки {size}")
    axes = axes.ravel()
    sample = signs(size, p)
    frequencies = [np.mean(sample == -1), np.mean(sample == 1)]  # Относительные частоты.
    axes[0].bar([-1, 1], frequencies, width=0.4, alpha=0.6, label="Выборка")
    axes[0].bar([-1, 1], [1 - p, p], width=0.6, fill=False,
                edgecolor="red", label="Теоретические вероятности")
    axes[0].set(title=f"Два значения, p = {p}", xticks=[-1, 1],
                xlabel="Значение", ylabel="Частота / вероятность")
    distributions = [
        (f"Показательное, λ = {lam}", exponential(size, lam), 0, 8 / lam,
         lambda x: lam * np.exp(-lam * x)),
        ("Нормальное", normal(size), -4, 4,
         lambda x: np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)),
        ("Парето: окно [1, 10]", pareto(size), 1, 10, lambda x: 1.5 / x ** 2.5),
        ("Коши: окно [−10, 10]", cauchy(size), -10, 10,
         lambda x: 1 / (np.pi * (1 + x ** 2))),
    ]
    for axis, (title, sample, left, right, pdf) in zip(axes[1:5], distributions):
        histogram(axis, sample, np.linspace(left, right, 51))
        x = np.linspace(left, right, 500)
        axis.plot(x, pdf(x), "r-", label="Теоретическая плотность")
        axis.set_title(title)
    fig.delaxes(axes[5])  # Распределений пять, шестая панель не нужна.
    finish_figure(fig)

elif TASK == 2:
    sample, trial_x, trial_y = rejection(size, return_trials=True)
    accepted = trial_y < density(trial_x)  # Какие пробные точки были приняты.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Задача 2. Метод отбраковки")
    x = np.linspace(0, 1, 500)
    for mask, label in [(accepted[:400], "Принятые"), (~accepted[:400], "Отброшенные")]:
        axes[0].scatter(trial_x[:400][mask], trial_y[:400][mask], s=12, label=label)
    axes[0].plot(x, density(x), "k-", label="Плотность f(x)")
    axes[0].set(xlabel="U", ylabel="V", ylim=(0, 1.5), title="Первые 400 пробных точек")
    histogram(axes[1], sample, np.linspace(0, 1, 41))
    axes[1].plot(x, density(x), "r-", label="Плотность f(x)")
    axes[1].set_title("Полученная выборка и плотность")
    print(f"Принято: {len(sample)} из {len(trial_x)} попыток")
    print(f"Доля принятых: {len(sample) / len(trial_x):.4f}; теория: {2 / 3:.4f}")
    finish_figure(fig)

elif TASK == 3:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Задача 3. Блуждания: n = {n}, N = {N}")
    axes = axes.ravel()
    fig_std, axis_std = plt.subplots(figsize=(11, 5))  # Разброс — на отдельном графике по ТЗ.
    k = np.arange(n + 1)
    for i, (title, generator, mu, variance) in enumerate(walk_types):
        positions = random_walk(generator, max(n, 10000), N)  # Хватает и для числового сравнения.
        draw_walks(axes[i], positions[:, :n + 1], mu, title)
        std = positions[:, :n + 1].std(axis=0)  # Разброс по N траекториям при каждом k.
        line, = axis_std.plot(k, std, label=title)
        axis_std.plot(k, np.sqrt(variance * k), "--", color=line.get_color(),
                      label=f"Теория σ√k: {title}")
        print_comparison(positions, mu, variance, title)
        if i == 2:  # Центрируем именно те же показательные шаги: Sₖ заменяется на Sₖ−k.
            positions -= np.arange(positions.shape[1])
            draw_walks(axes[5], positions[:, :n + 1], 0, "Те же показательные шаги минус 1")
            print_comparison(positions, 0, 1, "Показательные шаги минус 1")
        del positions  # Для следующего распределения предыдущий массив уже не нужен.
    axis_std.set(title=f"Задача 3. Разброс по траекториям: n = {n}, N = {N}",
                 xlabel="Число шагов k", ylabel="Стандартное отклонение")
    print("\nПри шагах ξ−1 среднее уменьшается на k, стандартное отклонение не меняется.")
    print("При увеличении числа шагов в 100 раз среднее kμ умножается на 100,")
    print("а стандартное отклонение σ√k — на 10. При μ=0 среднее остаётся нулём.")
    finish_figure(fig)
    finish_figure(fig_std)

elif TASK == 4:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    fig.suptitle("Задача 4. Центральная предельная теорема")
    axes = axes.ravel()
    bins = np.linspace(-4, 4, 41)  # Одинаковые интервалы и масштабы всех шести панелей.
    x = np.linspace(-4, 4, 500)
    pdf = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    for axis, steps in zip(axes[:4], [1, 2, 10, 100]):
        sums = rejection(N * steps).reshape(N, steps).sum(axis=1)  # Свои шаги для каждой суммы.
        eta = (sums - steps * mu_f) / np.sqrt(steps * var_f)  # Теоретические μ и σ² задачи 3.
        histogram(axis, eta, bins)
        axis.set_title(f"Плотность задачи 2: n = {steps}, N = {N}")
        if steps == 10:
            eta_small = eta  # Сохраняем для сравнения N при фиксированном n=10.
    sums = rejection(10 * N * 10).reshape(10 * N, 10).sum(axis=1)
    eta_large = (sums - 10 * mu_f) / np.sqrt(10 * var_f)
    histogram(axes[4], eta_small, bins, label=f"N = {N}")
    histogram(axes[4], eta_large, bins, label=f"N = {10 * N}")
    axes[4].set_title("Влияние N при n = 10")
    histogram(axes[5], normal(N), bins)  # При n=1 стандартизованная сумма равна самому шагу.
    axes[5].set_title(f"Нормальные шаги: n = 1, N = {N}")
    for axis in axes:
        axis.plot(x, pdf, "r-", label="Стандартная нормальная плотность")
        axis.set_xlim(-4, 4)
    print("Увеличение n приближает закон стандартизованной суммы к нормальному.")
    print("Увеличение N уменьшает случайные колебания гистограммы при том же законе.")
    print("Отдельные гистограммы не обязаны улучшаться при каждом увеличении n.")
    print("Для нормальных шагов нормальный закон получается точно при любом n.")
    finish_figure(fig)

elif TASK == 5:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Задача 5. Тяжёлые хвосты: одинаковое число шагов n = {n}")
    k = np.arange(n + 1)
    for title, generator, mu, variance in walk_types:
        positions = random_walk(generator, n, 1)  # По одной траектории для сравнения пяти законов.
        axes[0].plot(k, positions[0], label=title)
    axes[0].set_title("Распределения из задачи 3")
    for axis, generator, title in [(axes[1], pareto, "Парето"), (axes[2], cauchy, "Коши")]:
        positions = random_walk(generator, n, 3)  # Несколько независимых траекторий каждого закона.
        for j in range(3):
            axis.plot(k, positions[j], label=f"Траектория {j + 1}")
        axis.set_title(title)
    for axis in axes:
        axis.set(xlabel="Число шагов k", ylabel="Положение Sₖ")
    print("Парето: редкие большие положительные скачки; E[Sₖ]=3k, дисперсия бесконечна при k≥1.")
    print("Коши: большие скачки обоих знаков; математическое ожидание не существует,")
    print("конечной дисперсии нет. Формулы kμ и σ√k для него неприменимы.")
    finish_figure(fig)

plt.show()  # Показываем только графики выбранной задачи.

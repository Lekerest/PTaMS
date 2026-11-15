# УСЛОВИЕ ЗАДАЧИ: практическое задание 4. Пуассоновский процесс.
# Источник: lectures (4).pdf, страницы 134–137.
# 1. Схема Бернулли: n ячеек на [0, 1], p = min(lambda / n, 1),
#    события ставятся на правых концах успешных ячеек.
#    Повторить N раз, собрать числа событий и внутренние расстояния.
# 2. Число событий взять из распределения Пуассона, положения —
#    равномерно на [0, 1] и отсортировать. Сравнить расстояния с задачей 1.
# 3. Получить моменты событий накоплением показательных ожиданий.
#    Остановиться, когда сумма превысит 1; вышедшую точку не сохранять.
# 4. Повторить третий алгоритм N раз. Сравнить числа событий всех трёх
#    способов с Пуассоном, а внутренние расстояния — друг с другом.
#    Отдельно исследовать увеличение n и увеличение N.
# Во всех способах учитывать нулевые реализации; не добавлять расстояния
# до границ и между разными реализациями. Для сравнений гистограммы
# расстояний должны иметь одинаковые границы и нормировку.

import numpy as np  # Случайные числа и вычисления с массивами.
import matplotlib.pyplot as plt  # Рисование графиков.
from math import exp, factorial  # Формула вероятностей распределения Пуассона.


####################################################
# Выбор задачи и параметры
####################################################
TASK = 4  # 1, 2, 3, 4 — задачи; 5 — дополнительное исследование из ТЗ.
LAMBDA = 5  # Среднее число событий на отрезке длины 1 в модели Пуассона.
n = 10  # Число ячеек в схеме Бернулли.
N = 10000  # Число независимых реализаций каждого алгоритма.
np.random.seed(42)  # Один раз фиксируем генератор для повторяемости.


####################################################
# Задача 1. Схема Бернулли
####################################################
def bernoulli_process(lam, n, N):
    p = min(lam / n, 1)  # Вероятность успеха в одной ячейке.
    counts = []  # Количество событий в каждой реализации, включая нули.
    gaps = []  # Только расстояния между соседними событиями.

    for _ in range(N):
        successes = np.random.random(n) < p  # n независимых испытаний Бернулли.
        points = np.arange(1, n + 1)[successes] / n  # Правые концы успешных ячеек.
        counts.append(len(points))  # Учитываем также реализации без событий.
        gaps.extend(np.diff(points))  # У пустого массива и одной точки разностей нет.

    return np.array(counts), np.array(gaps)


####################################################
# Задача 2. Число событий и равномерные положения
####################################################
def uniform_process(lam, N):
    counts = np.random.poisson(lam, size=N)  # Число событий в каждой реализации.
    gaps = []

    for m in counts:
        points = np.sort(np.random.uniform(0, 1, size=m))  # m точек по возрастанию.
        gaps.extend(np.diff(points))  # Разности только внутри текущей реализации.

    return counts, np.array(gaps)


####################################################
# Задача 3. Одна реализация через времена ожидания
####################################################
def waiting_realization(lam):
    first_wait = np.random.exponential(scale=1 / lam)  # Первое ожидание W1.
    t = first_wait  # Первый момент события равен первому ожиданию.
    points = []
    while t <= 1:  # Сохраняем только точки внутри [0, 1].
        points.append(t)
        t += np.random.exponential(scale=1 / lam)  # Накапливаем независимые ожидания.
    return np.array(points), first_wait  # W1 возвращаем даже при W1 > 1.


####################################################
# Задача 4. N реализаций третьего алгоритма
####################################################
def waiting_process(lam, N):
    counts = []
    gaps = []
    first_waits = []
    for _ in range(N):
        points, first_wait = waiting_realization(lam)
        counts.append(len(points))
        gaps.extend(np.diff(points))  # Расстояния до 0 и до 1 не включаем.
        first_waits.append(first_wait)  # Нужны для дополнительного исследования.
    return np.array(counts), np.array(gaps), np.array(first_waits)


####################################################
# Общие функции оформления графиков
####################################################
def draw_counts(axis, counts, lam, label):
    k = np.arange(counts.max() + 1)  # Возможные целые значения до наблюдённого максимума.
    frequencies = np.bincount(counts) / len(counts)  # Доля реализаций для каждого k.
    axis.bar(k, frequencies, width=0.5, alpha=0.7, label=label)
    probabilities = [exp(-lam) * lam ** int(value) / factorial(int(value)) for value in k]
    axis.plot(k, probabilities, "ko--", markersize=4, label="Теория Пуассона")
    axis.set(xlabel="Количество событий k", ylabel="Частота / вероятность")
    axis.set_xticks(k)  # Целочисленные подписи оси.


def finish_figure(fig):
    for axis in fig.axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()


# Общие интервалы нужны только для сравнения непрерывных расстояний по ТЗ.
gap_bins = np.linspace(0, 1, 22)


####################################################
# Графики только выбранной задачи
####################################################
if TASK == 1:
    counts, gaps = bernoulli_process(LAMBDA, n, N)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Задача 1. Бернулли: λ = {LAMBDA}, n = {n}, N = {N}")
    draw_counts(axes[0], counts, LAMBDA, "Схема Бернулли")
    axes[0].set_title("Число событий за один опыт")

    # Возвращаем расстояниям номера шагов: 0.1 -> 1, 0.2 -> 2 и так далее.
    # Округление убирает только погрешность вычислений вроде 0.3 - 0.2.
    steps = np.rint(gaps * n).astype(int)
    lengths = np.arange(1, n) / n  # Точные возможные длины: от 1/n до (n-1)/n.
    frequencies = np.bincount(steps, minlength=n)[1:n] / len(gaps)
    axes[1].stem(lengths, frequencies, basefmt=" ", label="Доля каждого точного расстояния")
    axes[1].set_xticks(lengths)
    axes[1].set(title="Точные расстояния между событиями", xlabel="Длина расстояния", ylabel="Доля расстояний")
    finish_figure(fig)

elif TASK == 2:
    counts2, gaps2 = uniform_process(LAMBDA, N)
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.hist(gaps2, bins=gap_bins, density=True, histtype="step", label="Пуассон")
    for grid_n in [10, 200]:  # В задаче 2 требуется сравнение с несколькими сетками.
        counts1, gaps1 = bernoulli_process(LAMBDA, grid_n, N)
        axis.hist(gaps1, bins=gap_bins, density=True, histtype="step", label=f"Бернулли, n = {grid_n}")
    axis.set(title=f"Задача 2. Сравнение расстояний: λ = {LAMBDA}, N = {N}", xlabel="Расстояние", ylabel="Плотность")
    finish_figure(fig)

elif TASK == 3:
    points, first_wait = waiting_realization(LAMBDA)
    print("Моменты событий:", points)  # Одна реализация, полученная суммированием ожиданий.
    fig, axis = plt.subplots(figsize=(10, 3))
    axis.vlines(points, 0, 1, label="Моменты событий")  # Каждая линия — одна точная точка времени.
    axis.plot(points, np.full(len(points), 0.5), "o")
    axis.set(xlim=(0, 1), ylim=(0, 1), title=f"Задача 3. Одна реализация: λ = {LAMBDA}", xlabel="Время")
    axis.set_yticks([])  # Высота здесь ничего не измеряет: показываем только моменты времени.
    finish_figure(fig)

elif TASK == 4:
    results = []
    for grid_n in [10, 200]:
        counts, gaps = bernoulli_process(LAMBDA, grid_n, N)
        results.append((counts, gaps, f"Бернулли, n = {grid_n}"))
    counts2, gaps2 = uniform_process(LAMBDA, N)
    counts3, gaps3, first_waits = waiting_process(LAMBDA, N)
    results.extend([(counts2, gaps2, "Пуассон + равномерные точки"), (counts3, gaps3, "Показательные ожидания")])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f"Задача 4. Итоговое сравнение: λ = {LAMBDA}, основное N = {N}")
    k = np.arange(max(counts.max() for counts, gaps, label in results) + 1)
    for counts, gaps, label in results:
        frequencies = np.bincount(counts, minlength=len(k)) / len(counts)
        axes[0].plot(k, frequencies, "o-", markersize=3, label=label)
        axes[1].hist(gaps, bins=gap_bins, density=True, histtype="step", label=label)
    probabilities = [exp(-LAMBDA) * LAMBDA ** int(value) / factorial(int(value)) for value in k]
    axes[0].plot(k, probabilities, "k--", label="Теория Пуассона")
    axes[0].set(title="Числа событий и влияние n", xlabel="Количество событий k", ylabel="Частота / вероятность")
    axes[0].set_xticks(k[::2])
    axes[1].set(title="Внутренние расстояния и влияние n", xlabel="Расстояние", ylabel="Плотность")

    # Отдельный эксперимент: меняем только N, сохраняя λ и n.
    counts_small, gaps_small = bernoulli_process(LAMBDA, 10, 100)
    counts_large = results[0][0]  # Уже полученные числа событий при n = 10 и большом N.
    k_N = np.arange(max(counts_small.max(), counts_large.max()) + 1)
    for counts, label in [(counts_small, "N = 100"), (counts_large, f"N = {N}")]:
        frequencies = np.bincount(counts, minlength=len(k_N)) / len(counts)
        axes[2].plot(k_N, frequencies, "o-", label=label)
    probabilities_N = [exp(-LAMBDA) * LAMBDA ** int(value) / factorial(int(value)) for value in k_N]
    axes[2].plot(k_N, probabilities_N, "k--", label="Теория Пуассона")
    axes[2].set(title=f"Влияние N: Бернулли, n = 10, λ = {LAMBDA}", xlabel="Количество событий k", ylabel="Частота / вероятность")
    axes[2].set_xticks(k_N)
    finish_figure(fig)

elif TASK == 5:
    counts, gaps, first_waits = waiting_process(LAMBDA, N)
    t = np.linspace(0, first_waits.max(), 300)  # Включаем ожидания больше 1.
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.hist(first_waits, bins=30, density=True, label="Первые ожидания W1")
    axis.plot(t, LAMBDA * np.exp(-LAMBDA * t), "r-", label="λ · exp(-λt)")
    axis.set(title=f"Дополнительное исследование. Первое ожидание: λ = {LAMBDA}, N = {N}", xlabel="Время ожидания", ylabel="Плотность")
    finish_figure(fig)

plt.show()  # Открывается только рисунок выбранной задачи.

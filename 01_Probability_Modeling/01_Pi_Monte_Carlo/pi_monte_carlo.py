import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


# ============================================================================
# 1) Структуры данных и интерфейс генератора
# ============================================================================
class RandomSource(Protocol):
    def random(self) -> float:
        """Вернуть случайное число из полуинтервала [0, 1)."""


@dataclass(eq=True)
class SimulationResult:
    x_coordinates: list[float]
    y_coordinates: list[float]
    inside_circle: list[bool]
    estimates: list[float]
    absolute_errors: list[float]
    cumulative_hits: list[int]

    @property
    def hits(self) -> int:
        return self.cumulative_hits[-1]

    @property
    def estimate(self) -> float:
        return self.estimates[-1]


# ============================================================================
# 2) Геометрическая проверка
# ============================================================================
def is_inside_circle(x: float, y: float) -> bool:
    """Проверяет принадлежность точки кругу с центром (0.5, 0.5)."""
    return (x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.25


# ============================================================================
# 3) Моделирование методом Монте-Карло
# ============================================================================
def simulate_pi(n_points: int, rng: RandomSource) -> SimulationResult:
    """Генерирует точки и строит накопительную оценку числа pi."""
    if n_points <= 0:
        raise ValueError("Количество точек должно быть положительным")

    x_coordinates: list[float] = []
    y_coordinates: list[float] = []
    inside_circle: list[bool] = []
    estimates: list[float] = []
    absolute_errors: list[float] = []
    cumulative_hits: list[int] = []
    hits = 0

    for point_number in range(1, n_points + 1):
        x = rng.random()
        y = rng.random()
        inside = is_inside_circle(x, y)
        if inside:
            hits += 1

        estimate = 4.0 * hits / point_number
        x_coordinates.append(x)
        y_coordinates.append(y)
        inside_circle.append(inside)
        estimates.append(estimate)
        absolute_errors.append(abs(estimate - math.pi))
        cumulative_hits.append(hits)

    return SimulationResult(
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        inside_circle=inside_circle,
        estimates=estimates,
        absolute_errors=absolute_errors,
        cumulative_hits=cumulative_hits,
    )


# ============================================================================
# 4) Контрольные точки и сохранение численных результатов
# ============================================================================
def choose_checkpoints(n_points: int) -> list[int]:
    """Выбирает компактный набор n для CSV вместо записи каждой точки."""
    checkpoints = set(range(1, min(10, n_points) + 1))
    scale = 10
    while scale <= n_points:
        for multiplier in (1, 2, 5):
            value = multiplier * scale
            if value <= n_points:
                checkpoints.add(value)
        scale *= 10
    checkpoints.add(n_points)
    return sorted(checkpoints)


def save_numerical_results(
    result: SimulationResult,
    output_dir: Path,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_points": len(result.estimates),
        "seed": seed,
        "hits": result.hits,
        "pi_estimate": result.estimate,
        "absolute_error": result.absolute_errors[-1],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    with (output_dir / "checkpoints.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("n", "hits", "pi_estimate", "absolute_error"),
        )
        writer.writeheader()
        for n in choose_checkpoints(len(result.estimates)):
            index = n - 1
            writer.writerow(
                {
                    "n": n,
                    "hits": result.cumulative_hits[index],
                    "pi_estimate": result.estimates[index],
                    "absolute_error": result.absolute_errors[index],
                }
            )


# ============================================================================
# 5) Построение обязательных графиков
# ============================================================================
def create_plots(
    result: SimulationResult,
    output_dir: Path,
    show: bool = False,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    shown_points = min(5_000, len(result.estimates))
    inside_x = [
        result.x_coordinates[index]
        for index in range(shown_points)
        if result.inside_circle[index]
    ]
    inside_y = [
        result.y_coordinates[index]
        for index in range(shown_points)
        if result.inside_circle[index]
    ]
    outside_x = [
        result.x_coordinates[index]
        for index in range(shown_points)
        if not result.inside_circle[index]
    ]
    outside_y = [
        result.y_coordinates[index]
        for index in range(shown_points)
        if not result.inside_circle[index]
    ]

    angles = [2.0 * math.pi * index / 360 for index in range(361)]
    circle_x = [0.5 + 0.5 * math.cos(angle) for angle in angles]
    circle_y = [0.5 + 0.5 * math.sin(angle) for angle in angles]

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(inside_x, inside_y, s=8, alpha=0.55, label="Внутри круга")
    axis.scatter(outside_x, outside_y, s=8, alpha=0.55, label="Вне круга")
    axis.plot(circle_x, circle_y, color="black", linewidth=1.5, label="Граница круга")
    axis.set(
        title=f"Первые {shown_points} точек Монте-Карло",
        xlabel="x",
        ylabel="y",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output_dir / "point_cloud.png", dpi=160)

    error_figure, error_axis = plt.subplots(figsize=(9, 5))
    point_numbers = range(1, len(result.absolute_errors) + 1)
    error_axis.plot(point_numbers, result.absolute_errors, linewidth=0.8)
    error_axis.set_xscale("log")
    error_axis.set(
        title="Абсолютная ошибка оценки числа π",
        xlabel="Количество сгенерированных точек n (логарифмическая шкала)",
        ylabel="|оценка π − math.pi|",
    )
    error_axis.grid(True, which="both", alpha=0.25)
    error_figure.tight_layout()
    error_figure.savefig(output_dir / "absolute_error.png", dpi=160)

    if show:
        plt.show()
    plt.close(figure)
    plt.close(error_figure)


# ============================================================================
# 6) Параметры командной строки
# ============================================================================
def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("значение должно быть положительным")
    return number


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Оценка числа pi методом Монте-Карло"
    )
    parser.add_argument(
        "--points",
        type=positive_integer,
        default=100_000,
        help="количество случайных точек (по умолчанию: 100000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="начальное состояние генератора (по умолчанию: 42)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="показать графики после сохранения",
    )
    return parser.parse_args(arguments)


# ============================================================================
# 7) Полный запуск эксперимента
# ============================================================================
def main(arguments: Sequence[str] | None = None) -> SimulationResult:
    args = parse_arguments(arguments)
    output_dir = Path(__file__).resolve().parent / "results"

    # Seed задаётся ровно один раз перед всей генерацией точек.
    rng = random.Random(args.seed)
    result = simulate_pi(args.points, rng)

    save_numerical_results(result, output_dir, args.seed)
    create_plots(result, output_dir, args.show)

    print(f"Количество точек: {args.points}")
    print(f"Попаданий в круг: {result.hits}")
    print(f"Оценка pi: {result.estimate:.8f}")
    print(f"Абсолютная ошибка: {result.absolute_errors[-1]:.8f}")
    print(f"Результаты сохранены в: {output_dir}")
    return result


if __name__ == "__main__":
    main()

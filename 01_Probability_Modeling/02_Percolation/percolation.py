"""Оценка вероятности перколяции прямоугольного клеточного поля."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import deque
from pathlib import Path
from typing import Sequence


# ============================================================================
# 1) Параметры эксперимента по умолчанию
# ============================================================================

DEFAULT_PROBABILITIES = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.45,
    0.5,
    0.525,
    0.55,
    0.575,
    0.6,
    0.625,
    0.65,
    0.675,
    0.7,
    0.75,
    0.8,
    0.9,
    1.0,
]
Cell = tuple[int, int]
Field = list[list[bool]]


# ============================================================================
# 2) Генерация нового случайного поля
# ============================================================================

def generate_field(rows: int, cols: int, p: float, rng: random.Random) -> Field:
    """Создать поле, где каждая клетка открыта независимо с вероятностью p."""
    if rows <= 0 or cols <= 0:
        raise ValueError("Число строк и столбцов должно быть положительным")
    if not 0.0 <= p <= 1.0:
        raise ValueError("Вероятность p должна лежать в диапазоне [0, 1]")

    return [[rng.random() < p for _ in range(cols)] for _ in range(rows)]


# ============================================================================
# 3) Поиск перколирующего пути с помощью BFS
# ============================================================================

def find_percolating_path(field: Sequence[Sequence[bool]]) -> list[Cell] | None:
    """Найти путь от верхней строки до нижней по открытым клеткам.

    Переходы разрешены только между клетками с общей стороной.
    """
    if not field or not field[0]:
        return None

    rows = len(field)
    cols = len(field[0])
    if any(len(row) != cols for row in field):
        raise ValueError("Все строки поля должны иметь одинаковую длину")

    queue: deque[Cell] = deque()
    parents: dict[Cell, Cell | None] = {}

    for col in range(cols):
        if field[0][col]:
            start = (0, col)
            queue.append(start)
            parents[start] = None

    directions = ((1, 0), (0, 1), (-1, 0), (0, -1))

    while queue:
        row, col = queue.popleft()

        if row == rows - 1:
            path: list[Cell] = []
            current: Cell | None = (row, col)
            while current is not None:
                path.append(current)
                current = parents[current]
            path.reverse()
            return path

        for row_step, col_step in directions:
            neighbour = (row + row_step, col + col_step)
            next_row, next_col = neighbour
            inside = 0 <= next_row < rows and 0 <= next_col < cols
            if inside and field[next_row][next_col] and neighbour not in parents:
                parents[neighbour] = (row, col)
                queue.append(neighbour)

    return None


# ============================================================================
# 4) Серия опытов и оценка вероятности K / M
# ============================================================================

def estimate_percolation(
    rows: int,
    cols: int,
    probabilities: Sequence[float],
    trials: int,
    rng: random.Random,
    report_progress: bool = False,
) -> tuple[dict[float, dict[str, float | int]], dict[float, dict[str, object]]]:
    """Оценить вероятность перколяции для каждого p и сохранить реальные поля."""
    if trials <= 0:
        raise ValueError("Число опытов должно быть положительным")

    results: dict[float, dict[str, float | int]] = {}
    examples: dict[float, dict[str, object]] = {}

    for p in probabilities:
        successes = 0
        fallback_example: dict[str, object] | None = None
        percolating_example: dict[str, object] | None = None

        for _ in range(trials):
            field = generate_field(rows, cols, p, rng)
            path = find_percolating_path(field)
            fallback_example = {"field": field, "path": path}
            if path is not None:
                successes += 1
                if percolating_example is None:
                    percolating_example = fallback_example

        results[float(p)] = {
            "successes": successes,
            "trials": trials,
            "estimate": successes / trials,
        }
        if fallback_example is None:
            raise RuntimeError("Не удалось сохранить пример поля")
        examples[float(p)] = percolating_example or fallback_example
        if report_progress:
            print(f"p = {p:>5.3f}: K = {successes:>4}, K/M = {successes / trials:.4f}")

    return results, examples


# ============================================================================
# 5) Сохранение численных результатов
# ============================================================================

def save_results(
    output_dir: Path,
    rows: int,
    cols: int,
    trials: int,
    seed: int,
    results: dict[float, dict[str, float | int]],
) -> None:
    """Сохранить таблицу CSV и сводку JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "percolation_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["p", "successes", "trials", "estimated_probability"])
        for p, values in results.items():
            writer.writerow([p, values["successes"], values["trials"], values["estimate"]])

    summary = {
        "parameters": {"rows": rows, "cols": cols, "trials": trials, "seed": seed},
        "results": [
            {
                "p": p,
                "successes": values["successes"],
                "trials": values["trials"],
                "estimated_probability": values["estimate"],
            }
            for p, values in results.items()
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


# ============================================================================
# 6) Построение обязательного графика и примеров полей
# ============================================================================

def save_plots(
    output_dir: Path,
    results: dict[float, dict[str, float | int]],
    examples: dict[float, dict[str, object]],
    rows: int,
    cols: int,
    trials: int,
    show: bool,
) -> None:
    """Построить график оценки и показать реальные поля из серии опытов."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    probabilities = list(results)
    estimates = [float(results[p]["estimate"]) for p in probabilities]

    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(probabilities, estimates, marker="o", linewidth=2, color="#2455a4")
    axis.set(
        title=f"Оценка вероятности перколяции\nПоле {rows} × {cols}, опытов для каждого p: {trials}",
        xlabel="Вероятность проходимости клетки p",
        ylabel="Доля полей с путем K / M",
        xlim=(0, 1),
        ylim=(-0.03, 1.03),
    )
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "percolation_probability.png", dpi=180)

    targets = (0.3, 0.575, 0.8)
    selected = [min(probabilities, key=lambda p: abs(p - target)) for target in targets]
    fig_examples, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    field_cmap = ListedColormap(["#ececec", "#4c78a8"])

    for axis, p in zip(axes, selected):
        example = examples[p]
        field = example["field"]
        path = example["path"]
        axis.imshow(field, cmap=field_cmap, vmin=0, vmax=1, interpolation="none")
        if path:
            path_rows = [cell[0] for cell in path]
            path_cols = [cell[1] for cell in path]
            axis.plot(path_cols, path_rows, color="#e45756", linewidth=2)
        outcome = "путь есть" if path else "пути нет"
        axis.set_title(f"p = {p:g}: {outcome}")
        axis.set_xticks([])
        axis.set_yticks([])

    legend_items = [
        Patch(facecolor="#4c78a8", label="проходимая клетка"),
        Patch(facecolor="#ececec", edgecolor="#cccccc", label="непроходимая клетка"),
        Line2D([0], [0], color="#e45756", linewidth=2, label="найденный путь"),
    ]
    fig_examples.suptitle(f"Примеры полей {rows} × {cols} из выполненных опытов")
    fig_examples.legend(handles=legend_items, loc="lower center", ncol=3)
    fig_examples.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig_examples.savefig(output_dir / "field_examples.png", dpi=180)

    if show:
        plt.show()
    plt.close("all")


# ============================================================================
# 7) Аргументы командной строки
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=30, help="число строк поля")
    parser.add_argument("--cols", type=int, default=30, help="число столбцов поля")
    parser.add_argument("--trials", type=int, default=1000, help="число опытов для каждого p")
    parser.add_argument("--seed", type=int, default=42, help="seed генератора случайных чисел")
    parser.add_argument("--show", action="store_true", help="показать графики после сохранения")
    return parser.parse_args()


# ============================================================================
# 8) Запуск полного исследования
# ============================================================================

def main() -> None:
    args = parse_args()
    output_dir = Path(__file__).resolve().parent / "results"

    # Seed задается один раз: этот генератор используется во всех опытах.
    rng = random.Random(args.seed)

    print("Моделирование перколяции")
    print(f"Поле: {args.rows} x {args.cols}; опытов на p: {args.trials}; seed: {args.seed}\n")
    results, examples = estimate_percolation(
        rows=args.rows,
        cols=args.cols,
        probabilities=DEFAULT_PROBABILITIES,
        trials=args.trials,
        rng=rng,
        report_progress=True,
    )
    save_results(output_dir, args.rows, args.cols, args.trials, args.seed, results)
    save_plots(output_dir, results, examples, args.rows, args.cols, args.trials, args.show)
    print(f"\nРезультаты сохранены в: {output_dir}")


if __name__ == "__main__":
    main()

import math
import random
import unittest

from pi_monte_carlo import is_inside_circle, simulate_pi


class SequenceRandom:
    """Минимальный генератор с заранее известной последовательностью."""

    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def random(self) -> float:
        return next(self._values)


class PiMonteCarloTests(unittest.TestCase):
    def test_known_points_are_classified_by_circle_equation(self) -> None:
        cases = [
            ((0.5, 0.5), True),   # центр
            ((0.0, 0.5), True),   # граница окружности
            ((1.0, 0.5), True),   # граница окружности
            ((0.0, 0.0), False),  # угол квадрата
            ((1.0, 1.0), False),  # противоположный угол
        ]

        for point, expected in cases:
            with self.subTest(point=point):
                self.assertEqual(is_inside_circle(*point), expected)

    def test_simulation_builds_running_estimate_from_known_points(self) -> None:
        # Три точки: попадание, промах, попадание на границу.
        rng = SequenceRandom([0.5, 0.5, 0.0, 0.0, 0.5, 0.0])

        result = simulate_pi(3, rng)

        self.assertEqual(result.inside_circle, [True, False, True])
        self.assertEqual(result.hits, 2)
        self.assertEqual(result.estimates, [4.0, 2.0, 8.0 / 3.0])
        self.assertEqual(
            result.absolute_errors,
            [abs(4.0 - math.pi), abs(2.0 - math.pi), abs(8.0 / 3.0 - math.pi)],
        )

    def test_fixed_seed_reproduces_the_whole_simulation(self) -> None:
        first = simulate_pi(25, random.Random(42))
        second = simulate_pi(25, random.Random(42))

        self.assertEqual(first, second)

    def test_non_positive_number_of_points_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            simulate_pi(0, random.Random(42))


if __name__ == "__main__":
    unittest.main()

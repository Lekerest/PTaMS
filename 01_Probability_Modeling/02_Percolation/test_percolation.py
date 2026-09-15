import random
import unittest

from percolation import estimate_percolation, find_percolating_path, generate_field


class FindPercolatingPathTests(unittest.TestCase):
    def test_diagonal_cells_do_not_form_a_path(self):
        field = [
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ]

        self.assertIsNone(find_percolating_path(field))

    def test_path_may_start_from_any_open_cell_in_top_row(self):
        field = [
            [False, False, True],
            [False, True, True],
            [False, True, False],
        ]

        path = find_percolating_path(field)

        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 2))
        self.assertEqual(path[-1][0], 2)

    def test_path_may_move_sideways_and_upward(self):
        field = [
            [True, False, False, False, False],
            [True, False, False, False, False],
            [True, False, True, True, True],
            [True, False, True, False, True],
            [True, True, True, False, True],
            [False, False, False, False, True],
            [False, False, False, False, True],
        ]

        path = find_percolating_path(field)

        self.assertIsNotNone(path)
        steps = list(zip(path, path[1:]))
        self.assertTrue(any(second[0] < first[0] for first, second in steps))
        self.assertTrue(any(second[1] != first[1] for first, second in steps))
        self.assertTrue(all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in steps))

    def test_one_row_percolates_when_it_contains_an_open_cell(self):
        self.assertEqual(find_percolating_path([[False, True, False]]), [(0, 1)])
        self.assertIsNone(find_percolating_path([[False, False, False]]))

    def test_one_column_requires_a_continuous_open_chain(self):
        self.assertEqual(
            find_percolating_path([[True], [True], [True]]),
            [(0, 0), (1, 0), (2, 0)],
        )
        self.assertIsNone(find_percolating_path([[True], [False], [True]]))


class SimulationTests(unittest.TestCase):
    def test_p_zero_and_one_have_exact_probabilities(self):
        results, examples = estimate_percolation(
            rows=4,
            cols=7,
            probabilities=[0.0, 1.0],
            trials=20,
            rng=random.Random(42),
        )

        self.assertEqual(results[0.0]["successes"], 0)
        self.assertEqual(results[0.0]["estimate"], 0.0)
        self.assertEqual(results[1.0]["successes"], 20)
        self.assertEqual(results[1.0]["estimate"], 1.0)
        self.assertEqual(len(examples[1.0]["field"]), 4)
        self.assertEqual(len(examples[1.0]["field"][0]), 7)

    def test_rectangular_field_has_requested_shape(self):
        field = generate_field(2, 5, 0.5, random.Random(7))

        self.assertEqual(len(field), 2)
        self.assertTrue(all(len(row) == 5 for row in field))

    def test_fixed_seed_reproduces_complete_simulation(self):
        arguments = {
            "rows": 5,
            "cols": 8,
            "probabilities": [0.2, 0.5, 0.8],
            "trials": 40,
        }

        first = estimate_percolation(rng=random.Random(123), **arguments)
        second = estimate_percolation(rng=random.Random(123), **arguments)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

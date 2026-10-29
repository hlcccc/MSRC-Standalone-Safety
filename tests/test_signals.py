import copy
import unittest

import numpy as np

from msrc_safety.signals import fit_directions, project, selected_layers, unit_vectors


class SignalTests(unittest.TestCase):
    def records(self):
        return [dict(id=str(i), group=str(i), position=1, role='fit', label=i % 2,
                     label_source='response_prefix_judge') for i in range(64)]

    def test_layer_selection_and_shapes(self):
        self.assertEqual(selected_layers(28), (14, 21, 28))
        for bad in (True, 3, 28., 0):
            with self.assertRaises(ValueError):
                selected_layers(bad)
        self.assertEqual(unit_vectors(np.zeros((3, 4))).shape, (3, 4))
        with self.assertRaises(ValueError):
            unit_vectors(np.ones((4, 3)))

    def test_projection_normalizes_and_is_not_a_probability(self):
        model = dict(weight=np.ones((3, 2)) * 10, bias=np.zeros(3))
        result = project(np.ones((3, 2)), model)
        np.testing.assert_allclose(result, np.ones(3) * np.sqrt(200.))
        self.assertTrue((result > 1).all())
        np.testing.assert_allclose(result, project(np.ones((3, 2)) * 9, model))

    def test_fit_is_deterministic_and_safe_standardized(self):
        rows = self.records()
        values = np.random.default_rng(7).normal(size=(64, 3, 8))
        first = fit_directions(rows, values)
        second = fit_directions(rows, values)
        np.testing.assert_array_equal(first['weight'], second['weight'])
        scores = project(values, first)
        np.testing.assert_allclose(scores[::2].mean(0), np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(scores[::2].std(0), np.ones(3), atol=1e-12)
        self.assertEqual(first['hold_threshold'], float(np.quantile(scores[::2].std(-1), .95)))

    def test_fit_rejects_test_and_duplicate_checkpoints(self):
        rows = self.records()
        values = np.ones((64, 3, 8))
        for field, value in (('role', 'test'), ('label_source', 'request_risk')):
            invalid = copy.deepcopy(rows)
            invalid[0][field] = value
            with self.assertRaises(ValueError):
                fit_directions(invalid, values)
        with self.assertRaises(ValueError):
            fit_directions(rows + rows[:1], np.ones((65, 3, 8)))
        with self.assertRaises(ValueError):
            fit_directions(rows[:32], values[:32])

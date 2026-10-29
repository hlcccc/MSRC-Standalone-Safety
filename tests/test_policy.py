import inspect
import math
import unittest

from msrc_safety.policy import Evidence, StandaloneController, metrics, select_request_threshold


class PolicyTests(unittest.TestCase):
    def test_causal_equation_and_invalid_input_preserve_state(self):
        evidence = Evidence()
        self.assertEqual(evidence.step([1.] * 3), dict(semantic=1., temporal=.75, disagreement=0.))
        self.assertAlmostEqual(evidence.step([1.] * 3)['temporal'], 1.375)
        self.assertAlmostEqual(evidence.step([0.] * 3)['temporal'], 1.2375)
        before = evidence.mean, evidence.value
        for values in ([math.nan, 0., 0.], [math.inf] * 3, [0., 0.]):
            with self.assertRaises(ValueError):
                evidence.step(values)
            self.assertEqual((evidence.mean, evidence.value), before)

    def test_no_guard_input_and_strict_threshold(self):
        self.assertEqual(list(inspect.signature(StandaloneController.step).parameters), ['self', 'values'])
        controller = StandaloneController(.75, .5)
        self.assertFalse(controller.step([1.] * 3)['alarm'])
        decision = controller.step([1.] * 3)
        self.assertTrue(decision['alarm'])
        self.assertEqual((decision['generated'], decision['released'], decision['pending']), (2, 0, 0))
        with self.assertRaises(RuntimeError):
            controller.finish()
        with self.assertRaises(RuntimeError):
            controller.step([0.] * 3)

    def test_hold_hysteresis_buffer_and_finish(self):
        controller = StandaloneController(100., .5)
        for _ in range(70):
            decision = controller.step([-1., 0., 1.])
        self.assertEqual(decision['state'], 'HOLD')
        self.assertEqual((decision['released'], decision['pending']), (6, 64))
        self.assertEqual(controller.step([-.4, 0., .4])['state'], 'HOLD')
        decision = controller.step([0.] * 3)
        self.assertEqual(decision['state'], 'NORMAL')
        self.assertEqual((decision['released'], decision['pending']), (56, 16))
        self.assertEqual(controller.finish()['released'], 72)
        with self.assertRaises(RuntimeError):
            controller.finish()

    def test_alarm_keeps_only_earlier_released_tokens(self):
        controller = StandaloneController(1., .5)
        for _ in range(20):
            controller.step([0.] * 3)
        self.assertEqual(controller.released, 4)
        alarm = controller.step([4.] * 3)
        self.assertTrue(alarm['alarm'])
        self.assertEqual((alarm['released'], alarm['pending']), (4, 0))

    def test_calibration_role_and_tie_break(self):
        rows = [dict(id=str(i), group=str(i), role='calibration', label=i % 2,
                     score=float(i % 2)) for i in range(40)]
        selection = select_request_threshold(rows)
        self.assertEqual(selection['threshold'], 0.)
        self.assertEqual(selection['calibration']['accuracy'], 1.)
        self.assertFalse(selection['test_used'])
        rows[0]['role'] = 'test'
        with self.assertRaisesRegex(ValueError, 'calibration'):
            select_request_threshold(rows)

    def test_small_or_duplicate_calibration_is_rejected(self):
        rows = [dict(id=str(i), group=str(i), role='calibration', label=i % 2,
                     score=float(i % 2)) for i in range(40)]
        with self.assertRaises(ValueError):
            select_request_threshold(rows[:20])
        rows[-1]['id'] = rows[0]['id']
        with self.assertRaises(ValueError):
            select_request_threshold(rows)

    def test_imbalanced_all_alarm_baseline(self):
        value = metrics([dict(label=int(i < 30), warn=True) for i in range(40)])
        self.assertEqual(value['accuracy'], .75)
        self.assertEqual(value['balanced_accuracy'], .5)
        self.assertEqual(value['benign_false_positive_rate'], 1.)
        self.assertIsNone(metrics([])['accuracy'])

    def test_threshold_and_buffer_validation(self):
        for bad in (-1, math.inf, math.nan, True):
            with self.assertRaises(ValueError):
                StandaloneController(bad, .5)
        with self.assertRaises(ValueError):
            StandaloneController(1., .5, base_buffer=1)

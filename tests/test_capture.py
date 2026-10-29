import unittest

import numpy as np

from msrc_safety.generation import LayerCapture


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.ndim, self.shape = self.values.ndim, self.values.shape

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def detach(self):
        return self

    def clone(self):
        return Tensor(self.values.copy())

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class Torch:
    @staticmethod
    def stack(values):
        return Tensor(np.stack([value.values for value in values]))


class Block:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        class Handle:
            def remove(handle):
                self.hooks.remove(hook)
        return Handle()


class Decoder:
    def __init__(self):
        self.layers = [Block() for _ in range(28)]


class CaptureTests(unittest.TestCase):
    def test_only_three_last_token_vectors_are_copied(self):
        decoder = Decoder()
        with LayerCapture(decoder) as capture:
            self.assertEqual([i + 1 for i, layer in enumerate(decoder.layers) if layer.hooks], [14, 21, 28])
            for number in capture.layers:
                value = Tensor(np.arange(12).reshape(1, 3, 4) + number)
                decoder.layers[number - 1].hooks[0](None, (), (value,))
                value.values.fill(-999)
            result = capture.result(Torch)
            self.assertEqual(result.shape, (3, 4))
            np.testing.assert_array_equal(result[0], np.arange(8, 12) + 14)
        self.assertFalse(any(layer.hooks for layer in decoder.layers))

    def test_hooks_removed_after_error_and_missing_capture_fails(self):
        decoder = Decoder()
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with LayerCapture(decoder) as capture:
                with self.assertRaises(ValueError):
                    capture.result(Torch)
                raise RuntimeError('test failure')
        self.assertFalse(any(layer.hooks for layer in decoder.layers))

import unittest
import numpy as np

from diffusionkit.mlx import DiffusionPipeline

class TestCFGEuler(unittest.TestCase):
    def setUp(self):
        self.pipe = DiffusionPipeline()

    def _mean_abs(self, cfg):
        image, _ = self.pipe.generate_image(
            text="photo of a cat",
            num_steps=1,
            cfg_weight=cfg,
            seed=0,
            negative_text="",
            verbose=False,
        )
        arr = np.array(image).astype(np.float32) / 255.0
        return np.mean(np.abs(arr))

    def test_cfg_effect(self):
        m0 = self._mean_abs(0)
        m7 = self._mean_abs(7)
        self.assertLess(m0, 0.05)
        self.assertGreater(m7, 0.20)

if __name__ == "__main__":
    unittest.main()

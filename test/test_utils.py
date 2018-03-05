from utils import resize_batch, makeGenerator
import numpy as np

class TestUtils:
	def test_resize_batch_1(self):
		data = np.random.randn(10, 50, 50, 3)
		newData = resize_batch(data, (25, 25, 3))
		assert newData.shape == (10, 25, 25, 3)

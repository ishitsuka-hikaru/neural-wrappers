from neural_wrappers.utilities import resize_batch
import numpy as np

class TestUtils:
	def test_resize_batch_1(self):
		data = np.random.randn(10, 50, 50, 3)
		newData = resize_batch(data, height=25, width=25)
		assert newData.shape == (10, 25, 25, 3)

	def test_resize_batch_2(self):
		data = np.random.randn(10, 50, 50, 3)
		newData = resize_batch(data, height=25, width=25, mode="black_bars")
		assert newData.shape == (10, 25, 25, 3)

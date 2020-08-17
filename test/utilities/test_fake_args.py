from neural_wrappers.utilities import FakeArgs, npCloseEnough
import numpy as np

class TestRunningMean:
	def test_FakeArgs_1(self):
		arr = np.random.randn(10, 20)
		args = FakeArgs({"test" : 1, "npArray" : arr})

		assert args.test == 1
		assert npCloseEnough(args.npArray, arr)
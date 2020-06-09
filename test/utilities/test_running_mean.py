from neural_wrappers.utilities import RunningMean
import numpy as np

class TestRunningMean:
	def test_RunningMean_update_1(self):
		rm = RunningMean(0)
		for i in range(10):
			rm.update(i)
		assert abs(rm.get() - 4.5) < 1e-5

	def test_RunningMean_update_2(self):
		rm = RunningMean([0, 0])
		for i in range(10):
			rm.update([i, i + 1])
		assert np.abs(rm.get() - [4.5, 5.5]).sum() < 1e-5

	def test_RunningMean_update_3(self):
		rm = RunningMean({"a" : 0, "b" : 0})
		for i in range(10):
			rm.update({"b" : i, "a" : i + 1})
		assert np.abs(np.array(list(rm.get().values())) - [5.5, 4.5]).sum() < 1e-5

	def test_RunningMean_updateBatch_1(self):
		rm = RunningMean(0)
		rm.updateBatch([1,2,3])
		assert abs(rm.get() - 2) < 1e-5

	def test_RunningMean_updateBatch_2(self):
		rm = RunningMean([0, 0])
		rm.updateBatch(np.arange(8).reshape((4, 2)))
		assert np.abs(rm.get() - [3, 4]).sum() < 1e-4
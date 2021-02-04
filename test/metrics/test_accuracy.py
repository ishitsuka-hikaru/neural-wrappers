import numpy as np
from neural_wrappers.metrics import accuracy
from neural_wrappers.utilities import toCategorical, npCloseEnough

class TestPrecision:
	def test_call_accuracy_1(self):
		results = toCategorical([0, 1, 2, 3, 1], numClasses=5)
		labels = toCategorical([0, 0, 2, 2, 3], numClasses=5)
		expected = np.array([1, 0, 1, 0, 0], dtype=np.bool)

		res = accuracy(results, labels)
		assert npCloseEnough(res, expected)

	def test_call_accuracy_2(self):
		results = toCategorical([0, 1, 2, 3, 1], numClasses=5)
		labels = toCategorical([0, 0, 2, 2, 3], numClasses=5)

		res1 = accuracy(results, labels)
		res2 = accuracy(results, labels, meanResult=True)
		assert res1.mean() == 0.4
		assert res2 == 0.4

	# # Example from: https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
	# def test_call_accuracy_2(self):
	# 	# Cat:0, Fish:1, Hen: 2
	# 	results = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
	# 	labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0, 1, 1, 0, 1, 1, 2, 2, 2, 2, 2, 2]
	# 	results = toCategorical(results, numClasses=3)
	# 	labels = toCategorical(labels, numClasses=3)

	# 	res = precision(results, labels)
	# 	assert npCloseEnough(res, np.array([0.58051]))

def main():
	# TestPrecision().test_compute_precision1()
	# TestPrecision().test_call_accuracy_1()
	# TestPrecision().test_call_accuracy_2()
	pass

if __name__ == "__main__":
	main()
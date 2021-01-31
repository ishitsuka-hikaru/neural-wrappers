import numpy as np
from neural_wrappers.metrics import f1score
from neural_wrappers.utilities import toCategorical, npCloseEnough

class TestF1Score:
	# Example from: https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
	def test_call_f1score_1(self):
		# Cat:0, Fish:1, Hen: 2
		results = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
		labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0, 1, 1, 0, 1, 1, 2, 2, 2, 2, 2, 2]
		results = toCategorical(results, numClasses=3)
		labels = toCategorical(labels, numClasses=3)

		res = f1score(results, labels)
		assert npCloseEnough(res, np.array([0.464129]))

def main():
	# TestF1Score().test_call_f1score_1()
	pass

if __name__ == "__main__":
	main()
import numpy as np

# Generic metric class
class Metric:
	def __init__(self):
		pass

	def __call__(self, results, labels, **kwargs):
		raise NotImplementedError("Should have implemented this")

	def __str__(self):
		return "Generic neural network metric"

class Loss(Metric):
	def __call__(self, results, labels, **kwargs):
		return kwargs["loss"]

class Accuracy(Metric):
	def __call__(self, results, labels, **kwargs):
		predicted = np.argmax(results, axis=1)
		correct = np.sum(predicted == labels)
		total = labels.shape[0]
		accuracy = correct / total * 100
		return accuracy
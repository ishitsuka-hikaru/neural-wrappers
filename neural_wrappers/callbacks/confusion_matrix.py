import numpy as np
from copy import deepcopy
from .callback import Callback
from sklearn.metrics import confusion_matrix

class ConfusionMatrix(Callback):
	def __init__(self, numClasses, categoricalLabels=True, printMatrix=False, **kwargs):
		name = "ConfusionMatrix" if not "name" in kwargs else kwargs["name"]
		super().__init__(name=name)
		self.numClasses = numClasses
		self.nClasses = np.arange(self.numClasses)
		self.categoricalLabels = categoricalLabels
		self.printMatrix = printMatrix
		self.epochMatrix = {
			"Train" : np.zeros((numClasses, numClasses), dtype=np.int32),
			"Validation" : np.zeros((numClasses, numClasses), dtype=np.int32),
			"Test" : np.zeros((numClasses, numClasses), dtype=np.int32)
		}

	@staticmethod
	def computeMeanAcc(cfMatrix):
		# Sum the rows (TP + FP)
		TPFN = np.sum(cfMatrix, axis=-1)
		# TP are on diagonal of confusion matrix
		TP = np.diag(cfMatrix)
		return 100 * (TP / TPFN).mean()

	@staticmethod
	def computeMeanF1(cfMatrix):
		TP = np.diag(cfMatrix)
		FN = np.sum(cfMatrix, axis=-1) - TP
		FP = np.sum(cfMatrix, axis=0) - TP
		P = TP / (TP + FP + 1e-5)
		R = TP / (TP + FN + 1e-5)
		F1 = 2 * P * R / (P + R + 1e-5)
		return F1.mean()

	def onEpochStart(self, **kwargs):
		# Reset the confusion matrix for the next epoch
		for Key in self.epochMatrix:
			self.epochMatrix[Key] *= 0

	def onEpochEnd(self, **kwargs):
		if kwargs["isTraining"]:
			print("%s (validation)\n%s" % (self.name, self.epochMatrix["Validation"]))

			# Accuracy
			accTrain = ConfusionMatrix.computeMeanAcc(self.epochMatrix["Train"])
			accValidation = ConfusionMatrix.computeMeanAcc(self.epochMatrix["Validation"])
			print("True accuracy. Train: %2.2f%%. Validation: %2.2f%%" % (accTrain, accValidation))

			# F1
			F1Train = ConfusionMatrix.computeMeanF1(self.epochMatrix["Train"])
			F1Validation = ConfusionMatrix.computeMeanF1(self.epochMatrix["Validation"])
			print("True F1 score. Train: %2.2f. Validation: %2.2f" % (F1Train, F1Validation))
		else:
			F1Test = ConfusionMatrix.computeMeanF1(self.epochMatrix["Test"])
			accTest = ConfusionMatrix.computeMeanAcc(self.epochMatrix["Test"])
			print("%s (test)\n%s" % (self.name, self.epochMatrix["Test"]))
			print("True F1 score: %2.2f" % (F1Test))
			print("True accuracy: %2.2f%%" % (accTest))

		if kwargs["isTraining"]:
			kwargs["trainHistory"][-1]["Train"][self.name] = deepcopy(self.epochMatrix["Train"])
			kwargs["trainHistory"][-1]["Validation"][self.name] = deepcopy(self.epochMatrix["Validation"])

			# Update F1 and Accuracy as well with their better values (even if these metrics might not be used or if
			#  they are updated later.
			kwargs["trainHistory"][-1]["Train"]["Accuracy"] = accTrain
			kwargs["trainHistory"][-1]["Validation"]["Accuracy"] = accValidation
			kwargs["trainHistory"][-1]["Train"]["F1Score"] = F1Train
			kwargs["trainHistory"][-1]["Validation"]["F1Score"] = F1Validation

		return self.epochMatrix

	def onIterationEnd(self, results, labels, **kwargs):
		if kwargs["isTraining"]:
			Key = "Train" if kwargs["isOptimizing"] else "Validation"
		else:
			Key = "Test"
		results = np.argmax(results, axis=1)
		if self.categoricalLabels:
			labels = np.where(labels == 1)[1]
		self.epochMatrix[Key] += confusion_matrix(labels, results, self.nClasses)
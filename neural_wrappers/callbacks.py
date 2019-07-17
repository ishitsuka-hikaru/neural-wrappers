import os
import sys
import numpy as np
import torch as tr
from copy import deepcopy, copy
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/pytorch")
from pytorch_utils import plotModelMetricHistory

class Callback:
	def __init__(self, name=None):
		if name is None:
			name = str(self)
		self.name = name

	def onEpochStart(self, **kwargs):
		pass

	def onEpochEnd(self, **kwargs):
		pass

	def onIterationStart(self, **kwargs):
		pass

	def onIterationEnd(self, results, labels, **kwargs):
		pass

	# Some callbacks requires some special/additional tinkering when loading a neural network model from a pickle
	#  binary file (i.e scheduler callbacks must update the optimizer using the new model, rather than the old one).
	#  @param[in] additional Usually is the same as returned by onCallbackSave (default: None)
	def onCallbackLoad(self, additional, **kwargs):
		pass

	# Some callbacks require some special/additional tinkering when saving (such as closing files). It should be noted
	#  that it's safe to close files (or any other side-effect action) because callbacks are deepcopied before this
	#  method is called (in saveModel)
	def onCallbackSave(self, **kwargs):
		pass

# This class is used to convert metrics to callbacks which are called at each iteration. This is done so we unify
#  metrics and callbacks in one way. Stats and iteration messages can be computed for both cases thanks to this.
class MetricAsCallback(Callback):
	def __init__(self, metricName, metric):
		super().__init__(metricName)
		self.metric = metric

	def onIterationEnd(self, results, labels, **kwargs):
		return self.metric(results, labels, **kwargs)

# TODO: add format to saving files
class SaveHistory(Callback):
	def __init__(self, fileName, mode="write", **kwargs):
		super().__init__(**kwargs)
		assert mode in ("write", "append")
		self.mode = "w" if mode == "write" else "a"
		self.fileName = fileName
		self.file = None

	def onEpochEnd(self, **kwargs):
		# SaveHistory should be just in training mode.
		if not kwargs["trainHistory"]:
			print("Warning! Using SaveHistory callback with no history (probably testing mode).")
			return

		if kwargs["epoch"] == 1:
			self.file = open(self.fileName, mode=self.mode, buffering=1)
			self.file.write(kwargs["model"].summary() + "\n")
		message = kwargs["trainHistory"][-1]["message"]
		self.file.write(message + "\n")

	def onCallbackSave(self, **kwargs):
		self.file.close()
		self.file = None

	def onCallbackLoad(self, additional, **kwargs):
		# Make sure we're appending to the file now that we're using a loaded model (to not overwrite previous info).
		self.file = open(self.fileName, mode="a", buffering=1)

# TODO: add format to saving files
class SaveModels(Callback):
	def __init__(self, type="all", metric="Loss", metricDirection="min", **kwargs):
		super().__init__(**kwargs)
		assert type in ("all", "improvements", "last", "best")
		self.type = type
		self.best = float("nan")
		self.metric = metric
		assert metricDirection in ("min", "max")
		self.metricDirection = metricDirection

	# Saving by best train loss is validation is not available, otherwise validation. Nasty situation can occur if one
	#  epoch there is a validation loss and the next one there isn't, so we need formats to avoid this and error out
	#  nicely if the format asks for validation loss and there's not validation metric reported.
	def onEpochEnd(self, **kwargs):
		if not kwargs["isTraining"]:
			return

		trainHistory = kwargs["trainHistory"][-1]
		metricFunc = (lambda x, y : x < y) if self.metricDirection == "min" else (lambda x, y : x > y)
		Key = "Validation" if self.metric in trainHistory["Validation"] else Train
		score = trainHistory[Key][self.metric]

		fileName = "model_weights_%d_%s_%2.2f.pkl" % (kwargs["epoch"], self.metric, score)
		if self.type == "improvements":
			# nan != nan is True
			if self.best != self.best or metricFunc(score, self.best):
				kwargs["model"].saveModel(fileName)
				sys.stdout.write("Epoch %d. Improvement (%s) from %2.2f to %2.2f\n" % \
					(kwargs["epoch"], self.metric, self.best, score))
				self.best = score
			else:
				sys.stdout.write("Epoch %d did not improve best metric (%s: %2.2f)\n" % \
					(kwargs["epoch"], self.emtric, self.best))
			sys.stdout.flush()
		elif self.type == "all":
			kwargs["model"].saveModel(fileName)
		elif self.type == "last":
			kwargs["model"].saveModel("model_last_%s.pkl" % (self.metric))
		elif self.type == "best":
			# nan != nan is True
			if self.best != self.best or metricFunc(score, self.best):
				kwargs["model"].saveModel("model_best_%s.pkl" % (self.metric))
				sys.stdout.write("Epoch %d. Improvement (%s) from %2.2f to %2.2f\n" % \
					(kwargs["epoch"], self.metric, self.best, score))
				self.best = score

# Used to save self-supervised models.
class SaveModelsSelfSupervised(SaveModels):
	def __init__(self, type="all", **kwargs):
		super().__init__(**kwargs)
		self.name = "SaveModelsSelfSupervised"

	def onEpochEnd(self, **kwargs):
		model = deepcopy(kwargs["model"]).cpu()
		model.setPretrainMode(False)
		kwargs["model"] = model
		super().onEpochEnd(**kwargs)

class ConfusionMatrix(Callback):
	def __init__(self, numClasses, categoricalLabels=True, printMatrix=False, **kwargs):
		name = "ConfusionMatrix" if not "name" in kwargs else kwargs["name"]
		super().__init__(name=name)
		self.numClasses = numClasses
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

		return self.epochMatrix

	def onIterationEnd(self, results, labels, **kwargs):
		if kwargs["isTraining"]:
			Key = "Train" if kwargs["isOptimizing"] else "Validation"
		else:
			Key = "Test"
		results = np.argmax(results, axis=1)
		if self.categoricalLabels:
			labels = np.where(labels == 1)[1]
		for i in range(len(labels)):
			self.epochMatrix[Key][labels[i], results[i]] += 1

class PlotMetricsCallback(Callback):
	def __init__(self, metrics, plotBestBullet=None, dpi=120, **kwargs):
		super().__init__(**kwargs)
		assert len(metrics) > 0, "Expected a list of at least one metric which will be plotted."
		self.metrics = metrics
		self.dpi = dpi
		self.plotBestBullet = plotBestBullet
		if self.plotBestBullet == None:
			self.plotBestBullet = ["none"] * len(self.metrics)

	def onEpochEnd(self, **kwargs):
		trainHistory = kwargs["trainHistory"]
		if not kwargs["isTraining"] or len(trainHistory) == 1:
			return

		for i, metric in enumerate(self.metrics):
			bullet = self.plotBestBullet[i]
			plotModelMetricHistory(metric, trainHistory, bullet)

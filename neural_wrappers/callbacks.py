import os
import sys
import numpy as np
from copy import deepcopy
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/pytorch")
from utils import plotModelHistory

class Callback:
	def __init__(self):
		pass

	def onEpochStart(self, **kwargs):
		pass

	def onEpochEnd(self, **kwargs):
		pass

	def onIterationStart(self, **kwargs):
		pass

	def onIterationEnd(self, **kwargs):
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

	def __str__(self):
		return "Generic neural network callback"

# TODO: add format to saving files
class SaveHistory(Callback):
	def __init__(self, fileName, mode="write"):
		assert mode in ("write", "append")
		mode = "w" if mode == "write" else "a"
		self.fileName = fileName
		self.file = open(fileName, mode=mode, buffering=1)

	def onEpochEnd(self, **kwargs):
		if kwargs["epoch"] == 1:
			self.file.write(kwargs["model"].summary() + "\n")
		message = kwargs["model"].computePrintMessage(**kwargs)
		self.file.write(message + "\n")

	def onCallbackSave(self, **kwargs):
		self.file.close()
		self.file = None

	def onCallbackLoad(self, additional, **kwargs):
		# Make sure we're appending to the file now that we're using a loaded model (to not overwrite previous info).
		self.file = open(self.fileName, mode="a", buffering=1)

# TODO: add format to saving files
class SaveModels(Callback):
	def __init__(self, type="all", metric="Loss", metricDirection="min"):
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
		metricFunc = (lambda x, y : x < y) if self.metricDirection == "min" else (lambda x, y : x > y)
		metrics = (kwargs["validationMetrics"] if kwargs["validationMetrics"] != None else kwargs["trainMetrics"])
		score = metrics[self.metric]

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
	def __init__(self, type="all"):
		super().__init__(type)

	def onEpochEnd(self, **kwargs):
		model = deepcopy(kwargs["model"]).cpu()
		model.setPretrainMode(False)
		kwargs["model"] = model
		super().onEpochEnd(**kwargs)

class ConfusionMatrix(Callback):
	def __init__(self, numClasses, categoricalLabels):
		self.numClasses = numClasses
		self.categoricalLabels = categoricalLabels
		self.confusionMatrix = np.zeros((numClasses, numClasses), dtype=np.int32)

	def onEpochStart(self, **kwargs):
		# Reset the confusion matrix for the next epoch
		self.confusionMatrix *= 0

	def onEpochEnd(self, **kwargs):
		# Add to history dictionary
		if not kwargs["trainHistory"] is None:
			kwargs["trainHistory"]["confusionMatrix"] = np.copy(self.confusionMatrix)
		print("\nMatrix:", self.confusionMatrix)

	def onIterationEnd(self, **kwargs):
		results = np.argmax(kwargs["results"], axis=1)
		if self.categoricalLabels:
			labels = np.where(kwargs["labels"] == 1)[1]
		else:
			labels = kwargs["labels"]
		for i in range(len(labels)):
			self.confusionMatrix[labels[i], results[i]] += 1

class PlotMetricsCallback(Callback):
	def __init__(self, metrics, plotBestBullet=None, dpi=120):
		assert len(metrics) > 0, "Expected a list of at least one metric which will be plotted."
		self.metrics = metrics
		self.dpi = dpi
		self.plotBestBullet = plotBestBullet
		if self.plotBestBullet == None:
			self.plotBestBullet = ["none"] * len(self.metrics)

	def onEpochEnd(self, **kwargs):
		trainHistory = kwargs["model"].trainHistory
		for i in range(len(self.metrics)):
			plotModelHistory(trainHistory, self.metrics[i], self.plotBestBullet[i], self.dpi)
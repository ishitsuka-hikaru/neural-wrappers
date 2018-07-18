import sys
import numpy as np
from neural_wrappers.utilities import isBaseOf
from copy import deepcopy

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

	# Some callbacks require some special/additional tinkering when saving (such as closing files). It shoul be noted
	#  that it's safe to close files (or any other side-effect action) because callbacks are deepcopied before this
	#  method is called (is saveModel)
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
	def __init__(self, type="all"):
		assert type in ("all", "improvements", "last", "best")
		self.type = type
		self.best = float("nan")

	# Saving by best train loss is validation is not available, otherwise validation. Nasty situation can occur if one
	#  epoch there is a validation loss and the next one there isn't, so we need formats to avoid this and error out
	#  nicely if the format asks for validation loss and there's not validation metric reported.
	def onEpochEnd(self, **kwargs):
		loss = (kwargs["validationMetrics"] if kwargs["validationMetrics"] != None else kwargs["trainMetrics"])["Loss"]
		fileName = "model_weights_%d_%2.2f.pkl" % (kwargs["epoch"], loss)
		if self.type == "improvements":
			# nan != nan is True
			if self.best != self.best or loss < self.best:
				kwargs["model"].saveModel(fileName)
				sys.stdout.write("Epoch %d. Improvement from %2.2f to %2.2f\n" % (kwargs["epoch"], self.best, loss))
				self.best = loss
			else:
				sys.stdout.write("Epoch %d did not improve best loss (%2.2f)\n" % (kwargs["epoch"], self.best))
			sys.stdout.flush()
		elif self.type == "all":
			kwargs["model"].saveModel(fileName)
		elif self.type == "last":
			if kwargs["epoch"] == kwargs["numEpochs"]:
				kwargs["model"].saveModel(fileName)
		elif self.type == "best":
			# nan != nan is True
			if self.best != self.best or loss < self.best:
				kwargs["model"].saveModel("model_best.pkl")
				sys.stdout.write("Epoch %d. Improvement from %2.2f to %2.2f\n" % (kwargs["epoch"], self.best, loss))
				self.best = loss

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

	def onIterationEnd(self, **kwargs):
		results = np.argmax(kwargs["results"], axis=1)
		if self.categoricalLabels:
			labels = np.where(kwargs["labels"] == 1)[1]
		else:
			labels = kwargs["labels"]
		for i in range(len(labels)):
			self.confusionMatrix[labels[i], results[i]] += 1
import sys
import numpy as np

class Callback:
	def __init__(self):
		pass

	def onEpochStart(self, **kwargs):
		pass

	def onEpochEnd(self, **kwargs):
		pass

	def onIterationEnd(self, **kwargs):
		pass

	def __str__(self):
		return "Generic neural network callback"

# TODO: add format to saving files
class SaveHistory(Callback):
	def __init__(self, fileName):
		self.fileName = fileName
		self.file = open(fileName, "w", buffering=1)

	def onEpochEnd(self, **kwargs):
		if kwargs["epoch"] == 1:
			self.file.write(kwargs["model"].summary() + "\n")

		done = kwargs["epoch"] / kwargs["numEpochs"] * 100
		metrics = kwargs["validationMetrics"] if kwargs["validationMetrics"] != None else kwargs["trainMetrics"]
		message = "Epoch %d/%d. Done: %2.2f%%." % (kwargs["epoch"], kwargs["numEpochs"], done)

		for metric in sorted(kwargs["trainMetrics"]):
			message += " %s: %2.2f." % (metric, kwargs["trainMetrics"][metric])

		if kwargs["validationMetrics"] != None:
			for metric in sorted(kwargs["validationMetrics"]):
				message += " %s: %2.2f." % (metric, kwargs["validationMetrics"][metric])
		self.file.write(message + "\n")

	def write(self, message):
		sys.stdout.write(message + "\n")
		sys.stdout.flush()
		self.file.write(message + "\n")

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
				kwargs["model"].save_model(fileName)
				sys.stdout.write("Epoch %d. Improvement from %2.2f to %2.2f\n" % (kwargs["epoch"], self.best, loss))
				self.best = loss
			else:
				sys.stdout.write("Epoch %d did not improve best loss (%2.2f)\n" % (kwargs["epoch"], self.best))
			sys.stdout.flush()
		elif self.type == "all":
			kwargs["model"].save_model(fileName)
		elif self.type == "last":
			if kwargs["epoch"] == kwargs["numEpochs"]:
				kwargs["model"].save_model(fileName)
		elif self.type == "best":
			# nan != nan is True
			if self.best != self.best or loss < self.best:
				kwargs["model"].save_model("model_best.pkl")
				sys.stdout.write("Epoch %d. Improvement from %2.2f to %2.2f\n" % (kwargs["epoch"], self.best, loss))
				self.best = loss

# TODO: add parameters if values are not one-hot encoded (for now it's assumed for both results and labels)
class ConfusionMatrix(Callback):
	def __init__(self, numClasses):
		self.numClasses = numClasses
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
		labels = np.where(kwargs["labels"] == 1)[1]
		for i in range(len(labels)):
			self.confusionMatrix[labels[i], results[i]] += 1
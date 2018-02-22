import sys

class Callback:
	def __init__(self):
		pass

	def __call__(self, **kwargs):
		raise NotImplementedError("Should have implemented this")

	def __str__(self):
		return "Generic neural network callback"

# TODO: add format to saving files
class SaveHistory(Callback):
	def __init__(self, fileName):
		self.fileName = fileName
		self.file = open(fileName, "w", buffering=1)

	def __call__(self, **kwargs):
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
		assert type in ("all", "improvements")
		self.type = type
		self.best = float("nan")

	# Saving by best train loss is validation is not available, otherwise validation. Nasty situation can occur if one
	#  epoch there is a validation loss and the next one there isn't, so we need formats to avoid this and error out
	#  nicely if the format asks for validation loss and there's not validation metric reported.
	def __call__(self, **kwargs):
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
		else:
			kwargs["model"].save_model(fileName)

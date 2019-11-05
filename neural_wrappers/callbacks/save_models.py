import sys
from .callback import Callback

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
		Key = "Validation" if self.metric in trainHistory["Validation"] else "Train"
		score = trainHistory[Key][self.metric]

		fileName = "model_weights_%d_%s_%2.2f.pkl" % (kwargs["epoch"], self.metric, score)
		if self.type == "improvements":
			# nan != nan is True
			if self.best != self.best or metricFunc(score, self.best):
				kwargs["model"].saveModel(fileName)
				print("Epoch %d. Improvement (%s) from %2.2f to %2.2f" % \
					(kwargs["epoch"], self.metric, self.best, score))
				self.best = score
			else:
				print("Epoch %d did not improve best metric (%s: %2.2f)" % \
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
				print("Epoch %d. Improvement (%s) from %2.2f to %2.2f" % \
					(kwargs["epoch"], self.metric, self.best, score))
				self.best = score
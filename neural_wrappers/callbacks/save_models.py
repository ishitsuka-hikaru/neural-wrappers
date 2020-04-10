import sys
from .callback import Callback

# TODO: add format to saving files
# Note: This callback should be called after all (relevant) callbacks were called, otherwise we risk of storing a model
#  that hasn't updated all it's callbacks. This is relevant, for example in EarlyStopping, where we'd save the state
#  of the N-1th epoch instead of last, causing it to lead to different behavioiur pre/post loading.
class SaveModels(Callback):
	def __init__(self, type="all", metric="Loss", **kwargs):
		assert type in ("all", "improvements", "last", "best")
		self.type = type
		self.best = float("nan")
		self.metric = metric
		super().__init__(**kwargs)

	# Saving by best train loss is validation is not available, otherwise validation. Nasty situation can occur if one
	#  epoch there is a validation loss and the next one there isn't, so we need formats to avoid this and error out
	#  nicely if the format asks for validation loss and there's not validation metric reported.
	def onEpochEnd(self, **kwargs):
		if not kwargs["isTraining"]:
			return

		trainHistory = kwargs["trainHistory"][-1]
		metricDirection = kwargs["model"].getMetrics()[self.metric].getDirection()
		metricFunc = (lambda x, y : x < y) if metricDirection == "min" else (lambda x, y : x > y)

		if (not "Validation" in trainHistory) or (trainHistory["Validation"] is None):
			Key = "Train"
		else:
			Key = "Validation"
		score = trainHistory[Key][self.metric]

		fileName = "model_weights_%d_%s_%2.2f.pkl" % (kwargs["epoch"], self.metric, score)
		if self.type == "improvements":
			# nan != nan is True
			if self.best != self.best or metricFunc(score, self.best):
				self.best = score
				kwargs["model"].saveModel(fileName)
				print("[SaveModels] Epoch %d. Improvement (%s) from %2.2f to %2.2f" % \
					(kwargs["epoch"], self.metric, self.best, score))
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
				self.best = score
				kwargs["model"].saveModel("model_best_%s.pkl" % (self.metric))
				print("[SaveModels] Epoch %d. Improvement (%s) from %2.2f to %2.2f" % \
					(kwargs["epoch"], self.metric, self.best, score))

	def __str__(self):
		return "SaveModels (Metric: %s. Type: %s)" % (self.metric, self.type)
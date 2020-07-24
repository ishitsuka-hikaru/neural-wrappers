import sys
from .callback import Callback
from ..metrics import Metric, MetricWrapper
from typing import Union, Tuple, Optional

# TODO: add format to saving files
# Note: This callback should be called after all (relevant) callbacks were called, otherwise we risk of storing a model
#  that hasn't updated all it's callbacks. This is relevant, for example in EarlyStopping, where we'd save the state
#  of the N-1th epoch instead of last, causing it to lead to different behavioiur pre/post loading.
class SaveModels(Callback):
	def __init__(self, mode:str = "all", metricName:Union[str, Tuple[str]] = "Loss", \
		metricDirecion:str = "min", **kwargs):
		assert mode in ("all", "improvements", "last", "best")
		self.mode = mode
		if type(metricName) == str:
			metricName = (metricName, )
		self.metricName = metricName
		self.metricDirection = metricDirecion
		self.best = {
			"min" : 1<<31,
			"max" : -1<<31
		}[self.metricDirection]
		self.metricFunc = {
			"min" : lambda x : x < self.best,
			"max" : lambda x : x > self.best,
		}[self.metricDirection]
		super().__init__(**kwargs)

	def saveModelsImprovements(self, score, **kwargs):
		try:
			if not self.metricFunc(score):
				return
		except Exception:
			# TODO: Add comparison function for metrics
			print("Skipping metric %s because it doesn't return a comparable number.")
			return
		fileName = "model_weights_%d_%s_%s.pkl" % (kwargs["epoch"], self.metricName, score)
		kwargs["model"].saveModel(fileName)
		print("[SaveModels] Epoch %d. Improvement (%s) from %2.2f to %2.2f" % \
				(kwargs["epoch"], self.metricName, self.best, score))
		self.best = score

	def saveModelsBest(self, score, **kwargs):
		if not self.metricFunc(score):
			return
		fileName = "model_weights_best_%s.pkl" % (str(self.metricName))
		kwargs["model"].saveModel(fileName)
		print("[SaveModels] Epoch %d. Improvement (%s) from %2.2f to %2.2f" % \
				(kwargs["epoch"], self.metricName, self.best, score))
		self.best = score

	def saveModelsLast(self, **kwargs):
		fileName = "model_last.pkl"
		kwargs["model"].saveModel(fileName)
		print("[SaveModels] Epoch %d. Saved last model" % (kwargs["epoch"]))

	# Saving by best train loss is validation is not available, otherwise validation. Nasty situation can occur if one
	#  epoch there is a validation loss and the next one there isn't, so we need formats to avoid this and error out
	#  nicely if the format asks for validation loss and there's not validation metric reported.
	def onEpochEnd(self, **kwargs):
		if not kwargs["isTraining"]:
			return

		trainHistory = kwargs["trainHistory"][-1]
		if (not "Validation" in trainHistory) or (trainHistory["Validation"] is None):
			trainHistory = trainHistory["Train"]
		else:
			trainHistory = trainHistory["Validation"]
		score = trainHistory
		for k in self.metricName:
			score = score[k]

		fileName = "model_weights_%d_%s_%s.pkl" % (kwargs["epoch"], self.metricName, score)
		if self.mode == "improvements":
			self.saveModelsImprovements(score, **kwargs)
		elif self.mode == "best":
			self.saveModelsBest(score, **kwargs)
		elif self.mode == "last":
			self.saveModelsLast(**kwargs)
		else:
			assert False
		# 	# nan != nan is True
		# 	if self.best != self.best or metricFunc(score, self.best):
		# 		kwargs["model"].saveModel(fileName)
		# 		print("[SaveModels] Epoch %d. Improvement (%s) from %2.2f to %2.2f" % \
		# 			(kwargs["epoch"], strMetric, self.best, score))
		# 		self.best = score
		# 	else:
		# 		print("Epoch %d did not improve best metric (%s: %2.2f)" % \
		# 			(kwargs["epoch"], strMetric, self.best))
		# 	sys.stdout.flush()
		# elif self.type == "all":
		# 	kwargs["model"].saveModel(fileName)
		# elif self.type == "last":
		# 	kwargs["model"].saveModel("model_last_%s.pkl" % (strMetric))
		# elif self.type == "best":
		# 	# nan != nan is True
		# 	if self.best != self.best or metricFunc(score, self.best):
		# 		kwargs["model"].saveModel("model_best_%s.pkl" % (strMetric))
		# 		print("[SaveModels] Epoch %d. Improvement (%s) from %2.2f to %2.2f" % \
		# 			(kwargs["epoch"], strMetric, self.best, score))
		# 		self.best = score

	def onCallbackLoad(self, additional, **kwargs):
		self.metricFunc = {
			"min" : lambda x : x < self.best,
			"max" : lambda x : x > self.best,
		}[self.metricDirection]

	# Some callbacks require some special/additional tinkering when saving (such as closing files). It should be noted
	#  that it's safe to close files (or any other side-effect action) because callbacks are deepcopied before this
	#  method is called (in saveModel)
	def onCallbackSave(self, **kwargs):
		self.metricFunc = None

	def __str__(self):
		return "SaveModels (Metric: %s. Type: %s)" % (str(self.metricName), self.mode)
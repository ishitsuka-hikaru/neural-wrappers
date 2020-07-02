import sys
from .callback import Callback
from .callback_name import CallbackName
from ..metrics import Metric, MetricWrapper
from typing import Union, Tuple, Optional

# TODO: add format to saving files
# Note: This callback should be called after all (relevant) callbacks were called, otherwise we risk of storing a model
#  that hasn't updated all it's callbacks. This is relevant, for example in EarlyStopping, where we'd save the state
#  of the N-1th epoch instead of last, causing it to lead to different behavioiur pre/post loading.
class SaveModels(Callback):
	def __init__(self, mode:str, metricName:CallbackName, **kwargs):
		assert mode in ("all", "improvements", "last", "best")
		self.mode = mode
		self.metricName = metricName
		self.best = None
		self.metricFunc = None
		super().__init__(**kwargs)

	# Do the setup at the end of the first epoch, so we know direction. This is expected to not change ever afterwards.
	def setup(self, model):
		if not self.best is None:
			return

		metric = model.getMetric(self.metricName)
		metricDirection = metric.getDirection()

		self.best = {
			"min" : 1<<31,
			"max" : -1<<31
		}[metricDirection]
		self.metricFunc = {
			"min" : lambda x : x < self.best,
			"max" : lambda x : x > self.best,
		}[metricDirection]

	def saveModelsImprovements(self, score, **kwargs):
		if not self.metricFunc(score):
			return
		metricName = self.metricName[0] if len(self.metricName) == 1 else self.metricName
		fileName = "model_improvement_%d_%s_%s.pkl" % (kwargs["epoch"], str(metricName), score)
		kwargs["model"].saveModel(fileName)
		print("[SaveModels] Epoch %d. Improvement (%s) from %2.2f to %2.2f" % \
				(kwargs["epoch"], self.metricName, self.best, score))
		self.best = score

	def saveModelsBest(self, score, **kwargs):
		if not self.metricFunc(score):
			return
		fileName = "model_best_%s.pkl" % (self.metricName)
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
		self.setup(kwargs["model"])

		trainHistory = kwargs["trainHistory"][-1]
		if (not "Validation" in trainHistory) or (trainHistory["Validation"] is None):
			trainHistory = trainHistory["Train"]
		else:
			trainHistory = trainHistory["Validation"]

		score = trainHistory[self.metricName]
		fileName = "model_weights_%d_%s_%s.pkl" % (kwargs["epoch"], self.metricName, score)
		if self.mode == "improvements":
			self.saveModelsImprovements(score, **kwargs)
		elif self.mode == "best":
			self.saveModelsBest(score, **kwargs)
		elif self.mode == "last":
			self.saveModelsLast(**kwargs)
		else:
			assert False

	def onCallbackLoad(self, additional, **kwargs):
		metric = kwargs["model"].getMetric(self.metricName)
		metricDirection = metric.getDirection()

		self.metricFunc = {
			"min" : lambda x : x < self.best,
			"max" : lambda x : x > self.best,
		}[metricDirection]

	# Some callbacks require some special/additional tinkering when saving (such as closing files). It should be noted
	#  that it's safe to close files (or any other side-effect action) because callbacks are deepcopied before this
	#  method is called (in saveModel)
	def onCallbackSave(self, **kwargs):
		self.metricFunc = None

	def __str__(self):
		return "SaveModels (Metric: %s. Type: %s)" % (str(self.metricName), self.mode)
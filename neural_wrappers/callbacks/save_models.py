import sys
import numpy as np
from typing import Union, Tuple, Optional
from overrides import overrides

from .callback import Callback
from .callback_name import CallbackName
from tqdm import tqdm

# TODO: add format to saving files
# Note: This callback should be called after all (relevant) callbacks were called, otherwise we risk of storing a model
#  that hasn't updated all it's callbacks. This is relevant, for example in EarlyStopping, where we'd save the state
#  of the N-1th epoch instead of last, causing it to lead to different behavioiur pre/post loading.
class SaveModels(Callback):
	def __init__(self, mode:str, metricName:CallbackName, **kwargs):
		assert mode in ("all", "improvements", "last", "best")
		self.mode = mode
		self.metricName = CallbackName(metricName)
		self.best = None
		self.metric = None
		# self.metricFunc = None
		super().__init__(**kwargs)

	# Do the setup at the end of the first epoch, so we know direction. This is expected to not change ever afterwards.
	def setup(self, model):
		if not self.best is None:
			return

		self.metric = model.getMetric(self.metricName)
		direction = self.metric.getDirection()
		extremes = self.metric.getExtremes()
		self.best = {
			"max" : extremes["min"],
			"min" : extremes["max"]
		}[direction]

	def saveModelsImprovements(self, score, **kwargs):
		if not self.metric.compareFunction(score, self.best):
			return
		fileName = "model_improvement_%d_%s_%s.pkl" % (kwargs["epoch"], self.metricName, score)
		kwargs["model"].saveModel(fileName)
		tqdm.write("[SaveModels] Epoch %d. Improvement (%s) from %s to %s" % \
				(kwargs["epoch"], self.metricName, self.best, score))
		self.best = score

	def saveModelsBest(self, score, **kwargs):
		res = self.metric.compareFunction(score, self.best)
		if not res:
			return
		fileName = "model_best_%s.pkl" % (self.metricName)
		kwargs["model"].saveModel(fileName)
		tqdm.write("[SaveModels] Epoch %d. Improvement (%s) from %s to %s" % \
				(kwargs["epoch"], self.metricName, self.best, score))
		self.best = score

	def saveModelsLast(self, **kwargs):
		fileName = "model_last.pkl"
		kwargs["model"].saveModel(fileName)
		tqdm.write("[SaveModels] Epoch %d. Saved last model" % (kwargs["epoch"]))

	# Saving by best train loss is validation is not available, otherwise validation. Nasty situation can occur if one
	#  epoch there is a validation loss and the next one there isn't, so we need formats to avoid this and error out
	#  nicely if the format asks for validation loss and there's not validation metric reported.
	@overrides
	def onEpochEnd(self, **kwargs):
		from ..pytorch import getMetricScoreFromHistory
		if not kwargs["isTraining"]:
			return
		self.setup(kwargs["model"])

		trainHistory = kwargs["trainHistory"][-1]
		if (not "Validation" in trainHistory) or (trainHistory["Validation"] is None):
			trainHistory = trainHistory["Train"]
		else:
			trainHistory = trainHistory["Validation"]

		score = getMetricScoreFromHistory(trainHistory, self.metricName)
		fileName = "model_weights_%d_%s_%s.pkl" % (kwargs["epoch"], self.metricName, score)
		if self.mode == "improvements":
			self.saveModelsImprovements(score, **kwargs)
		elif self.mode == "best":
			self.saveModelsBest(score, **kwargs)
		elif self.mode == "last":
			self.saveModelsLast(**kwargs)
		else:
			assert False

	@overrides
	def onEpochStart(self, **kwargs):
		pass

	@overrides
	def onIterationStart(self, **kwargs):
		pass

	@overrides
	def onIterationEnd(self, results, labels, **kwargs):
		pass

	@overrides
	def onCallbackLoad(self, additional, **kwargs):
		self.metric = kwargs["model"].getMetric(self.metricName)

	@overrides
	def onCallbackSave(self, **kwargs):
		self.metric = None

	@overrides
	def __str__(self):
		return "SaveModels (Metric: %s. Type: %s)" % (str(self.metricName), self.mode)
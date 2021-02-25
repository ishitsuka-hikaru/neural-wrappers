import sys
import numpy as np
from typing import Union, Tuple, Optional
from overrides import overrides

from .callback import Callback
from .callback_name import CallbackName
from ..utilities import Debug, deepDictGet

# TODO: add format to saving files
# Note: This callback should be called after all (relevant) callbacks were called, otherwise we risk of storing a model
#  that hasn't updated all it's callbacks. This is relevant, for example in EarlyStopping, where we'd save the state
#  of the N-1th epoch instead of last, causing it to lead to different behavioiur pre/post loading.
class SaveModels(Callback):
	def __init__(self, mode:str, metricName:str, **kwargs):
		assert mode in ("all", "improvements", "last", "best")
		self.mode = mode
		if isinstance(metricName, Callback):
			metricName = metricName.getName()
		self.metricName = CallbackName(metricName)
		self.best = None
		super().__init__(**kwargs)

	def saveImprovements(self, model, metric, score, epoch):
		if self.best is None:
			direction = metric.getDirection()
			extremes = metric.getExtremes()
			self.best = {
				"max" : extremes["min"],
				"min" : extremes["max"]
			}[direction]

		compareResult = self.metric.compareFunction(score, self.best)
		if compareResult == False:
			Debug.log(2, "[SaveModels] Epoch %d. Metric %s did not improve best score %s with %s" % \
				(epoch, metricName, self.best, score))
			return

		Debug.log(2, "[SaveModels::saveImprovements] Epoch %d. Metric %s improved best score from %s to %s" % \
			(epoch, metricName, self.best, score))
		self.best = score
		model.saveModel("model_improvement_%s_epoch-%d_score-%s" % (self.metricName, epoch, score))

	def saveBest(self, model, metric, score, epoch):
		if self.best is None:
			direction = metric.getDirection()
			extremes = metric.getExtremes()
			self.best = {
				"max" : extremes["min"],
				"min" : extremes["max"]
			}[direction]

		compareResult = metric.compareFunction(score, self.best)
		if compareResult == False:
			Debug.log(2, "[SaveModels] Epoch %d. Metric %s did not improve best score %s with %s" % \
				(epoch, self.metricName, self.best, score))
			return

		Debug.log(2, "[SaveModels::saveBest] Epoch %d. Metric %s improved best score from %s to %s" % \
			(epoch, self.metricName, self.best, score))
		self.best = score
		model.saveModel("model_best_%s.pkl" % self.metricName)
	
	def saveLast(self, model, metric, score, epoch):
		model.saveModel("model_last.pkl")
		Debug.log(2, "[SaveModels::saveLast] Epoch %d. Saved last model." % epoch)

	# Saving by best train loss is validation is not available, otherwise validation. Nasty situation can occur if one
	#  epoch there is a validation loss and the next one there isn't, so we need formats to avoid this and error out
	#  nicely if the format asks for validation loss and there's not validation metric reported.
	@overrides
	def onEpochEnd(self, **kwargs):
		model = kwargs["model"]
		trainHistory = kwargs["trainHistory"][-1]
		epoch = kwargs["epoch"]

		if not kwargs["isTraining"]:
			return

		metric = model.getMetric(self.metricName)
		Key = "Validation" if "Validation" in trainHistory and (not trainHistory["Validation"] is None) else "Train"
		trainHistory = trainHistory[Key]
		score = deepDictGet(trainHistory, self.metricName.name)

		f = {
			"improvements" : self.saveImprovements,
			"best" : self.saveBest,
			"last" : self.saveLast
		}[self.mode]
		f(model, metric, score, epoch)

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
		pass

	@overrides
	def onCallbackSave(self, **kwargs):
		pass

	@overrides
	def __str__(self):
		return "SaveModels (Metric: %s. Type: %s)" % (str(self.metricName), self.mode)
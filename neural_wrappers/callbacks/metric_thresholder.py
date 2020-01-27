import numpy as np
import torch as tr
import matplotlib.pyplot as plt
from .callback import Callback
from ..utilities import npCloseEnough, RunningMean

# @brief MetricThresholder is a class that takes as inputs a metric and a list of thresholds. The provided metric must
#  be callable and not be agnosti to the parameter "threshold", which will be used to compute the metric for each
#  provided threshold (i.e. Accuracy for prediction>0.1, >0.3, >0.5...>1)
class MetricThresholder(Callback):
	def __init__(self, metric, thresholds):
		super().__init__()
		self.metric = metric
		self.thresholds = np.array(thresholds)
		assert npCloseEnough(self.thresholds, np.sort(self.thresholds)), "Thresholds must be an ordered range."
		assert len(self.thresholds) > 1
		assert hasattr(self.metric, "__call__"), "The user provided metric %s must be callable" % (self.metric)

	def onEpochStart(self, **kwargs):
		isOptimizing = tr.is_grad_enabled()
		print("[onEpochStart] isOptimizing %d" % (isOptimizing))
		initValue = np.zeros((len(self.thresholds), ), dtype=np.float32)
		self.currentResult = RunningMean(initValue=initValue)

	def onIterationEnd(self, results, labels, **kwargs):
		isOptimizing = tr.is_grad_enabled()
		if isOptimizing:
			return

		MB = results.shape[0]
		iterResults = []
		for threshold in self.thresholds:
			result = self.metric(results, labels, threshold=threshold) * MB
			iterResults.append(result)
		self.currentResult.update(iterResults, MB)
	
	def onEpochEnd(self, **kwargs):
		values = self.currentResult.get()[1 : ]
		diffs = self.thresholds[1 : ] - self.thresholds[0 : -1]
		AUC = np.dot(values, diffs)

		plt.figure()
		plt.plot(self.thresholds, self.currentResult.get(), marker="x")
		plt.ylim(0, 1)
		plt.title("AUC: %2.3f" % (AUC))
		plt.savefig("metric_thresholder_%s_epoch%d.png" % (self.metric, kwargs["epoch"]))
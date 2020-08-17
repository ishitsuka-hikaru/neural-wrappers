# import numpy as np
# import torch as tr
# import matplotlib.pyplot as plt
# from inspect import getfullargspec

# from .callback import Callback
# from .metric_as_callback import MetricAsCallback
# from ..utilities import RunningMean, npCloseEnough, isBaseOf

# # @brief MetricThresholder is a class that takes as inputs a metric and a list of thresholds. The provided metric must
# #  be callable and not be agnosti to the parameter "threshold", which will be used to compute the metric for each
# #  provided threshold (i.e. Accuracy for prediction>0.1, >0.3, >0.5...>1)
# class MetricThresholder(Callback):
# 	def __init__(self, metricName, metric, thresholds, ylim=[0, 1]):
# 		super().__init__()
# 		self.metricName = metricName
# 		self.metric = MetricThresholder.setupMetric(metricName, metric)
# 		self.thresholds = np.array(thresholds)
# 		self.ylim = ylim
# 		assert npCloseEnough(self.thresholds, np.sort(self.thresholds)), "Thresholds must be an ordered range."
# 		assert len(self.thresholds) > 1

# 		self.currentKey = None
# 		self.currentResult = {}

# 	def getCurrentKey(self):
# 		currentKey = "Train" if tr.is_grad_enabled() else "Validation"
# 		# Reset, if we're switching from grad to non-grad (Train vs val)
# 		if currentKey != self.currentKey:
# 			self.currentKey = currentKey
# 			initValue = np.zeros((len(self.thresholds), ), dtype=np.float32)
# 			self.currentResult[self.currentKey] = RunningMean(initValue=initValue)
# 		return currentKey

# 	def onIterationEnd(self, results, labels, **kwargs):
# 		Key = self.getCurrentKey()

# 		MB = results.shape[0]
# 		iterResults = []
# 		# This is just so I don't shoot myself in the leg. Theoretically, I can do some thresholds that are outside,
# 		#  however in most of the cases it's possible I just messed up something.
# 		for threshold in self.thresholds:
# 			result = self.metric(results, labels, threshold=threshold) * MB
# 			iterResults.append(result)
# 		self.currentResult[Key].update(iterResults, MB)

# 	def onEpochEnd(self, **kwargs):
# 		y = self.thresholds
# 		plt.figure()
# 		for key in self.currentResult:
# 			x = self.currentResult[key].get()
# 			AUC = MetricThresholder.computeAUC(x, y)
# 			label = "AUC (%s): %2.3f" % (key, AUC)
# 			MetricThresholder.doPlot(x, y, self.metricName, self.ylim, label)
# 		plt.legend()
# 		plt.title("Metric thresholder %s. Epoch %d" % (self.metricName, kwargs["epoch"]))
# 		figureName = "metric_thresholder_%s_epoch%d.png" % (self.metricName, kwargs["epoch"])
# 		plt.savefig(figureName)

# 	@staticmethod
# 	def computeAUC(x, y):
# 		diffs = x[1 : ] - x[0 : -1]
# 		values = y[1 : ]
# 		AUC = np.dot(values, diffs)
# 		return AUC

# 	@staticmethod
# 	def setupMetric(metricName, metric):
# 		assert hasattr(metric, "__call__"), "The user provided metric %s must be callable" % (metricName)
# 		assert "threshold" in getfullargspec(metric.__call__).args, \
# 			"The use provided metric %s must have threshold as a kwarg." % (metricName)
# 		if not isBaseOf(metric, MetricAsCallback):
# 			metric = MetricAsCallback(metricName=metricName, metric=metric)
# 		return metric

# 	@staticmethod
# 	def doPlot(x, y, metricName, ylim, label):
# 		plt.plot(x, y, marker="x", label=label)
# 		plt.ylim(*ylim)
# 		plt.ylabel(metricName)
# 		plt.xlabel("Thresholds (%d total)" % (len(x)))
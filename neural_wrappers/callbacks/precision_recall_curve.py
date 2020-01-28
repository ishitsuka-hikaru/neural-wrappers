from neural_wrappers.metrics import Metric, ThresholdPrecisionRecall, ThresholdAccuracy
from neural_wrappers.callbacks import MetricThresholder
from neural_wrappers.utilities import RunningMean
import matplotlib.pyplot as plt
import torch as tr
import numpy as np

class PrecisionRecallCurve(MetricThresholder):
	def __init__(self, thresholds, ylim=[0, 1]):
		super().__init__("PrecisionRecallCurcve", ThresholdPrecisionRecall(), thresholds, ylim)
	
	def onEpochStart(self, **kwargs):
		isOptimizing = tr.is_grad_enabled()
		print("[onEpochStart] isOptimizing %d" % (isOptimizing))
		initValue = np.zeros((len(self.thresholds), 2), dtype=np.float32)
		self.currentResult = RunningMean(initValue=initValue)

	def onEpochEnd(self, **kwargs):
		res = self.currentResult.get()
		precision, recall = res[:, 0], res[:, 1]
		print(precision)
		print(recall)
		exit()
		figureName = "metric_thresholder_%s_epoch%d.png" % (self.metricName, kwargs["epoch"])
		plt.plot(x, y)
		plt.show()
		exit()
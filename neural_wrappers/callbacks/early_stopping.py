import sys
import numpy as np
from .callback import Callback

# TODO: Remove mode from ctor and infer at epoch end.
class EarlyStopping(Callback):
	def __init__(self, metric="Loss", min_delta=0, patience=10, percentage=False):
		self.metric = metric
		self.min_delta = min_delta
		self.patience = patience
		self.percentage = percentage
		super().__init__()

		self.bestMetricScore = None
		self.numBadEpochs = 0
		self.metricDirection = None

	def onEpochEnd(self, **kwargs):
		trainHistory = kwargs["trainHistory"][-1]
		Key = "Validation" if "Validation" in trainHistory else "Train"
		score = trainHistory[Key][self.metric]
		assert not np.isnan(score)

		# First epoch we need to get some value running.
		if self.bestMetricScore is None:
			self.numBadEpochs = 0
			self.bestMetricScore = score
			self.metricDirection = kwargs["model"].getMetrics()[self.metric].getDirection()
			assert self.metricDirection in ("min", "max")
			return

		fIsBetter = EarlyStopping._init_is_better(self.metricDirection, self.patience, self.percentage, self.min_delta)
		if fIsBetter(score, self.bestMetricScore):
			self.numBadEpochs = 0
			self.bestMetricScore = score
		else:
			self.numBadEpochs += 1
			print("[EarlyStopping] Early Stopping is being applied. Num bad in a row: %d. Patience: %d" % \
				(self.numBadEpochs, self.patience))

		if self.numBadEpochs >= self.patience:
			print("[EarlyStopping] Num bad epochs in a row: %d. Stopping the training!" % (self.numBadEpochs))
			sys.exit(0)

	def _init_is_better(metricDirection, patience, modePercentage, minDelta):
		if patience == 0:
			return lambda a, best: True
		fDirection = lambda a, b, c : (a < b - c if metricDirection == "min" else a > b + c)
		if modePercentage:
			minDelta = best * minDelta / 100
		return lambda a, best : fDirection(a, best, minDelta)

	def __str__(self):
		return "EarlyStopping (Metric: %s. Min delta: %2.2f. Patience: %d. Percentage: %s)" \
			% (self.metric, self.min_delta, self.patience, self.percentage)
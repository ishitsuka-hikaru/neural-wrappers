import sys
import numpy as np
from .callback import Callback

class EarlyStopping(Callback):
	def __init__(self, metric="Loss", mode="min", min_delta=0, patience=10, percentage=False):
		super().__init__()
		assert mode in ("min", "max"), "Mode %s is unknown!" % (mode)
		self.metric = metric
		self.mode = mode
		self.min_delta = min_delta
		self.patience = patience
		self.percentage = percentage

		self.bestMetricScore = None
		self.numBadEpochs = 0
		self.fIsBetter = self._init_is_better()

	def onEpochEnd(self, **kwargs):
		trainHistory = kwargs["trainHistory"][-1]
		Key = "Validation" if "Validation" in trainHistory else "Train"
		score = trainHistory[Key][self.metric]
		assert not np.isnan(score)

		# First epoch we need to get some value running.
		if self.bestMetricScore is None:
			self.numBadEpochs = 0
			self.bestMetricScore = score
			return

		if self.fIsBetter(score, self.bestMetricScore):
			self.numBadEpochs = 0
			self.bestMetricScore = score
		else:
			print("[EarlyStopping] Early Stopping is being applied. Num bad in a row: %d. Patience: %d" % \
				(self.numBadEpochs, self.patience))
			self.numBadEpochs += 1

		if self.numBadEpochs >= self.patience:
			print("[EarlyStopping] Num bad epochs in a row: %d. Stopping the training!" % (self.numBadEpochs))
			sys.exit(0)

	def onCallbackLoad(self, additional, **kwargs):
		self.fIsBetter = self._init_is_better()

	def onCallbackSave(self, **kwargs):
		self.fIsBetter = None

	def _init_is_better(self):
		if self.patience == 0:
			return lambda a, best: True

		if not self.percentage:
			if self.mode == "min":
				return lambda a, best: a < best - self.min_delta
			if self.mode == "max":
				return lambda a, best: a > best + self.min_delta
		else:
			if self.mode == "min":
				return lambda a, best: a < best - (best * self.min_delta / 100)
			if self.mode == "max":
				return lambda a, best: a > best + (best * self.min_delta / 100)
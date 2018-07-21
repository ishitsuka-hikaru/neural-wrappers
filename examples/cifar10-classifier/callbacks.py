import matplotlib.pyplot as plt
import numpy as np
from neural_wrappers.callbacks import Callback
from torch.optim.lr_scheduler import ReduceLROnPlateau

class PlotLossCallback(Callback):
	def onEpochEnd(self, **kwargs):
		epoch = kwargs["epoch"]
		numEpochs = kwargs["numEpochs"]
		# Do everything at last epoch
		if epoch < numEpochs:
			return

		trainLoss, valLoss = [], []
		trainAcc, valAcc = [], []
		trainHistory = kwargs["model"].trainHistory
		for epoch in range(len(trainHistory)):
			trainLoss.append(trainHistory[epoch]["trainMetrics"]["Loss"])
			valLoss.append(trainHistory[epoch]["validationMetrics"]["Loss"])
			trainAcc.append(trainHistory[epoch]["trainMetrics"]["Accuracy"])
			valAcc.append(trainHistory[epoch]["validationMetrics"]["Accuracy"])

		x = np.arange(len(trainLoss)) + 1

		plt.figure()
		plt.plot(x, trainLoss, label="Train loss")
		plt.plot(x, valLoss, label="Val loss")
		plt.legend()
		plt.savefig("loss.png")

		plt.figure()
		plt.plot(x, trainAcc, label="Train accuracy")
		plt.plot(x, valAcc, label="Val accuracy")
		plt.legend()
		plt.savefig("accuracy.png")

class SchedulerCallback(Callback):
	def __init__(self, optimizer):
		self.scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=4, eps=1e-4)

	def onEpochEnd(self, **kwargs):
		if not kwargs["validationMetrics"]:
			loss = kwargs["trainMetrics"]["Loss"]
		else:
			loss = kwargs["validationMetrics"]["Loss"]
		self.scheduler.step(loss)

	def onCallbackLoad(self, additional, **kwargs):
		self.scheduler.optimizer = kwargs["model"].optimizer

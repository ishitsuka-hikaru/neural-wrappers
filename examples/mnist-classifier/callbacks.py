import matplotlib.pyplot as plt
import numpy as np
from neural_wrappers.callbacks import Callback
from torch.optim.lr_scheduler import ReduceLROnPlateau

class PlotLossCallback(Callback):
	def __init__(self):
		self.trainLoss = []
		self.valLoss = []
		self.trainAcc = []
		self.valAcc = []

	def onEpochEnd(self, **kwargs):
		epoch = kwargs["epoch"]
		numEpochs = kwargs["numEpochs"]
		# Do everything at last epoch
		if epoch < numEpochs:
			return

		trainHistory = kwargs["model"].trainHistory
		for epoch in range(len(trainHistory)):
			self.trainLoss.append(trainHistory[epoch]["trainMetrics"]["Loss"])
			self.valLoss.append(trainHistory[epoch]["validationMetrics"]["Loss"])
			self.trainAcc.append(trainHistory[epoch]["trainMetrics"]["Accuracy"])
			self.valAcc.append(trainHistory[epoch]["validationMetrics"]["Accuracy"])

		x = np.arange(len(self.trainLoss)) + 1

		plt.figure()
		plt.plot(x, self.trainLoss, label="Train loss")
		plt.plot(x, self.valLoss, label="Val loss")
		plt.legend()
		plt.savefig("loss.png")

		plt.figure()
		plt.plot(x, self.trainAcc, label="Train accuracy")
		plt.plot(x, self.valAcc, label="Val accuracy")
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
# Implementation of the classical MNIST FC and Conv classifier
import sys
import numpy as np
import torch as tr
import torch.nn as nn
import torch.optim as optim
from neural_wrappers.readers import MNISTReader
from neural_wrappers.pytorch import NeuralNetworkPyTorch, maybeCuda
from neural_wrappers.callbacks import SaveModels, ConfusionMatrix, Callback
from neural_wrappers.metrics import Loss, Accuracy
from functools import partial
import matplotlib.pyplot as plt

class ModelFC(NeuralNetworkPyTorch):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(28 * 28, 100)
		self.fc2 = nn.Linear(100, 100)
		self.fc3 = nn.Linear(100, 10)

	def forward(self, x):
		x = x.view(-1, 28 * 28)
		y1 = self.fc1(x)
		y2 = self.fc2(y1)
		y3 = self.fc3(y2)
		y4 = nn.functional.softmax(y3, dim=1)
		return y4

class ModelConv(NeuralNetworkPyTorch):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=3, stride=1)
		self.conv2 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, stride=1)
		self.fc1 = nn.Linear(24 * 24 * 100, 100)
		self.fc2 = nn.Linear(100, 10)

	def forward(self, x):
		x = x.view(-1, 1, 28, 28)
		y1 = self.conv1(x)
		y2 = self.conv2(y1)
		y2 = y2.view(-1, 24 * 24 * 100)
		y3 = self.fc1(y2)
		y4 = self.fc2(y3)
		y5 = nn.functional.softmax(y4, dim=1)
		return y5

class PlotLossCallback(Callback):
	def __init__(self):
		self.trainLoss = []
		self.valLoss = []
		self.trainAcc = []
		self.valAcc = []

	def onEpochEnd(self, **kwargs):
		self.trainLoss.append(kwargs["trainMetrics"]["Loss"])
		self.valLoss.append(kwargs["validationMetrics"]["Loss"])
		self.trainAcc.append(kwargs["trainMetrics"]["Accuracy"])
		self.valAcc.append(kwargs["validationMetrics"]["Accuracy"])

	def __del__(self):
		x = np.arange(len(self.trainLoss))

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

def main():
	assert len(sys.argv) >= 4, "Usage: python main.py <train/test> <model_fc/model_conv> <path/to/mnist.h5> [model]"
	assert sys.argv[1] in ("train", "test")
	assert sys.argv[2] in ("model_fc", "model_conv")

	model = maybeCuda(ModelFC() if sys.argv[2] == "model_fc" else ModelConv())
	model.setOptimizer(optim.SGD, lr=0.01, momentum=0.3)
	model.setMetrics({"Loss" : Loss(), "Accuracy" : Accuracy(categoricalLabels=True)})
	# Negative log-likeklihood (used for softmax+NLL for classification), expecting targets are one-hot encoded
	model.setCriterion(lambda y, t : tr.mean(-tr.log(y[t] + 1e-5)))
	print(model.summary())

	reader = MNISTReader(sys.argv[3])
	testGenerator = reader.iterate("test", miniBatchSize=5)
	testNumIterations = reader.getNumIterations("test", miniBatchSize=5)
	confusionMatrixCallback = ConfusionMatrix(numClasses=10)

	if sys.argv[1] == "train":
		assert len(sys.argv) == 4
		trainGenerator = reader.iterate("train", miniBatchSize=20)
		trainNumIterations = reader.getNumIterations("train", miniBatchSize=20)

		callbacks = [SaveModels(type="best"), confusionMatrixCallback, PlotLossCallback()]
		model.train_generator(trainGenerator, stepsPerEpoch=trainNumIterations, numEpochs=10, callbacks=callbacks, \
			validationGenerator=testGenerator, validationSteps=testNumIterations)
	elif sys.argv[1] == "test":
		assert len(sys.argv) == 5
		model.load_weights(sys.argv[4])
		metrics = model.test_generator(testGenerator, testNumIterations, callbacks=[confusionMatrixCallback])

		loss, accuracy = metrics["Loss"], metrics["Accuracy"]
		print("Testing complete. Loss: %2.2f. Accuracy: %2.2f" % (loss, accuracy))
		print("Confusion matrix:\n", confusionMatrixCallback.confusionMatrix)

if __name__ == "__main__":
	main()
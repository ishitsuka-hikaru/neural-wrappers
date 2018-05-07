# Implementation of the classical MNIST FC and Conv classifier
import sys
import numpy as np
import torch as tr
import torch.nn as nn
import torch.optim as optim
from neural_wrappers.readers import MNISTReader, Cifar10Reader
from neural_wrappers.pytorch import NeuralNetworkPyTorch, maybeCuda
from neural_wrappers.callbacks import SaveModels, ConfusionMatrix, Callback, SaveHistory
from neural_wrappers.metrics import Loss, Accuracy
import matplotlib.pyplot as plt

class ModelFC(NeuralNetworkPyTorch):
	def __init__(self, inputShape=(28, 28, 1), outputShape=10):
		super().__init__()

		self.outputShape = outputShape
		self.inputShape = inputShape
		self.inputShapeProd = inputShape[0] * inputShape[1] * inputShape[2]

		self.fc1 = nn.Linear(self.inputShapeProd, 100)
		self.fc2 = nn.Linear(100, 100)
		self.fc3 = nn.Linear(100, self.outputShape)

	def forward(self, x):
		x = x.view(-1, self.inputShapeProd)
		y1 = self.fc1(x)
		y2 = self.fc2(y1)
		y3 = self.fc3(y2)
		y4 = nn.functional.softmax(y3, dim=1)
		return y4

class ModelConv(NeuralNetworkPyTorch):
	def __init__(self, inputShape=(28, 28, 1), outputShape=10):
		super().__init__()

		if len(inputShape) == 2:
			inputShape = (*inputShape, 1)

		self.outputShape = outputShape
		self.inputShape = inputShape

		self.conv1 = nn.Conv2d(in_channels=inputShape[2], out_channels=100, kernel_size=3, stride=1)
		self.conv2 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, stride=1)
		self.fc1 = nn.Linear((inputShape[0] - 4) * (inputShape[1] - 4) * 100, 100)
		self.fc2 = nn.Linear(100, outputShape)

	def forward(self, x):
		x = x.view(-1, self.inputShape[2], self.inputShape[0], self.inputShape[1])
		y1 = self.conv1(x)
		y2 = self.conv2(y1)
		y2 = y2.view(-1, (self.inputShape[0] - 4) * (self.inputShape[1] - 4) * 100)
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

def getReader(readerType, readerPath):
	if readerType == "mnist":
		reader = MNISTReader(readerPath)
		inputShape, outputShape = (28, 28, 1), 10
	else:
		reader = Cifar10Reader(readerPath)
		inputShape, outputShape = (32, 32, 3), 10
	return reader, inputShape, outputShape

def getModel(modelType, inputShape, outputShape):
	if modelType == "model_fc":
		model = ModelFC(inputShape, outputShape)
	else:
		model = ModelConv(inputShape, outputShape)
	return maybeCuda(model)

def main():
	assert len(sys.argv) >= 4, "Usage: python main.py <train/test/retrain> <model_fc/model_conv> " + \
		"<mnist/cifar10> <path/to/dataset.h5> [model]"
	assert sys.argv[1] in ("train", "test", "retrain")
	assert sys.argv[2] in ("model_fc", "model_conv")
	assert sys.argv[3] in ("mnist", "cifar10")

	reader, inputShape, outputShape = getReader(sys.argv[3], sys.argv[4])

	model = getModel(sys.argv[2], inputShape, outputShape)
	model.setOptimizer(optim.SGD, lr=0.01, momentum=0.5)
	model.setMetrics({"Accuracy" : Accuracy(categoricalLabels=True)})
	# Negative log-likeklihood (used for softmax+NLL for classification), expecting targets are one-hot encoded
	model.setCriterion(lambda y, t : tr.mean(-tr.log(y[t] + 1e-5)))
	print(model.summary())

	testGenerator = reader.iterate("test", miniBatchSize=5)
	testNumIterations = reader.getNumIterations("test", miniBatchSize=5)
	confusionMatrixCallback = ConfusionMatrix(numClasses=10)

	if sys.argv[1] == "train":
		assert len(sys.argv) == 5
		trainGenerator = reader.iterate("train", miniBatchSize=20)
		trainNumIterations = reader.getNumIterations("train", miniBatchSize=20)

		callbacks = [SaveModels(type="best"), confusionMatrixCallback, PlotLossCallback(), SaveHistory("history.txt")]
		model.train_generator(trainGenerator, stepsPerEpoch=trainNumIterations, numEpochs=10, callbacks=callbacks, \
			validationGenerator=testGenerator, validationSteps=testNumIterations)
	elif sys.argv[1] == "test":
		assert len(sys.argv) == 6
		model.load_model(sys.argv[5])
		metrics = model.test_generator(testGenerator, testNumIterations, callbacks=[confusionMatrixCallback])

		loss, accuracy = metrics["Loss"], metrics["Accuracy"]
		print("Testing complete. Loss: %2.2f. Accuracy: %2.2f" % (loss, accuracy))
		print("Confusion matrix:\n", confusionMatrixCallback.confusionMatrix)
	elif sys.argv[1] == "retrain":
		assert len(sys.argv) == 6
		model.load_model(sys.argv[5])
		trainGenerator = reader.iterate("train", miniBatchSize=20)
		trainNumIterations = reader.getNumIterations("train", miniBatchSize=20)

		callbacks = [SaveModels(type="best"), confusionMatrixCallback, PlotLossCallback(), \
			SaveHistory("history.txt", mode="append")]
		callbacks[0].best = model.trainHistory[-1]["validationMetrics"]["Loss"]
		model.train_generator(trainGenerator, stepsPerEpoch=trainNumIterations, numEpochs=10, callbacks=callbacks, \
			validationGenerator=testGenerator, validationSteps=testNumIterations)

if __name__ == "__main__":
	main()
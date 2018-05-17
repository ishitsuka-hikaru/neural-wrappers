import sys
import matplotlib.pyplot as plt
import numpy as np
import torch as tr
import torch.nn as nn
from models import ModelFC
from neural_wrappers.readers import MNISTReader, CorrupterReader
from neural_wrappers.pytorch import NeuralNetworkPyTorch, maybeCuda, SelfSupervisedNetwork, maybeCpu
from neural_wrappers.callbacks import Callback
from neural_wrappers.metrics import Accuracy
from Mihlib import plot_image, show_plots
from torch.autograd import Variable
from torch.optim import SGD
from scipy.misc import toimage

def plot_images(images, titles, gridShape):
	plt.gcf().set_size_inches((12, 8))
	for j in range(len(images)):
		image = images[j]
		plt.gcf().add_subplot(*gridShape, j + 1)
		plt.gcf().gca().axis("off")
		plt.gcf().gca().set_title(str(titles[j]))
		image = np.array(toimage(image))
		plt.imshow(image, cmap="gray")

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

class DenoisingModelFC(SelfSupervisedNetwork):
	def __init__(self):
		super().__init__(ModelFC())
		self.denoising_fc3 = nn.Linear(100, self.baseModel.inputShapeProd)

	def pretrain_forward(self, x):
		x = x.view(-1, self.baseModel.inputShapeProd)
		y1 = self.baseModel.fc1(x)
		y2 = self.baseModel.fc2(y1)
		y3 = self.denoising_fc3(y2)
		y3 = y3.view(-1, *self.baseModel.inputShape)
		return y3

def reconstructionLossFn(y, t):
	return tr.mean( (y - t)**2)

def classificationLossFn(y, t):
	return tr.mean(-tr.log(y[t] + 1e-5))

def main():
	assert len(sys.argv) == 3, "Usage: python main.py train/retrain/test/train_classifier/pretrained_train_classifier " + \
		"<path/to/mnist>"

	model = maybeCuda(DenoisingModelFC())
	mnistReader = MNISTReader(sys.argv[2])
	corrupterReader = CorrupterReader(mnistReader, corruptionPercent=20)
	# Train for self supervised reconstruction
	generator = corrupterReader.iterate("train", miniBatchSize=20, maxPrefetch=1)
	numIterations = corrupterReader.getNumIterations("train", miniBatchSize=20)
	valGenerator = corrupterReader.iterate("test", miniBatchSize=20, maxPrefetch=1)
	valNumIterations = corrupterReader.getNumIterations("test", miniBatchSize=20)
	model.setOptimizer(SGD, lr=0.01, momentum=0.5)

	model.setCriterion(reconstructionLossFn)

	if sys.argv[1] in ("train", "train_classifier"):
		model.train_generator(generator, numIterations, numEpochs=10, \
			validationGenerator=valGenerator, validationSteps=valNumIterations)
		model.save_model("model_reconstruction_weights.pkl")

		if sys.argv[1] == "train_classifier":
			# Pretraining done, now train for classification
			generator = mnistReader.iterate("train", miniBatchSize=20, maxPrefetch=1)
			numIterations = mnistReader.getNumIterations("train", miniBatchSize=20)
			valGenerator = mnistReader.iterate("test", miniBatchSize=20, maxPrefetch=1)
			valNumIterations = mnistReader.getNumIterations("test", miniBatchSize=20)

			model.setPretrainMode(False)
			model.setCriterion(classificationLossFn)
			model.setMetrics({"Accuracy" : Accuracy(categoricalLabels=True)})
			model.train_generator(generator, numIterations, numEpochs=10, callbacks=[PlotLossCallback()], \
				validationGenerator=valGenerator, validationSteps=valNumIterations)
			model.save_model("model_classifier_weights.pkl")

	# Load pretrained unsupervised model, just train on classification
	elif sys.argv[1] == "pretrained_train_classifier":
		# Pretraining done, now train for classification
		generator = mnistReader.iterate("train", miniBatchSize=20, maxPrefetch=1)
		numIterations = mnistReader.getNumIterations("train", miniBatchSize=20)
		valGenerator = mnistReader.iterate("test", miniBatchSize=20, maxPrefetch=1)
		valNumIterations = mnistReader.getNumIterations("test", miniBatchSize=20)

		model.load_model("model_reconstruction_weights.pkl")
		model.setPretrainMode(False)
		model.setCriterion(classificationLossFn)
		model.setMetrics({"Accuracy" : Accuracy(categoricalLabels=True)})
		model.train_generator(generator, numIterations, numEpochs=10, callbacks=[PlotLossCallback()], \
			validationGenerator=valGenerator, validationSteps=valNumIterations)
		model.save_model("model_classifier_weights.pkl")

	elif sys.argv[1] == "retrain":
		model.load_model("model_reconstruction_weights.pkl")
		model.train_generator(generator, numIterations, numEpochs=50, \
			validationGenerator=valGenerator, validationSteps=valNumIterations)
		model.save_model("model_reconstruction_weights.pkl")

	elif sys.argv[1] == "test":
		model.load_model("model_reconstruction_weights.pkl")
		for items in valGenerator:
			corrupted, original = items
			trCorrupted = Variable(maybeCuda(tr.from_numpy(corrupted)))
			trResults = model.forward(trCorrupted)
			npResults = maybeCpu(trResults.data).numpy()
			for j in range(len(npResults)):
				plot_images([original[j], corrupted[j], npResults[j]], ["Original", "Corrupted", "Result"], (1, 3))
				plt.show()

if __name__ == "__main__":
	main()
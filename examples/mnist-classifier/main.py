import sys
import h5py
import numpy as np
import torch as tr
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from functools import partial
from models import ModelFC, ModelConv
from neural_wrappers.readers import MNISTReader
from neural_wrappers.callbacks import SaveModels, SaveHistory, ConfusionMatrix, PlotMetrics, EarlyStopping
from neural_wrappers.schedulers import ReduceLRAndBacktrackOnPlateau
from neural_wrappers.utilities import getGenerators
from neural_wrappers.metrics import Accuracy
from neural_wrappers.pytorch import device

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("type")
	parser.add_argument("model_type")
	parser.add_argument("dataset_path")
	parser.add_argument("--weightsFile")
	parser.add_argument("--numEpochs", type=int, default=100)
	parser.add_argument("--batchSize", type=int, default=20)

	args = parser.parse_args()

	assert args.type in ("train", "test", "retrain")
	assert args.model_type in ("model_fc", "model_conv")
	if args.type in ("test", "retrain"):
		assert not args.weightsFile is None
	if args.type in ("train", "retrain"):
		assert not args.numEpochs is None
	return args

# Small wrapper used for current NWModule implementation of runOneEpoch. Will update when NWModule is updated.
class Reader(MNISTReader):
	def getBatchItem(self, index):
		item = super().getBatchItem(index)
		return item["data"]["images"], item["labels"]["labels"]

def lossFn(y, t):
	# Negative log-likeklihood (used for softmax+NLL for classification), expecting targets are one-hot encoded
	y = F.softmax(y, dim=1)
	t = t.type(tr.bool)
	return (-tr.log(y[t] + 1e-5)).mean()

def main():
	args = getArgs()

	reader = Reader(h5py.File(args.dataset_path, "r")["train"])
	trainReader = Reader(h5py.File(args.dataset_path, "r")["train"])
	validationReader = Reader(h5py.File(args.dataset_path, "r")["test"])

	trainGenerator, trainSteps = getGenerators(trainReader, batchSize=args.batchSize)
	validationGenerator, validationSteps = getGenerators(validationReader, batchSize=args.batchSize)
	trainReader.setBatchSize(args.batchSize)
	validationReader.setBatchSize(args.batchSize)
	print(trainReader)
	print(validationReader)
	trainGenerator = trainReader.iterateForever()
	validationGenerator = validationReader.iterateForever()

	model = {
		"model_fc" : ModelFC(inputShape=(28, 28, 1), outputNumClasses=10),
		"model_conv" : ModelConv(inputShape=(28, 28, 1), outputNumClasses=10)
	}[args.model_type].to(device)
	model.setCriterion(lossFn)
	model.addMetrics({"Accuracy" : Accuracy()})
	model.setOptimizer(optim.SGD, momentum=0.5, lr=0.1)
	model.setOptimizerScheduler(ReduceLRAndBacktrackOnPlateau(model, "Loss", 2, 10))
	callbacks = [SaveHistory("history.txt"), PlotMetrics(["Loss", "Accuracy"]), SaveModels("best", "Loss")]
	model.addCallbacks(callbacks)
	print(model.summary())

	if args.type == "train":
		model.train_generator(trainGenerator, len(trainGenerator), numEpochs=args.numEpochs, \
			validationGenerator=validationGenerator, validationSteps=len(validationGenerator))
	elif args.type == "retrain":
		model.loadModel(args.weightsFile)
		model.train_generator(trainGenerator, trainSteps, numEpochs=args.numEpochs, \
			validationGenerator=validationGenerator, validationSteps=validationSteps)
	elif args.type == "test":
		model.loadModel(args.weightsFile)
		metrics = model.test_generator(validationGenerator, validationSteps)
		print("Metrics: %s" % (metrics))

if __name__ == "__main__":
	main()
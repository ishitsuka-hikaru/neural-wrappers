import sys
import numpy as np
import torch as tr
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models import ModelFC, ModelConv
from neural_wrappers.readers import MNISTReader
from neural_wrappers.callbacks import SaveModels, SaveHistory, ConfusionMatrix, PlotMetrics, EarlyStopping
from neural_wrappers.schedulers import ReduceLRAndBacktrackOnPlateau
from neural_wrappers.utilities import getGenerators
from neural_wrappers.metrics import Accuracy
from neural_wrappers.pytorch import device

from neural_wrappers.utilities import isBaseOf

from argparse import ArgumentParser

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

def lossFn(y, t):
	# Negative log-likeklihood (used for softmax+NLL for classification), expecting targets are one-hot encoded
	y = F.softmax(y, dim=1)
	t = t.type(tr.bool)
	return (-tr.log(y[t] + 1e-5)).mean()

def main():
	args = getArgs()

	reader = MNISTReader(args.dataset_path)
	print(reader.summary())
	trainGenerator, trainSteps, valGenerator, valSteps = getGenerators(reader, \
		batchSize=args.batchSize, keys=["train", "test"])

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
		model.train_generator(trainGenerator, trainSteps, numEpochs=args.numEpochs, \
			validationGenerator=valGenerator, validationSteps=valSteps, printMessage="v2")
	elif args.type == "retrain":
		model.loadModel(args.weightsFile)
		model.train_generator(trainGenerator, trainSteps, numEpochs=args.numEpochs, \
			validationGenerator=valGenerator, validationSteps=valSteps)
	elif args.type == "test":
		model.loadModel(args.weightsFile)
		metrics = model.test_generator(valGenerator, valSteps)
		print("Metrics: %s" % (metrics))

if __name__ == "__main__":
	main()
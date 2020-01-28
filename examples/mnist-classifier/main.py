import sys
import numpy as np
import torch as tr
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from models import ModelFC, ModelConv
from neural_wrappers.readers import MNISTReader
from neural_wrappers.pytorch import maybeCuda
from neural_wrappers.callbacks import SaveModels, SaveHistory, ConfusionMatrix, PlotMetrics, EarlyStopping, PrecisionRecallCurve, MetricThresholder
from neural_wrappers.schedulers import ReduceLROnPlateau
from neural_wrappers.utilities import getGenerators
from neural_wrappers.metrics import Accuracy, F1Score, ThresholdAccuracy

from argparse import ArgumentParser

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("type")
	parser.add_argument("model_type")
	parser.add_argument("dataset_path")
	parser.add_argument("--weights_file")
	parser.add_argument("--num_epochs", type=int, default=100)
	parser.add_argument("--batch_size", type=int, default=20)

	args = parser.parse_args()

	assert args.type in ("train", "test", "retrain")
	assert args.model_type in ("model_fc", "model_conv")
	if args.type in ("test", "retrain"):
		assert not args.weights_file is None
	if args.type in ("train", "retrain"):
		assert not args.num_epochs is None
	return args

def lossFn(y, t):
	# Negative log-likeklihood (used for softmax+NLL for classification), expecting targets are one-hot encoded
	y = F.softmax(y, dim=1)
	t = t.type(tr.bool)
	return (-tr.log(y[t] + 1e-5)).mean()

def main():
	args = getArgs()

	reader = MNISTReader(args.dataset_path, normalizer={"images" : "standardization"})
	print(reader.summary())
	trainGenerator, trainSteps, valGenerator, valSteps = getGenerators(reader, \
		miniBatchSize=args.batch_size, keys=["train", "test"])

	if args.model_type == "model_fc":
		model = maybeCuda(ModelFC(inputShape=(28, 28, 1), outputNumClasses=10))
	elif args.model_type == "model_conv":
		model = maybeCuda(ModelConv(inputShape=(28, 28, 1), outputNumClasses=10))
	model.setCriterion(lossFn)
	model.addMetrics({"Accuracy" : Accuracy(), "F1" : F1Score()})
	model.setOptimizer(optim.SGD, momentum=0.5, lr=0.1)
	model.setOptimizerScheduler(ReduceLROnPlateau, metric="Loss")
	callbacks = [SaveHistory("history.txt"), PlotMetrics(["Loss", "Accuracy"]), \
		ConfusionMatrix(numClasses=10), EarlyStopping(patience=5), SaveModels("best"), MetricThresholder("ThresholdAcc", ThresholdAccuracy(), np.linspace(0, 1, 20)), PrecisionRecallCurve(np.linspace(0, 1, 10))]
	model.addCallbacks(callbacks)
	print(model.summary())

	if args.type == "train":
		model.train_generator(trainGenerator, 50, numEpochs=args.num_epochs, \
			validationGenerator=valGenerator, validationSteps=50, printMessage="v2")
	elif args.type == "retrain":
		model.loadModel(args.weights_file)
		model.train_generator(trainGenerator, trainSteps, numEpochs=args.num_epochs, \
			validationGenerator=valGenerator, validationSteps=valSteps)
	elif args.type == "test":
		model.loadModel(args.weights_file)
		metrics = model.test_generator(valGenerator, valSteps)
		print("Metrics: %s" % (metrics))

if __name__ == "__main__":
	main()
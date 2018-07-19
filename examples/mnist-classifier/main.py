import sys
import numpy as np
import torch as tr
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from models import ModelFC, ModelConv
from neural_wrappers.readers import MNISTReader
from neural_wrappers.pytorch import maybeCuda
from neural_wrappers.callbacks import SaveModels, SaveHistory, ConfusionMatrix
from callbacks import PlotLossCallback, SchedulerCallback
from neural_wrappers.metrics import Accuracy
from argparse import ArgumentParser

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("type")
	parser.add_argument("model_type")
	parser.add_argument("dataset_path")
	parser.add_argument("--weights_file")
	parser.add_argument("--num_epochs", type=int)

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
	return tr.mean(-tr.log(y[t] + 1e-5))

def main():
	args = getArgs()

	reader = MNISTReader(args.dataset_path, normalizer={"images" : "standardization"})
	print(reader.summary())
	trainGenerator = reader.iterate("train", miniBatchSize=20)
	trainSteps = reader.getNumIterations("train", miniBatchSize=20)
	valGenerator = reader.iterate("test", miniBatchSize=5)
	valSteps = reader.getNumIterations("test", miniBatchSize=5)

	if args.model_type == "model_fc":
		model = maybeCuda(ModelFC(inputShape=(28, 28, 1), outputNumClasses=10))
	elif args.model_type == "model_conv":
		model = maybeCuda(ModelConv(inputShape=(28, 28, 1), outputNumClasses=10))
	print(model.summary())
	model.setCriterion(lossFn)
	model.setMetrics({"Accuracy" : Accuracy(categoricalLabels=True)})

	if args.type == "train":
		model.setOptimizer(optim.SGD, momentum=0.5, lr=0.01)
		callbacks = [SaveHistory("history.txt"), PlotLossCallback(), SchedulerCallback(model.optimizer), \
			ConfusionMatrix(numClasses=10, categoricalLabels=True), SaveModels("best")]
		model.train_generator(trainGenerator, trainSteps, numEpochs=args.num_epochs, callbacks=callbacks, \
			validationGenerator=valGenerator, validationSteps=valSteps)
	elif args.type == "retrain":
		model.loadModel("model_best.pkl")
		model.train_generator(trainGenerator, trainSteps, numEpochs=args.num_epochs, \
			validationGenerator=valGenerator, validationSteps=valSteps)
	else:
		model.loadModel("model_best.pkl")
		callbacks = [ConfusionMatrix(numClasses=10, categoricalLabels=True)]
		metrics = model.test_generator(valGenerator, valSteps, callbacks=callbacks)
		print("Metrics: %s" % (metrics))
		confusionMatrix = model.trainHistory[-1]["confusionMatrix"]
		print("Confusion matrix:\n%s" % (confusionMatrix))

if __name__ == "__main__":
	main()
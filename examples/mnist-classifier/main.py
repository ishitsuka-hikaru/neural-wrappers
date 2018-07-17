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

def lossFn(y, t):
	# Negative log-likeklihood (used for softmax+NLL for classification), expecting targets are one-hot encoded
	y = F.softmax(y, dim=1)
	return tr.mean(-tr.log(y[t] + 1e-5))

def main():
	assert len(sys.argv) >= 4, "Usage: python main.py <train/test/retrain> <model_fc/model_conv> " + \
		"<path/to/mnist.h5> [model]"
	assert sys.argv[1] in ("train", "test", "retrain")

	reader = MNISTReader(sys.argv[3], normalizer={"images" : "standardization"})
	print(reader.summary())
	trainGenerator = reader.iterate("train", miniBatchSize=20)
	trainSteps = reader.getNumIterations("train", miniBatchSize=20)
	valGenerator = reader.iterate("test", miniBatchSize=5)
	valSteps = reader.getNumIterations("test", miniBatchSize=5)

	if sys.argv[2] == "model_fc":
		model = maybeCuda(ModelFC(inputShape=(28, 28, 1), outputNumClasses=10))
	elif sys.argv[2] == "model_conv":
		model = maybeCuda(ModelConv(inputShape=(28, 28, 1), outputNumClasses=10))
	print(model.summary())

	if sys.argv[1] == "train":
		model.setOptimizer(optim.SGD, momentum=0.5, lr=0.01)
		model.setCriterion(lossFn)
		model.setMetrics({"Accuracy" : Accuracy(categoricalLabels=True)})
		callbacks = [SaveModels("best"), SaveHistory("history.txt"), PlotLossCallback(), \
			SchedulerCallback(model.optimizer), ConfusionMatrix(numClasses=10, categoricalLabels=True)]
		model.train_generator(trainGenerator, trainSteps, numEpochs=10, callbacks=callbacks, \
			validationGenerator=valGenerator, validationSteps=valSteps)
	else:
		assert False

if __name__ == "__main__":
	main()
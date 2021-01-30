import sys
import h5py
from simple_caching import NpyFS
import numpy as np
import torch as tr
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from functools import partial
from models import ModelFC, ModelConv
from neural_wrappers.readers import MNISTReader, StaticBatchedDatasetReader, \
	RandomBatchedDatasetReader, CachedDatasetReader
from neural_wrappers.callbacks import SaveModels, SaveHistory, ConfusionMatrix, PlotMetrics, EarlyStopping, \
	RandomPlotEachEpoch
from neural_wrappers.schedulers import ReduceLRAndBacktrackOnPlateau
from neural_wrappers.utilities import changeDirectory
from neural_wrappers.metrics import Accuracy
from neural_wrappers.pytorch import device
from media_processing_lib.image import toImage
from scipy.special import softmax

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("type")
	parser.add_argument("model_type")
	parser.add_argument("dataset_path")
	parser.add_argument("--weightsFile")
	parser.add_argument("--numEpochs", type=int, default=100)
	parser.add_argument("--batchSize", type=int, default=20)
	parser.add_argument("--dir")

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

def plotFn(x, y, t):
	cnt = 0
	MB = len(y)
	x = x["images"]
	y = softmax(y, axis=-1)
	plt.figure()
	for i in range(MB):
		cnt += 1
		ix = np.argmax(t[i], axis=-1)
		thisPred = y[i][ix]
		thisImg = toImage(x[i])
		plt.imshow(thisImg)
		plt.title("Label: %s. Result: %2.3f" % (ix, thisPred))
		plt.axis("off")
		plt.savefig("%d.png" % cnt, bbox_inches="tight", pad_inches=0)
	plt.close()

def getReader(args, type):
	reader = MNISTReader(h5py.File(args.dataset_path, "r")["train"])
	reader = StaticBatchedDatasetReader(reader, args.batchSize)
	# reader = RandomBatchedDatasetReader(reader)
	reader = CachedDatasetReader(reader, cache=NpyFS(".cache/%s" % hash(reader)), buildCache=False)
	print(reader)
	return reader

def getModel(args):
	model = {
		"model_fc" : ModelFC(inputShape=(28, 28, 1), outputNumClasses=10),
		"model_conv" : ModelConv(inputShape=(28, 28, 1), outputNumClasses=10)
	}[args.model_type].to(device)
	model.setCriterion(lossFn)
	model.addMetrics({"Accuracy" : Accuracy()})
	model.setOptimizer(optim.SGD, momentum=0.5, lr=0.1)
	model.setOptimizerScheduler(ReduceLRAndBacktrackOnPlateau(model, "Loss", 2, 10))
	callbacks = [SaveHistory("history.txt"), PlotMetrics(["Loss", "Accuracy"]), SaveModels("best", "Loss"), \
		RandomPlotEachEpoch(plotFn)]
	model.addCallbacks(callbacks)
	print(model.summary())
	return model

def main():
	args = getArgs()

	trainReader = getReader(args, "train")
	validationReader = getReader(args, "test")
	trainGenerator = trainReader.iterate()
	validationGenerator = validationReader.iterate()

	model = getModel(args)

	if args.type == "train":
		changeDirectory(args.dir, expectExist=False)
		model.trainGenerator(trainGenerator, numEpochs=args.numEpochs, validationGenerator=validationGenerator)
	elif args.type == "retrain":
		model.loadModel(args.weightsFile)
		changeDirectory(args.dir)
		model.trainGenerator(trainGenerator, numEpochs=args.numEpochs, validationGenerator=validationGenerator)
	elif args.type == "test":
		model.loadModel(args.weightsFile)
		metrics = model.testGenerator(validationGenerator)
		print("Metrics: %s" % (metrics))

if __name__ == "__main__":
	main()
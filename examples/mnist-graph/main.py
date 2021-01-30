import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from argparse import ArgumentParser
from neural_wrappers.utilities import changeDirectory
from neural_wrappers.callbacks import SaveModels, PlotMetrics, RandomPlotEachEpoch
from neural_wrappers.metrics import Accuracy
from neural_wrappers.readers import StaticBatchedDatasetReader
from media_processing_lib.image import toImage

from reader import Reader
from model import getModel

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("type")
	parser.add_argument("datasetPath")

	parser.add_argument("--numEpochs", type=int, default=100)
	parser.add_argument("--batchSize", type=int, default=20)
	parser.add_argument("--dir")
	args = parser.parse_args()

	assert args.type in ("train", "plot_dataset")
	return args

def plotFn(x, y, t):
	cnt = 0
	x = x["rgb"]
	t = t["labels"]
	y = y["Result"][0]
	MB = len(y)
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

def main():
	args = getArgs()

	trainReader = StaticBatchedDatasetReader(Reader(h5py.File(args.datasetPath, "r")["train"]), args.batchSize)
	validationReader = StaticBatchedDatasetReader(Reader(h5py.File(args.datasetPath, "r")["test"]), args.batchSize)
	generator = trainReader.iterate()
	valGenerator = validationReader.iterate()

	model = getModel()
	model.setOptimizer(optim.SGD, lr=0.001)
	model.addCallbacks([PlotMetrics(["Loss"]), SaveModels("best", "Loss"), RandomPlotEachEpoch(plotFn)])
	print(model.summary())

	if args.type == "train":
		changeDirectory(args.dir, expectExist=False)
		model.trainGenerator(generator, args.numEpochs, valGenerator)
	elif args.type == "plot_dataset":
		while True:
			items = next(generator)
			data = items[0][0]
			ax = plt.subplots(1, 5)[1]
			for i, key in enumerate(list(data.keys())[0 : -1]):
				ax[i].imshow(data[key][0])
				ax[i].set_title(key)
			plt.show()
			plt.close()

if __name__ == "__main__":
	main()
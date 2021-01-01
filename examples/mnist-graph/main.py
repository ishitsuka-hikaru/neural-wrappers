import matplotlib.pyplot as plt
import torch.optim as optim
import h5py
from argparse import ArgumentParser
from neural_wrappers.utilities import getGenerators
from neural_wrappers.callbacks import SaveModels, PlotMetrics
from neural_wrappers.metrics import Accuracy
from neural_wrappers.readers import StaticBatchedDatasetReader

from reader import Reader
from model import getModel

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("type")
	parser.add_argument("datasetPath")

	parser.add_argument("--numEpochs", type=int, default=100)
	parser.add_argument("--batchSize", type=int, default=20)
	args = parser.parse_args()

	assert args.type in ("train", "plot_dataset")
	return args

def main():
	args = getArgs()

	trainReader = StaticBatchedDatasetReader(Reader(h5py.File(args.datasetPath, "r")["train"]), args.batchSize)
	validationReader = StaticBatchedDatasetReader(Reader(h5py.File(args.datasetPath, "r")["test"]), args.batchSize)
	generator = trainReader.iterate()
	valGenerator = validationReader.iterate()


	model = getModel()
	model.setOptimizer(optim.SGD, lr=0.001)
	model.addCallbacks([PlotMetrics(["Loss"]), SaveModels("best", "Loss")])
	print(model.summary())

	if args.type == "train":
		model.train_generator(generator, len(generator), args.numEpochs, valGenerator, len(valGenerator))
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
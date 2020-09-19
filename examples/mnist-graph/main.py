import matplotlib.pyplot as plt
import torch.optim as optim
from argparse import ArgumentParser
from neural_wrappers.readers import H5DatasetReader
from neural_wrappers.utilities import getGenerators
from neural_wrappers.callbacks import SaveModels, PlotMetrics
from neural_wrappers.metrics import Accuracy

from reader import Reader
from model import getModel

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("type")
	parser.add_argument("datasetPath")

	parser.add_argument("--numEpochs", type=int, default=100)
	parser.add_argument("--batchSize", type=int, default=20)
	args = parser.parse_args()

	assert args.type in ("train", )
	return args

def main():
	args = getArgs()

	reader = Reader(args.datasetPath)
	generator, numSteps, valGenerator, valNumSteps = getGenerators(reader, batchSize=args.batchSize, \
		keys=["train", "test"])

	# items = next(generator)
	# data = items[0]
	# ax = plt.subplots(5, )[1]
	# for i, key in enumerate(list(data.keys())[0 : -1]):
	# 	ax[i].imshow(data[key][0])
	# 	ax[i].set_title(key)
	# plt.show()
	# breakpoint()
	# exit()

	model = getModel()
	model.setOptimizer(optim.SGD, lr=0.001)

	model.saveModel("test.pkl")
	model.loadModel("test.pkl")
	exit()

	# graph.addCallbacks([PlotMetrics(["Loss", (str(reduceNode), "Accuracy")])])
	print(model.summary())

	if args.type == "train":
		model.train_generator(generator, numSteps, args.numEpochs, valGenerator, valNumSteps)
		pass

if __name__ == "__main__":
	main()
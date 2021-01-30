import h5py
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from functools import partial
from neural_wrappers.pytorch import GenerativeAdversarialNetwork, device
from neural_wrappers.callbacks import SaveModels, PlotMetrics, RandomPlotEachEpoch
from neural_wrappers.utilities import getGenerators, changeDirectory
from neural_wrappers.readers import StaticBatchedDatasetReader

from mnist_models import GeneratorLinear as Generator, DiscriminatorLinear as Discriminator
from reader import GANReader
from test_model import test_model

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("type")
	parser.add_argument("dataset_path")

	parser.add_argument("--dir", default="test")
	parser.add_argument("--batch_size", type=int, default=100)
	parser.add_argument("--num_epochs", type=int, default=200)
	parser.add_argument("--latent_space_size", type=int, default=200)
	parser.add_argument("--weights_file")

	args = parser.parse_args()
	assert args.type in ("train", "retrain", "test_model")

	return args

def plotFn(x, y, t, model, latentSpaceSize):
	# Generate 20 random gaussian inputs
	randomNoise = np.random.randn(20, latentSpaceSize).astype(np.float32)
	npRandomOutG = model.generator.npForward(randomNoise)[..., 0] / 2 + 0.5

	# Plot the inputs and discriminator's confidence in them
	items = [npRandomOutG[j] for j in range(len(npRandomOutG))]
	
	ax = plt.subplots(4, 5)[1]
	for i in range(4):
		for j in range(5):
			ax[i, j].imshow(npRandomOutG[i * 4 + j], cmap="gray")
	plt.savefig("results.png")
	plt.close()

def main():
	args = getArgs()

	# Define reader, generator and callbacks
	reader = StaticBatchedDatasetReader(GANReader(datasetPath=h5py.File(args.dataset_path, "r")["train"], \
		latentSpaceSize=args.latent_space_size), args.batch_size)
	generator, numIterations = getGenerators(reader)
	print(reader)

	generatorModel = Generator(inputSize=args.latent_space_size, outputSize=(28, 28, 1))
	discriminatorModel = Discriminator(inputSize=(28, 28, 1))

	# Define model
	model = GenerativeAdversarialNetwork(generator=generatorModel, discriminator=discriminatorModel).to(device)
	model.setOptimizer(optim.SGD, lr=0.01)
	model.addCallbacks([SaveModels("last", "GLoss"), PlotMetrics(["Loss", "GLoss", "DLoss"]), \
		RandomPlotEachEpoch(partial(plotFn, model=model, latentSpaceSize=args.latent_space_size))])
	print(model.summary())

	if args.type == "train":
		changeDirectory(args.dir, expectExist=False)
		model.trainGenerator(generator, numEpochs=args.num_epochs)
	elif args.type == "retrain":
		model.loadModel(args.weights_file)
		model.trainGenerator(generator, numEpochs=args.num_epochs)
	elif args.type == "test_model":
		model.loadModel(args.weights_file)
		test_model(args, model)

if __name__ == "__main__":
	main()
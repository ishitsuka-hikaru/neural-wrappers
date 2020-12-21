import torch.optim as optim
import h5py
from argparse import ArgumentParser
from neural_wrappers.pytorch import GenerativeAdversarialNetwork, device
from neural_wrappers.callbacks import SaveModels, PlotMetrics
from neural_wrappers.utilities import getGenerators, changeDirectory
from neural_wrappers.readers import StaticBatchedDatasetReader

from mnist_models import GeneratorLinear as Generator, DiscriminatorLinear as Discriminator
from reader import GANReader
from test_model import test_model
from utils import PlotCallback

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
		PlotCallback(args.latent_space_size)])
	print(model.summary())

	if args.type == "train":
		changeDirectory(args.dir, expectExist=False)
		model.train_generator(generator, numIterations, numEpochs=args.num_epochs)
	elif args.type == "retrain":
		model.loadModel(args.weights_file)
		model.train_generator(generator, numIterations, numEpochs=args.num_epochs)
	elif args.type == "test_model":
		model.loadModel(args.weights_file)
		test_model(args, model)

if __name__ == "__main__":
	main()
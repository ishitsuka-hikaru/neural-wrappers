# Implementation of Variational Auto Encoder (VAE) based on http://kvfrans.com/variational-autoencoders-explained/
#  and https://arxiv.org/abs/1606.05908 using binary MNIST.
# Notes: loss is used using the empirical (latent_loss / reconstruction_loss) * reconstruction_loss + latent_loss
# If you get NaNs during training, lower the learning rate or change the optimizer.

from argparse import ArgumentParser

import torch.optim as optim

from neural_wrappers.readers import MNISTReader
from neural_wrappers.pytorch import VariationalAutoencoderNetwork, device
from neural_wrappers.pytorch.variational_autoencoder_network import latentLossFn, decoderLossFn
from neural_wrappers.pytorch.utils import npToTrCall
from neural_wrappers.callbacks import SaveModels, PlotMetrics, SaveHistory
from neural_wrappers.utilities import getGenerators, changeDirectory

from reader import BinaryMNISTReader
from models import Encoder, Decoder
from utils import SampleResultsCallback

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("type")
	parser.add_argument("datasetPath")
	parser.add_argument("--noiseSize", type=int, default=100)
	parser.add_argument("--batchSize", type=int, default=20)
	parser.add_argument("--numEpochs", type=int, default=100)
	parser.add_argument("--dir", default="test")
	parser.add_argument("--weightsFile")
	args = parser.parse_args()
	return args

def main():
	args = getArgs()

	reader = BinaryMNISTReader(args.datasetPath)
	generator, numSteps = getGenerators(reader, args.batchSize, keys=["train"])

	encoder = Encoder(noiseSize=args.noiseSize)
	decoder = Decoder(noiseSize=args.noiseSize)
	model = VariationalAutoencoderNetwork(encoder, decoder, \
		lossWeights={"latent" : 1 / (28 * 28 * 100), "decoder" : 1}).to(device)
	model.setOptimizer(optim.SGD, lr=0.001)
	model.addMetrics({
		"Reconstruction Loss" : lambda y, t, **k : npToTrCall(decoderLossFn, y, t),
		"Latent Loss" : lambda y, t, **k : npToTrCall(latentLossFn, y, t)
	})
	model.addCallbacks([SaveModels("last", "Loss"), SampleResultsCallback(), SaveHistory("history.txt"), \
		PlotMetrics(["Loss", "Reconstruction Loss", "Latent Loss"])])
	print(model.summary())

	if args.type == "train":
		changeDirectory(args.dir, expectExist=False)
		model.train_generator(generator, numSteps, numEpochs=args.numEpochs)
	elif args.type == "retrain":
		model.loadModel(args.weightsFile)
		changeDirectory(args.dir, expectExist=True)
		model.train_generator(generator, numSteps, numEpochs=args.numEpochs)
	elif args.type == "test":
		model.loadModel(args.weightsFile)
		res = model.test_generator(generator, numSteps)
		print(res)

if __name__ == "__main__":
	main()
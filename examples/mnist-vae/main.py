# Implementation of Variational Auto Encoder (VAE) based on http://kvfrans.com/variational-autoencoders-explained/
#  and https://arxiv.org/abs/1606.05908 using binary MNIST.
# Notes: loss is used using the empirical (latent_loss / reconstruction_loss) * reconstruction_loss + latent_loss
# If you get NaNs during training, lower the learning rate or change the optimizer.

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from argparse import ArgumentParser
from functools import partial

from neural_wrappers.readers import MNISTReader, StaticBatchedDatasetReader
from neural_wrappers.pytorch import VariationalAutoencoderNetwork, device
from neural_wrappers.pytorch.variational_autoencoder_network import latentLossFn, decoderLossFn
from neural_wrappers.pytorch.utils import npToTrCall
from neural_wrappers.callbacks import SaveModels, PlotMetrics, SaveHistory, RandomPlotEachEpoch
from neural_wrappers.utilities import changeDirectory

from reader import BinaryMNISTReader
from models import Encoder, Decoder

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

def plotFn(x, y, t, model):
	noise = np.random.randn(25, model.decoder.noiseSize).astype(np.float32)
	results = model.decoder.npForward(noise)
	results = results.reshape(5, 5, 28, 28) > 0.5

	plt.gcf().clf()
	ax = plt.subplots(5, 5)[1]
	for i in range(5):
		for j in range(5):
			ax[i, j].imshow(results[i, j], cmap="gray")
	plt.savefig("results.png")
	plt.close()

def main():
	args = getArgs()

	reader = StaticBatchedDatasetReader(BinaryMNISTReader(h5py.File(args.datasetPath, "r")["train"]), args.batchSize)
	print(reader)
	generator = reader.iterate()

	encoder = Encoder(noiseSize=args.noiseSize)
	decoder = Decoder(noiseSize=args.noiseSize)
	model = VariationalAutoencoderNetwork(encoder, decoder, \
		lossWeights={"latent" : 1 / (1000), "decoder" : 1}).to(device)
	model.setOptimizer(optim.AdamW, lr=0.00001)
	model.addMetrics({
		"Reconstruction Loss" : lambda y, t, **k : npToTrCall(decoderLossFn, y, t),
		"Latent Loss" : lambda y, t, **k : npToTrCall(latentLossFn, y, t)
	})
	model.addCallbacks([SaveModels("last", "Loss"), RandomPlotEachEpoch(partial(plotFn, model=model)), \
		SaveHistory("history.txt"), PlotMetrics(["Loss", "Reconstruction Loss", "Latent Loss"])])
	print(model.summary())

	if args.type == "train":
		changeDirectory(args.dir, expectExist=False)
		model.trainGenerator(generator, numEpochs=args.numEpochs)
	elif args.type == "retrain":
		model.loadModel(args.weightsFile)
		changeDirectory(args.dir, expectExist=True)
		model.trainGenerator(generator, numEpochs=args.numEpochs)
	elif args.type == "test":
		model.loadModel(args.weightsFile)
		res = model.testGenerator(generator)
		print(res)

if __name__ == "__main__":
	main()
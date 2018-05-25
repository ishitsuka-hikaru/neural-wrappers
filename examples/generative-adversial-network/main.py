import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import toimage

from neural_wrappers.readers import MNISTReader, Cifar10Reader
from neural_wrappers.pytorch import NeuralNetworkPyTorch, maybeCuda, maybeCpu, GenerativeAdversialNetwork
from neural_wrappers.callbacks import Callback

import torch as tr
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

# For some reasons, results are much better if provided data is in range -1 : 1 (not 0 : 1 or standardized).
def GANNormalization(obj, data, type):
	data = obj.minMaxNormalizer(data, type)
	data = (data - 0.5) * 2
	return data

def plot_images(images, titles, gridShape):
	plt.gcf().set_size_inches((12, 8))
	for j in range(len(images)):
		image = images[j]
		plt.gcf().add_subplot(*gridShape, j + 1)
		plt.gcf().gca().axis("off")
		plt.gcf().gca().set_title(str(titles[j]))
		cmap = None if image.shape[2] == 3 else "gray"
		image = image[..., 0] if image.shape[2] == 1 else image
		image = np.array(toimage(image))
		plt.imshow(image, cmap=cmap)

class PlotCallback(Callback):
	def __init__(self, realDataGenerator, imageShape):
		self.realDataGenerator = realDataGenerator
		self.imageShape = imageShape

		if not os.path.exists("images"):
			os.mkdir("images")

	def onEpochEnd(self, **kwargs):
		GAN = kwargs["model"]
		latentSpaceSize = GAN.generator.inputShape
		randomInputsG = Variable(maybeCuda(tr.randn(10, latentSpaceSize)))
		randomOutG = GAN.generator.forward(randomInputsG).contiguous().view(-1, *self.imageShape)
		realItems = next(self.realDataGenerator)[0]
		trRealItems = maybeCuda(Variable(tr.from_numpy(realItems)))

		inD = tr.cat([randomOutG, trRealItems], dim=0)
		outD = maybeCpu(GAN.discriminator.forward(inD).data).numpy()

		items = [maybeCpu(inD[j].data).numpy().reshape(self.imageShape) for j in range(len(inD))]
		titles = ["%2.3f" % (outD[j]) for j in range(len(outD))]
		plot_images(items, titles, gridShape=(4, 5))
		plt.savefig("images/%d.png" % (kwargs["epoch"]))
		plt.clf()

class SaveModel(Callback):
	def onEpochEnd(self, **kwargs):
		kwargs["model"].save_model("GAN.pkl")

def getReader(readerType, readerPath):
	assert readerType in ("mnist", "cifar10")
	if readerType == "mnist":
		reader = MNISTReader(readerPath, normalization=("GAN Normalization", GANNormalization))
		imageShape = (28, 28, 1)
	else:
		reader = Cifar10Reader(readerPath, normalization=("GAN Normalization", GANNormalization))
		imageShape = (32, 32, 3)
	return reader, imageShape

def getModel(dataset, latentSpaceSize, imageShape):
	if dataset == "cifar10":
		from cifar10_models import GeneratorConvTransposed as Generator
		from cifar10_models import DiscriminatorConv as Discriminator
		# Technically, I can still use either of the discriminator/generator from MNIST (linear models), but they
		#  cannot understand Cifar10 as well, so results are bad (either too good discriminator, or too bad generator)
		# from mnist_models import DiscriminatorLinear as Discriminator
	else:
		from mnist_models import GeneratorLinear as Generator
		from mnist_models import DiscriminatorLinear as Discriminator

	generator = Generator(latentSpaceSize, imageShape)
	discriminator = Discriminator(imageShape)
	return generator, discriminator

def main():
	assert len(sys.argv) == 4, "Usage: python main.py <train/retrain/test> <mnist/cifar10> <path/to/dataset.h5>"
	MB = 100
	numEpochs = 200
	latentSpaceSize = 200

	# Define reader, generator and callbacks
	reader, imageShape = getReader(sys.argv[2], sys.argv[3])
	print(reader.summary())
	generator = reader.iterate("train", miniBatchSize=MB, maxPrefetch=1)
	numIterations = reader.getNumIterations("train", miniBatchSize=MB)
	callbacks = [PlotCallback(reader.iterate("test", miniBatchSize=10, maxPrefetch=1), imageShape), SaveModel()]

	generatorModel, discriminatorModel = getModel(sys.argv[2], latentSpaceSize, imageShape)

	# Define model
	GAN = GenerativeAdversialNetwork(generator=generatorModel, discriminator=discriminatorModel)
	GAN = maybeCuda(GAN)
	GAN.generator.setOptimizer(tr.optim.Adam, lr=0.01)
	GAN.discriminator.setOptimizer(tr.optim.Adam, lr=0.01)
	GAN.setCriterion(nn.BCELoss())
	print(GAN.summary())

	if sys.argv[1] == "train":
		GAN.train_generator(generator, numIterations, numEpochs=numEpochs, callbacks=callbacks)
	elif sys.argv[1] == "retrain":
		GAN.load_model("GAN.pkl")
		GAN.train_generator(generator, numIterations, numEpochs=numEpochs, \
			callbacks=callbacks, optimizeG=True, numStepsD=1)
	else:
		GAN.load_model("GAN.pkl")
		while True:
			# Generate 20 random gaussian inputs
			randomInputsG = Variable(maybeCuda(tr.randn(20, latentSpaceSize)))
			randomOutG = GAN.generator.forward(randomInputsG).view(-1, *imageShape)
			outD = maybeCpu(GAN.discriminator.forward(randomOutG).data).numpy()

			# Plot the inputs and discriminator's confidence in them
			items = [maybeCpu(randomOutG[j].data).numpy().reshape(imageShape) for j in range(len(randomOutG))]
			titles = ["%2.3f" % (outD[j]) for j in range(len(outD))]
			plot_images(items, titles, gridShape=(4, 5))
			plt.show()
			plt.clf()

if __name__ == "__main__":
	main()
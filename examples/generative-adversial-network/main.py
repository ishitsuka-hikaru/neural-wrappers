import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

from neural_wrappers.readers import MNISTReader, Cifar10Reader
from neural_wrappers.pytorch import GenerativeAdversarialNetwork
from neural_wrappers.callbacks import Callback, SaveModels

import torch as tr
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")

# For some reasons, results are much better if provided data is in range -1 : 1 (not 0 : 1 or standardized).
def GANNormalization(data, dim):
	return (data / 255 - 0.5) * 2

class GANReader(MNISTReader):
	def __init__(self, noiseSize, **kwargs):
		self.noiseSize = noiseSize
		super().__init__(**kwargs)

	def iterate_once(self, type, miniBatchSize):
		for data, labels in super().iterate_once(type, miniBatchSize):
			# We need to give items for the entire training epoch, which includes optimizing both the generator and
			#  the discriminator. We also don't need any labels for the generator, so we'll just pass None

			MB = data.shape[0]
			gNoise = np.random.randn(MB, self.noiseSize).astype(np.float32)
			dNoise = np.random.randn(MB, self.noiseSize).astype(np.float32)

			# Pack the data in two components, one for G and one for D
			yield (gNoise, None), ((data, dNoise), None)

def plot_image(image, title):
	plt.gcf().gca().axis("off")
	plt.gcf().gca().set_title(title)
	cmap = None if image.shape[2] == 3 else "gray"
	image = image[..., 0] if image.shape[2] == 1 else image
	# image = np.array(toimage(image))
	plt.imshow(image, cmap=cmap)

def plot_images(images, titles, gridShape):
	plt.gcf().set_size_inches((12, 8))
	for j in range(len(images)):
		image = images[j]
		plt.gcf().add_subplot(*gridShape, j + 1)
		plot_image(images[j], titles[j])

class PlotCallback(Callback):
	def __init__(self, latentSpaceSize, imageShape):
		super().__init__("PlotCallback")
		self.imageShape = imageShape
		self.latentSpaceSize = latentSpaceSize

		if not os.path.exists("images"):
			os.mkdir("images")

	def onEpochEnd(self, **kwargs):
		GAN = kwargs["model"]

		# Generate 20 random gaussian inputs
		randomNoise = np.random.randn(20, self.latentSpaceSize).astype(np.float32)
		npRandomOutG = GAN.generator.npForward(randomNoise).reshape(-1, *self.imageShape)[..., 0] / 2 + 0.5

		# Plot the inputs and discriminator's confidence in them
		items = [npRandomOutG[j] for j in range(len(npRandomOutG))]
		
		ax = plt.subplots(4, 5)[1]
		for i in range(4):
			for j in range(5):
				ax[i, j].imshow(npRandomOutG[i * 4 + j], cmap="gray")
		plt.savefig("results_epoch%d.png" % (GAN.currentEpoch))

# class SaveModel(Callback):
# 	def onEpochEnd(self, **kwargs):
# 		kwargs["model"].save_model("GAN.pkl")

def getReader(readerType, readerPath, latentSpaceSize):
	assert readerType in ("mnist", )
	if readerType == "mnist":
		reader = GANReader(noiseSize=latentSpaceSize, datasetPath=readerPath, \
			normalizer=({"images" : ("GAN", GANNormalization)}))
		imageShape = (28, 28, 1)
	else:
		reader = Cifar10Reader(readerPath, normalization=("GAN Normalization", GANNormalization))
		imageShape = (32, 32, 3)
	return reader, imageShape

def getModel(dataset, latentSpaceSize, imageShape):
	if dataset == "cifar10":
		from cifar10_models import GeneratorConvTransposed as Generator
		# from cifar10_models import DiscriminatorConv as Discriminator
		# Technically, I can still use either of the discriminator/generator from MNIST (linear models), but they
		#  cannot understand Cifar10 as well, so results are bad (either too good discriminator, or too bad generator)
		from mnist_models import DiscriminatorLinear as Discriminator
		# from mnist_models import GeneratorLinear as Generator
	else:
		from mnist_models import GeneratorLinear as Generator
		from mnist_models import DiscriminatorLinear as Discriminator

	generator = Generator(latentSpaceSize, imageShape)
	discriminator = Discriminator(imageShape)
	return generator, discriminator

def main():
	assert len(sys.argv) == 4, "Usage: python main.py <train/retrain/test/test_best> <mnist/cifar10> <path/to/data.h5>"
	MB = 100
	numEpochs = 200
	latentSpaceSize = 200

	# Define reader, generator and callbacks
	reader, imageShape = getReader(sys.argv[2], sys.argv[3], latentSpaceSize)
	print(reader.summary())
	generator = reader.iterate("train", miniBatchSize=MB, maxPrefetch=1)
	numIterations = reader.getNumIterations("train", miniBatchSize=MB)
	# numIterations = 10
	# callbacks = [PlotCallback(reader.iterate("test", miniBatchSize=10, maxPrefetch=1), imageShape), SaveModel()]

	generatorModel, discriminatorModel = getModel(sys.argv[2], latentSpaceSize, imageShape)

	# Define model
	GAN = GenerativeAdversarialNetwork(generator=generatorModel, discriminator=discriminatorModel).to(device)
	GAN.generator.setOptimizer(tr.optim.SGD, lr=0.01)
	GAN.discriminator.setOptimizer(tr.optim.SGD, lr=0.01)
	GAN.addCallbacks([SaveModels("last"), PlotCallback(latentSpaceSize, imageShape)])
	print(GAN.summary())

	if sys.argv[1] == "train":
		GAN.train_generator(generator, numIterations, numEpochs=numEpochs)
	elif sys.argv[1] == "retrain":
		GAN.loadModel("model_last_Loss.pkl")
		GAN.train_generator(generator, numIterations, numEpochs=numEpochs)
	elif sys.argv[1] == "test_best":
		GAN.loadModel("model_last_Loss.pkl")
		while True:
			# Generate 100 random gaussian inputs
			randomInputsG = maybeCuda(tr.randn(100, latentSpaceSize))
			randomOutG = GAN.generator.forward(randomInputsG).view(-1, *imageShape)
			outD = maybeCpu(GAN.discriminator.forward(randomOutG).detach()).numpy()
			npRandomOutG = maybeCpu(randomOutG.detach()).numpy()
			indexes = np.where(outD > 0.95)[0]
			for j in range(len(indexes)):
				index = indexes[j]
				image = npRandomOutG[index]
				title = "%2.3f" % (outD[index])
				plot_image(image, title)
				plt.show()
	else:
		GAN.loadModel("model_last_Loss.pkl")
		while True:
			# Generate 20 random gaussian inputs
			randomNoise = np.random.randn(20, latentSpaceSize).astype(np.float32)
			npRandomOutG = GAN.generator.npForward(randomNoise).reshape(-1, *imageShape)[..., 0] / 2 + 0.5
			print(npRandomOutG.min(), npRandomOutG.max())
			# exit()

			# # Plot the inputs and discriminator's confidence in them
			items = [npRandomOutG[j] for j in range(len(npRandomOutG))]
			# titles = ["%2.3f" % (outD[j]) for j in range(len(outD))]
			# plot_images(items, titles, gridShape=(4, 5))
			ax = plt.subplots(4, 5)[1]
			for i in range(4):
				for j in range(5):
					ax[i, j].imshow(npRandomOutG[i * 4 + j], cmap="gray")
			plt.show()
			plt.clf()

if __name__ == "__main__":
	main()
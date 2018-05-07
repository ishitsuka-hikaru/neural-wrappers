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

from models import DiscriminatorLinear, GeneratorLinear, DiscriminatorMobileNetV2

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
	def __init__(self, realDataGenerator, imagesShape):
		self.realDataGenerator = realDataGenerator
		self.imagesShape = imagesShape

		if not os.path.exists("images"):
			os.mkdir("images")

	def onEpochEnd(self, **kwargs):
		GAN = kwargs["model"]
		randomInputsG = Variable(maybeCuda(tr.randn(10, 100)))
		randomOutG = GAN.generator.forward(randomInputsG).view(-1, *self.imagesShape)
		realItems = next(self.realDataGenerator)[0]
		trRealItems = maybeCuda(Variable(tr.from_numpy(realItems)))

		inD = tr.cat([randomOutG, trRealItems], dim=0)
		outD = maybeCpu(GAN.discriminator.forward(inD).data).numpy()

		items = [maybeCpu(inD[j].data).numpy().reshape(self.imagesShape) for j in range(len(inD))]
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
		reader = MNISTReader(readerPath, normalization=GANNormalization)
		imagesShape = (28, 28, 1)
	else:
		reader = Cifar10Reader(readerPath, normalization=GANNormalization)
		imagesShape = (32, 32, 3)
	return reader, imagesShape

def getModel(dataset, generatorType, discriminatorType, latentSpaceSize, imagesShape):
	assert generatorType in ("generator_linear", )
	if dataset == "cifar10":
		assert discriminatorType in ("discriminator_linear", "discriminator_mobilenetv2")
	else:
		assert discriminatorType in ("discriminator_linear", )

	if generatorType == "generator_linear":
		generator = GeneratorLinear(latentSpaceSize, imagesShape)

	if discriminatorType == "discriminator_linear":
		discriminator = DiscriminatorLinear(imagesShape)
	elif discriminatorType == "discriminator_mobilenetv2":
		discriminator = DiscriminatorMobileNetV2()

	return generator, discriminator

def main():
	assert len(sys.argv) == 6, "Usage: python main.py <train/retrain/test> <generator_model> <discriminator_model>" + \
		" <mnist/cifar10> <path/to/dataset.h5>"
	MB = 64
	numEpochs = 200
	latentSpaceSize = 100

	# Define reader, generator and callbacks
	reader, imagesShape = getReader(sys.argv[4], sys.argv[5])
	generator = reader.iterate("train", miniBatchSize=MB, maxPrefetch=1)
	numIterations = reader.getNumIterations("train", miniBatchSize=MB)
	callbacks = [PlotCallback(reader.iterate("test", miniBatchSize=10, maxPrefetch=1), imagesShape), SaveModel()]

	generatorModel, discriminatorModel = getModel(sys.argv[4], sys.argv[2], sys.argv[3], latentSpaceSize, imagesShape)

	# Define model
	GAN = GenerativeAdversialNetwork(generator=generatorModel, discriminator=discriminatorModel)
	GAN = maybeCuda(GAN)
	GAN.generator.setOptimizer(tr.optim.Adam, lr=0.0002, betas=(0.5, 0.999))
	GAN.discriminator.setOptimizer(tr.optim.Adam, lr=0.0002, betas=(0.5, 0.999))
	GAN.setCriterion(nn.BCELoss())
	print(GAN.summary())

	if sys.argv[1] == "train":
		GAN.train_generator(generator, numIterations, numEpochs=numEpochs, callbacks=callbacks)
	elif sys.argv[1] == "retrain":
		GAN.load_model("GAN.pkl")
		GAN.train_generator(generator, numIterations, numEpochs=numEpochs, callbacks=callbacks)
	else:
		GAN.load_model("GAN.pkl")
		while True:
			# Generate 20 random gaussian inputs
			randomInputsG = Variable(maybeCuda(tr.randn(20, 100)))
			randomOutG = GAN.generator.forward(randomInputsG).view(-1, *imagesShape)
			outD = maybeCpu(GAN.discriminator.forward(randomOutG).data).numpy()

			# Plot the inputs and discriminator's confidence in them
			items = [maybeCpu(randomOutG[j].data).numpy().reshape(imagesShape) for j in range(len(randomOutG))]
			titles = ["%2.3f" % (outD[j]) for j in range(len(outD))]
			plot_images(items, titles, gridShape=(4, 5))
			plt.show()
			plt.clf()

if __name__ == "__main__":
	main()
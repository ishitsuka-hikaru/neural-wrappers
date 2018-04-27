import sys
import os
import matplotlib.pyplot as plt

from neural_wrappers.readers import MNISTReader
from neural_wrappers.pytorch import NeuralNetworkPyTorch, maybeCuda, maybeCpu, GenerativeAdversialNetwork
from neural_wrappers.callbacks import Callback

import torch as tr
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class Generator(NeuralNetworkPyTorch):
	def __init__(self, outputSize=28 * 28):
		super().__init__()
		self.fc1 = nn.Linear(100, 128)
		self.fc2 = nn.Linear(128, 256)
		self.bn2 = nn.BatchNorm1d(256)
		self.fc3 = nn.Linear(256, 512)
		self.bn3 = nn.BatchNorm1d(512)
		self.fc4 = nn.Linear(512, 1024)
		self.bn4 = nn.BatchNorm1d(1024)
		self.fc5 = nn.Linear(1024, outputSize)

	def forward(self, x):
		y1 = F.leaky_relu(self.fc1(x))
		y2 = F.leaky_relu(self.bn2(self.fc2(y1)), negative_slope=0.2)
		y3 = F.leaky_relu(self.bn3(self.fc3(y2)), negative_slope=0.2)
		y4 = F.leaky_relu(self.bn4(self.fc4(y3)), negative_slope=0.2)
		y5 = F.tanh(self.fc5(y4))
		return y5

class Discriminator(NeuralNetworkPyTorch):
	def __init__(self, inputSize=28 * 28):
		super().__init__()
		self.inputSize = inputSize
		self.fc1 = nn.Linear(inputSize, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 1)

	def forward(self, x):
		x = x.view(-1, self.inputSize)
		y1 = F.leaky_relu(self.fc1(x), negative_slope=0.2)
		y2 = F.leaky_relu(self.fc2(y1), negative_slope=0.2)
		y3 = F.sigmoid(self.fc3(y2))
		y3 = y3.view(y3.shape[0])
		return y3

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
		plt.imshow(image, cmap="gray")

class PlotCallback(Callback):
	def __init__(self, realDataGenerator):
		self.realDataGenerator = realDataGenerator

		if not os.path.exists("images"):
			os.mkdir("images")

	def onEpochEnd(self, **kwargs):
		GAN = kwargs["model"]
		randomInputsG = Variable(maybeCuda(tr.randn(10, 100)))
		randomOutG = GAN.generator.forward(randomInputsG).view(-1, 28, 28)
		realItems = next(self.realDataGenerator)[0]
		trRealItems = maybeCuda(Variable(tr.from_numpy(realItems)))

		inD = tr.cat([randomOutG, trRealItems], dim=0)
		outD = maybeCpu(GAN.discriminator.forward(inD).data).numpy()

		items = [maybeCpu(inD[j].data).numpy().reshape((28, 28)) for j in range(len(inD))]
		titles = ["%2.3f" % (outD[j]) for j in range(len(outD))]
		plot_images(items, titles, gridShape=(4, 5))
		plt.savefig("images/%d.png" % (kwargs["epoch"]))
		plt.clf()

class SaveModel(Callback):
	def onEpochEnd(self, **kwargs):
		kwargs["model"].save_model("GAN.pkl")

def main():
	assert len(sys.argv) == 3, "Usage: python main.py <train/retrain/test> <path/to/mnist.h5>"
	MB = 64
	numEpochs = 200

	# Define model
	GAN = maybeCuda(GenerativeAdversialNetwork(generator=Generator(), discriminator=Discriminator()))
	GAN.generator.setOptimizer(tr.optim.Adam, lr=0.0002, betas=(0.5, 0.999))
	GAN.discriminator.setOptimizer(tr.optim.Adam, lr=0.0002, betas=(0.5, 0.999))

	# Define reader, generator and callbacks
	reader = MNISTReader(sys.argv[2], normalization=GANNormalization)
	generator = reader.iterate("train", miniBatchSize=MB, maxPrefetch=1)
	numIterations = reader.getNumIterations("train", miniBatchSize=MB)
	callbacks = [PlotCallback(reader.iterate("test", miniBatchSize=10, maxPrefetch=1)), SaveModel()]

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
			randomOutG = GAN.generator.forward(randomInputsG).view(-1, 28, 28)
			outD = maybeCpu(GAN.discriminator.forward(randomOutG).data).numpy()

			# Plot the inputs and discriminator's confidence in them
			items = [maybeCpu(randomOutG[j].data).numpy().reshape((28, 28)) for j in range(len(randomOutG))]
			titles = ["%2.3f" % (outD[j]) for j in range(len(outD))]
			plot_images(items, titles, gridShape=(4, 5))
			plt.show()
			plt.clf()

if __name__ == "__main__":
	main()
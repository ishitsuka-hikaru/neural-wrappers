import sys

from neural_wrappers.readers import MNISTReader
from neural_wrappers.pytorch import GenerativeAdversialNetwork, NeuralNetworkPyTorch, maybeCuda, maybeCpu
from Mihlib import *

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
		y1 = F.relu(self.fc1(x))
		y2 = F.relu(self.bn2(self.fc2(y1)))
		y3 = F.relu(self.bn3(self.fc3(y2)))
		y4 = F.relu(self.bn4(self.fc4(y3)))
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
		y1 = F.relu(self.fc1(x))
		y2 = F.relu(self.fc2(y1))
		y3 = self.fc3(y2)
		y4 = F.sigmoid(y3)
		return y4

def main():
	assert len(sys.argv) == 3, "Usage: python main.py <type> <path/to/mnist.h5>"
	MB = 10
	numEpochs = 100
	GAN = maybeCuda(GenerativeAdversialNetwork(generator=Generator(), discriminator=Discriminator()))
	GAN.setCriterion(nn.BCELoss())

	reader = MNISTReader(sys.argv[2])
	generator = reader.iterate("train", miniBatchSize=MB, maxPrefetch=1)
	numIterations = reader.getNumIterations("train", miniBatchSize=MB)

	if sys.argv[1] == "train":
		GAN.generator.setOptimizer(optim.SGD, lr=0.01)
		GAN.discriminator.setOptimizer(optim.SGD, lr=0.0001)
		
		GAN.train_generator(generator, stepsPerEpoch=numIterations, numEpochs=numEpochs, generatorSteps=1)

	elif sys.argv[1] == "retrain":
		GAN.load_model("GAN.pkl")
		GAN.train_generator(generator, stepsPerEpoch=numIterations, numEpochs=numEpochs, generatorSteps=5)

	elif sys.argv[1] == "test":
		GAN.load_model("GAN.pkl")

		while True:
			randomInputsG = Variable(maybeCuda(tr.randn(MB, 100)), requires_grad=False)
			outG = GAN.generator.forward(randomInputsG)
			outD = maybeCpu(GAN.discriminator.forward(outG).data).numpy()
			realItems = next(generator)[0]
			items = maybeCuda(Variable(tr.from_numpy(realItems), requires_grad=False))
			outDReal = maybeCpu(GAN.discriminator.forward(items).data).numpy()

			for j in range(len(outG)):
				generatedImage = maybeCpu(outG[j].data).numpy().reshape((28, 28))
				plot_image(generatedImage, new_figure=(j==0), axis=(4, 5, j + 1), \
					title="%2.2f" % (outD[j]), show_axis=False)

				plot_image(realItems[j], new_figure=False, axis=(4, 5, j + 10 + 1), \
					title="%2.2f" % (outDReal[j]), show_axis=False)
			show_plots(size=(12, 8))


if __name__ == "__main__":
	main()
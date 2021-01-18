import os
import h5py
import pytest
import numpy as np
from functools import partial
from overrides import overrides

import torch as tr
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from neural_wrappers.pytorch import GenerativeAdversarialNetwork, device, FeedForwardNetwork
from neural_wrappers.readers import StaticBatchedDatasetReader, PercentDatasetReader
from neural_wrappers.readers import H5BatchedDatasetReader

class GeneratorLinear(FeedForwardNetwork):
	def __init__(self, inputSize, outputSize):
		super().__init__()
		assert len(outputSize) == 3
		self.noiseSize = inputSize
		self.inputSize = inputSize
		self.outputSize = outputSize

		self.fc1 = nn.Linear(self.inputSize, 128)
		self.fc2 = nn.Linear(128, 256)
		self.bn2 = nn.BatchNorm1d(256)
		self.fc3 = nn.Linear(256, 512)
		self.bn3 = nn.BatchNorm1d(512)
		self.fc4 = nn.Linear(512, 1024)
		self.bn4 = nn.BatchNorm1d(1024)
		self.fc5 = nn.Linear(1024, outputSize[0] * outputSize[1] * outputSize[2])

	def forward(self, x):
		y1 = F.leaky_relu(self.fc1(x))
		y2 = F.leaky_relu(self.bn2(self.fc2(y1)), negative_slope=0.2)
		y3 = F.leaky_relu(self.bn3(self.fc3(y2)), negative_slope=0.2)
		y4 = F.leaky_relu(self.bn4(self.fc4(y3)), negative_slope=0.2)
		y5 = tr.tanh(self.fc5(y4))
		y5 = y5.view(-1, *self.outputSize)
		return y5

class DiscriminatorLinear(FeedForwardNetwork):
	def __init__(self, inputSize):
		super().__init__()
		assert len(inputSize) == 3
		self.inputSize = inputSize
		self.fc1 = nn.Linear(inputSize[0] * inputSize[1] * inputSize[2], 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 1)

	def forward(self, x):
		x = x.view(-1, self.inputSize[0] * self.inputSize[1] * self.inputSize[2])
		y1 = F.leaky_relu(self.fc1(x), negative_slope=0.2)
		y2 = F.leaky_relu(self.fc2(y1), negative_slope=0.2)
		y3 = tr.sigmoid(self.fc3(y2))
		y3 = y3.view(y3.shape[0])
		return y3

# For some reasons, results are much better if provided data is in range -1 : 1 (not 0 : 1 or standardized).
class GANReader(H5BatchedDatasetReader):
	def __init__(self, datasetPath:str, latentSpaceSize:int):
		super().__init__(
			datasetPath,
			dataBuckets = {"data" : ["rgb"]},
			dimGetter = {"rgb" : \
				lambda dataset, index : dataset["images"][index.start : index.stop]},
			dimTransform = {
				"data" : {"rgb" : lambda x : (np.float32(x) / 255 - 0.5) * 2}
			}
		)
		self.latentSpaceSize = latentSpaceSize

	@overrides
	def __len__(self) -> int:
		return len(self.getDataset()["images"])

	def __getitem__(self, index):
		item, MB = super().__getitem__(index)
		return (np.random.randn(MB, self.latentSpaceSize).astype(np.float32), item["data"]["rgb"]), MB

try:
	# This path must be supplied manually in order to pass these tests
	MNIST_READER_PATH = os.environ["MNIST_READER_PATH"]
	pytestmark = pytest.mark.skipif(False, reason="Dataset path not found.")
except Exception:
	pytestmark = pytest.mark.skip("MNIST Dataset path must be set.", allow_module_level=True)

class TestMNISTGAN:
	def test(self):
		reader = GANReader(datasetPath=h5py.File(MNIST_READER_PATH, "r")["train"], latentSpaceSize=200)
		reader = PercentDatasetReader(StaticBatchedDatasetReader(reader, 10), 1)

		generatorModel = GeneratorLinear(inputSize=200, outputSize=(28, 28, 1))
		discriminatorModel = DiscriminatorLinear(inputSize=(28, 28, 1))

		# Define model
		model = GenerativeAdversarialNetwork(generator=generatorModel, discriminator=discriminatorModel).to(device)
		model.setOptimizer(optim.SGD, lr=0.01)
		model.trainGenerator(reader.iterate(), numEpochs=1, printMessage=None)

def main():
	TestMNISTGAN().test()

if __name__ == "__main__":
	main()
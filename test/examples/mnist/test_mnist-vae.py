import os
import h5py
import pytest
import numpy as np

import torch as tr
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from neural_wrappers.pytorch import VariationalAutoencoderNetwork, device, FeedForwardNetwork
from neural_wrappers.pytorch.variational_autoencoder_network import latentLossFn, decoderLossFn
from neural_wrappers.readers import StaticBatchedDatasetReader, PercentDatasetReader
from neural_wrappers.readers import MNISTReader

class Encoder(FeedForwardNetwork):
	def __init__(self, noiseSize):
		super().__init__()
		self.noiseSize = noiseSize
		self.fc1 = nn.Linear(28 * 28, 100)
		self.fc2 = nn.Linear(100, 100)
		self.mean_fc = nn.Linear(100, noiseSize)
		self.mean_std = nn.Linear(100, noiseSize)

	def forward(self, x):
		x = x.view(-1, 28 * 28)
		y1 = F.relu(self.fc1(x))
		y2 = F.relu(self.fc2(y1))
		y_mean = self.mean_fc(y2)
		y_std = self.mean_std(y2)
		return y_mean, y_std

class Decoder(FeedForwardNetwork):
	def __init__(self, noiseSize):
		super().__init__()
		self.noiseSize = noiseSize
		self.fc1 = nn.Linear(noiseSize, 300)
		self.fc2 = nn.Linear(300, 28 * 28)

	def forward(self, z_samples):
		y1 = F.relu(self.fc1(z_samples))
		y2 = self.fc2(y1)
		y_decoder = tr.sigmoid(y2)
		return y_decoder

class BinaryMNISTReader(MNISTReader):
	def __getitem__(self, index):
		item, B = super().__getitem__(index)
		images = item["data"]["images"]
		images = np.float32(images > 0)
		return (images, images), B

try:
	# This path must be supplied manually in order to pass these tests
	MNIST_READER_PATH = os.environ["MNIST_READER_PATH"]
	pytestmark = pytest.mark.skipif(False, reason="Dataset path not found.")
except Exception:
	pytestmark = pytest.mark.skip("MNIST Dataset path must be set.", allow_module_level=True)

class TestMNISTVAE:
	def test(self):
		reader = BinaryMNISTReader(datasetPath=h5py.File(MNIST_READER_PATH, "r")["train"])
		reader = PercentDatasetReader(StaticBatchedDatasetReader(reader, 10), 1)

		encoder = Encoder(noiseSize=200)
		decoder = Decoder(noiseSize=200)
		model = VariationalAutoencoderNetwork(encoder, decoder, \
			lossWeights={"latent" : 1 / (1000), "decoder" : 1}).to(device)
		model.setOptimizer(optim.SGD, lr=0.01)
		model.trainGenerator(reader.iterate(), numEpochs=1, printMessage=None)

def main():
	TestMNISTVAE().test()

if __name__ == "__main__":
	main()
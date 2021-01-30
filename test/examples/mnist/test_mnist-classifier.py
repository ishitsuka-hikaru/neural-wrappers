import os
import h5py
import pytest
import numpy as np
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neural_wrappers.pytorch import device, FeedForwardNetwork
from neural_wrappers.readers import MNISTReader, StaticBatchedDatasetReader, PercentDatasetReader
from neural_wrappers.pytorch import FeedForwardNetwork

class ModelFC(FeedForwardNetwork):
	# (28, 28, 1) => (10, 1)
	def __init__(self, inputShape, outputNumClasses):
		super().__init__()

		self.inputShapeProd = int(np.prod(np.array(inputShape)))
		self.fc1 = nn.Linear(self.inputShapeProd, 100)
		self.fc2 = nn.Linear(100, 100)
		self.fc3 = nn.Linear(100, outputNumClasses)

	def forward(self, x):
		x = x["images"].view(-1, self.inputShapeProd)
		y1 = F.relu(self.fc1(x))
		y2 = F.relu(self.fc2(y1))
		y3 = self.fc3(y2)
		return y3

def lossFn(y, t):
	# Negative log-likeklihood (used for softmax+NLL for classification), expecting targets are one-hot encoded
	y = F.softmax(y, dim=1)
	t = t.type(tr.bool)
	return (-tr.log(y[t] + 1e-5)).mean()

try:
	# This path must be supplied manually in order to pass these tests
	MNIST_READER_PATH = os.environ["MNIST_READER_PATH"]
	pytestmark = pytest.mark.skipif(False, reason="Dataset path not found.")
except Exception:
	pytestmark = pytest.mark.skip("MNIST Dataset path must be set.", allow_module_level=True)

class TestMNISTClassifier:
	def test(self):
		trainReader = PercentDatasetReader(
			StaticBatchedDatasetReader(
				MNISTReader(h5py.File(MNIST_READER_PATH, "r")["train"]),
			batchSize=10),
		percent=1)
		model = ModelFC(inputShape=(28, 28, 1), outputNumClasses=10).to(device)
		model.setCriterion(lossFn)
		model.setOptimizer(optim.SGD, lr=0.01)
		model.trainGenerator(trainReader.iterate(), numEpochs=1)

def main():
	TestMNISTClassifier().test()

if __name__ == "__main__":
	main()
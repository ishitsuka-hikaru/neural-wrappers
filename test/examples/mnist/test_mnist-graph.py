import os
import h5py
import pytest
import numpy as np
from overrides import overrides
from functools import partial

import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neural_wrappers.readers import StaticBatchedDatasetReader, PercentDatasetReader
from neural_wrappers.pytorch import FeedForwardNetwork, device
from neural_wrappers.metrics import Accuracy
from neural_wrappers.graph import Graph, Edge, Node, ReduceNode
from neural_wrappers.readers import H5BatchedDatasetReader
from neural_wrappers.readers.batched_dataset_reader.h5_batched_dataset_reader import defaultH5DimGetter
from neural_wrappers.utilities import toCategorical

def topLeftFn(x):
	x = np.float32(x) / 255
	x[:, 0:14, 0:14] = 0
	return x

def topRightFn(x):
	x = np.float32(x) / 255
	x[:, 0:14, 14:] = 0
	return x

def bottomLeftFn(x):
	x = np.float32(x) / 255
	x[:, 14:, 0:14] = 0
	return x

def bottomRightFn(x):
	x = np.float32(x) / 255
	x[:, 14:, 14:] = 0
	return x

class Reader(H5BatchedDatasetReader):
	def __init__(self, datasetPath:str, normalization:str = "min_max_0_1"):
		assert normalization in ("none", "min_max_0_1")

		getterFn = partial(defaultH5DimGetter, dim="images")
		super().__init__(datasetPath,
			dataBuckets = {
				"data" : ["rgb", "rgb_top_left", "rgb_top_right", "rgb_bottom_left", "rgb_bottom_right", "labels"],
			},
			dimGetter = {
				"rgb" : getterFn,
				"rgb_top_left" : getterFn,
				"rgb_top_right" : getterFn,
				"rgb_bottom_left" : getterFn,
				"rgb_bottom_right" : getterFn
			},
			dimTransform = {
				"data" : {
					"rgb" : lambda x : np.float32(x) / 255,
					"rgb_top_left" : topLeftFn,
					"rgb_top_right" : topRightFn,
					"rgb_bottom_left" : bottomLeftFn,
					"rgb_bottom_right" : bottomRightFn,
					"labels" : lambda x : toCategorical(x, numClasses=10)
				},
			}
		)

	@overrides
	def __len__(self) -> int:
		return len(self.getDataset()["images"])

	@overrides
	def __getitem__(self, key):
		item, B = super().__getitem__(key)
		return (item["data"], item["data"]), B

def f(obj, x):
	del x["GT"]
	return tr.cat(tuple(x.values())).mean(dim=0).unsqueeze(dim=0)

class FCEncoder(FeedForwardNetwork):
	# (28, 28, 1) => (10, 1)
	def __init__(self, inputShape):
		super().__init__()

		self.inputShapeProd = int(np.prod(np.array(inputShape)))
		self.fc1 = nn.Linear(self.inputShapeProd, 100)
		self.fc2 = nn.Linear(100, 100)

	def forward(self, x):
		x = x.view(-1, self.inputShapeProd)
		y1 = F.relu(self.fc1(x))
		y2 = F.relu(self.fc2(y1))
		return y2

class FCDecoder(FeedForwardNetwork):
	def __init__(self, numClasses):
		super().__init__()
		self.numClasses = numClasses
		self.fc = nn.Linear(100, numClasses)

	def forward(self, x):
		y = self.fc(x)
		y = F.softmax(y, dim=1)
		return y

class RGB(Node):
	def __init__(self, name="RGB", groundTruthKey="rgb"):
		super().__init__(name=name, groundTruthKey=groundTruthKey)

	def getEncoder(self, outputNodeType=None):
		return FCEncoder((28, 28, 1))

	def getDecoder(self, inputNodeType=None):
		assert False

	def getNodeMetrics(self):
		return {}

	def getNodeCriterion(self):
		return lambda y, t : (y - t)**2

	def getMetrics(self):
		return self.getNodeMetrics()

	def getCriterion(self):
		return self.getNodeCriterion()

class RGBTopLeft(RGB):
	def __init__(self):
		super().__init__(name="RGBTopLeft", groundTruthKey="rgb_top_left")

class RGBTopRight(RGB):
	def __init__(self):
		super().__init__(name="RGBTopRight", groundTruthKey="rgb_top_right")

class RGBBottomLeft(RGB):
	def __init__(self):
		super().__init__(name="RGBBottomLeft", groundTruthKey="rgb_bottom_left")

class RGBBottomRight(RGB):
	def __init__(self):
		super().__init__(name="RGBBottomRight", groundTruthKey="rgb_bottom_right")

class Label(Node):
	def __init__(self):
		super().__init__(name="Label", groundTruthKey="labels")

	def getEncoder(self, outputNodeType=None):
		assert False

	def getDecoder(self, inputNodeType=None):
		return FCDecoder(10)

	def getNodeMetrics(self):
		return {
			"Accuracy" : Accuracy(),
		}

	def getNodeCriterion(self):
		return Label.lossfn
	
	def lossfn(y, t):
		# Negative log-likeklihood (used for softmax+NLL for classification), expecting targets are one-hot encoded
		t = t.type(tr.bool)
		return (-tr.log(y[t] + 1e-5)).mean()

	def getMetrics(self):
		return self.getNodeMetrics()

	def getCriterion(self):
		return self.getNodeCriterion()

def getModel():
	rgb = RGB()
	rgbTopLeft = RGBTopLeft()
	rgbTopRight = RGBTopRight()
	rgbBottomLeft = RGBBottomLeft()
	rgbBottomRight = RGBBottomRight()
	label = Label()

	rgb2label = Edge(rgb, label)
	rgbTopLeft2Label = Edge(rgbTopLeft, label)
	rgbTopRight2Label = Edge(rgbTopRight, label)
	rgbBottomLeft2Label = Edge(rgbBottomLeft, label)
	rgbBottomRight2Label = Edge(rgbBottomRight, label)
	reduceNode = ReduceNode(label, forwardFn=f)
	graph = Graph([
		rgb2label,
		rgbTopLeft2Label,
		rgbTopRight2Label,
		rgbBottomLeft2Label,
		rgbBottomRight2Label,
		reduceNode
	]).to(device)
	return graph

try:
	# This path must be supplied manually in order to pass these tests
	MNIST_READER_PATH = os.environ["MNIST_READER_PATH"]
	pytestmark = pytest.mark.skipif(False, reason="Dataset path not found.")
except Exception:
	pytestmark = pytest.mark.skip("MNIST Dataset path must be set.", allow_module_level=True)

class TestMNISTGraph:
	def test(self):
		reader = Reader(datasetPath=h5py.File(MNIST_READER_PATH, "r")["train"])
		reader = PercentDatasetReader(StaticBatchedDatasetReader(reader, 10), 1)

		model = getModel().to(device)
		model.setOptimizer(optim.SGD, lr=0.01)
		model.trainGenerator(reader.iterate(), numEpochs=1)

def main():
	TestMNISTGraph().test()

if __name__ == "__main__":
	main()
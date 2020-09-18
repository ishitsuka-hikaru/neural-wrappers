import numpy as np
import matplotlib.pyplot as plt
from neural_wrappers.readers import H5DatasetReader
from neural_wrappers.utilities import getGenerators
from neural_wrappers.callbacks import SaveModels, PlotMetrics
from neural_wrappers.metrics import Accuracy, MetricWrapper
from argparse import ArgumentParser

import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neural_wrappers.pytorch import FeedForwardNetwork
from neural_wrappers.graph import Graph, Edge, Node, ReduceNode

from reader import Reader
accuracy = None

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
		super().__init__(name="Label", groundTruthKey="label")

	def getEncoder(self, outputNodeType=None):
		assert False

	def getDecoder(self, inputNodeType=None):
		return FCDecoder(10)

	def getNodeMetrics(self):
		return {
			"Accuracy" : MetricWrapper(Label.accuracy, direction="max"),
		}

	def getNodeCriterion(self):
		return Label.lossfn
	
	def lossfn(y, t):
		# Negative log-likeklihood (used for softmax+NLL for classification), expecting targets are one-hot encoded
		t = t.type(tr.bool)
		return (-tr.log(y[t] + 1e-5)).mean()

	@staticmethod
	def accuracy(y, t, **k):
		global accuracy
		if not accuracy:
			accuracy = Accuracy()
		return accuracy(y, t, **k)

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("type")
	parser.add_argument("datasetPath")

	parser.add_argument("--numEpochs", type=int, default=100)
	parser.add_argument("--batchSize", type=int, default=20)
	args = parser.parse_args()

	assert args.type in ("train", )
	return args

def f(obj, x):
	del x["GT"]
	return tr.cat(tuple(x.values())).mean(dim=0).unsqueeze(dim=0)

def main():
	args = getArgs()

	reader = Reader(args.datasetPath)
	generator, numSteps, valGenerator, valNumSteps = getGenerators(reader, batchSize=args.batchSize, \
		keys=["train", "test"])

	# items = next(generator)
	# data = items[0]
	# ax = plt.subplots(5, )[1]
	# for i, key in enumerate(list(data.keys())[0 : -1]):
	# 	ax[i].imshow(data[key][0])
	# 	ax[i].set_title(key)
	# plt.show()
	# breakpoint()
	# exit()

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
	])
	graph.setOptimizer(optim.SGD, lr=0.001)
	# graph.addCallbacks([PlotMetrics(["Loss", (str(reduceNode), "Accuracy")])])
	print(graph.summary())

	if args.type == "train":
		graph.train_generator(generator, numSteps, args.numEpochs, valGenerator, valNumSteps)
		pass

if __name__ == "__main__":
	main()
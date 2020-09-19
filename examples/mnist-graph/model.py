import numpy as np
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from neural_wrappers.pytorch import FeedForwardNetwork, device
from neural_wrappers.metrics import MetricWrapper
from neural_wrappers.graph import Graph, Edge, Node, ReduceNode

accuracy = None

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
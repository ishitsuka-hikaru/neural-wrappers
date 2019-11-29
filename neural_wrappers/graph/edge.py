import torch.nn as nn
from functools import partial
from .node import MapNode, VectorNode
from ..pytorch import NeuralNetworkPyTorch

# Default loss of this edge goes through all ground truths and all outputs of the output node and computes the
#  loss between them. This can be updated for a more specific edge algorithm for loss computation.
def defaultLossFn(self, y, t):
	A, B, edgeID = self.inputNode, self.outputNode, self.edgeID
	L = 0
	t = B.getGroundTruth()
	for y in B.outputs[edgeID]:
		L += B.decoders[edgeID].criterion(y, t)
	return L

# Communication between input and output node.
def defaultForward(self, x):
	A, B, model, edgeID = self.inputNode, self.outputNode, self.model, self.edgeID
	edgeInputs, inputNodeKeys = A.getInputs()
	B.outputs[edgeID] = []

	for x in edgeInputs:
		y = model.forward(x)
		B.outputs[edgeID].append(y)
	# print("[%s forward] Num messages: %d. In keys: %s. In Shape: %s. Out Shape: %s" % (edgeID, inputNodeKeys, \
		# len(B.outputs[edgeID]), edgeInputs[0].shape, B.outputs[edgeID][0].shape))
	return B.outputs[edgeID]

class Edge(NeuralNetworkPyTorch):
	def __init__(self, inputNode, outputNode, edgeType="edge-edge", forwardFn=None, lossFn=None, dependencies=[]):
		super().__init__()
		assert edgeType in ("node-node", "node-edge", "edge-node", "edge-edge")
		self.inputNode = inputNode
		self.outputNode = outputNode
		self.edgeType = edgeType

		# Model stuff
		self.edgeID = str(self)
		self.inputNode.addEncoder(type(self.outputNode), self.edgeID, self.edgeType)
		self.outputNode.addDecoder(type(self.inputNode), self.edgeID, self.edgeType)
		self.model = self.getModel()
		self.metrics = self.outputNode.decoders[self.edgeID].getMetrics()
		if not forwardFn:
			forwardFn = defaultForward
		if not lossFn:
			lossFn = defaultLossFn
		self.forwardFn = forwardFn
		self.lossFn = lossFn

		self.dependencies = dependencies

	def forward(self, x):
		return self.forwardFn(self, x)

	def loss(self, y, t):
		return self.lossFn(self, y, t)

	# Default model for this edge is just a sequential mapping between the A's encoder and B's decoder.
	#  Other edges may requires additional edge-specific parameters or some more complicated views to convert the
	#   output of A's encoder to the input of B's decoder.
	def getModel(self):
		return nn.Sequential(
			self.inputNode.encoders[self.edgeID],
			self.outputNode.decoders[self.edgeID]
		)

	def __str__(self):
		return "%s -> %s" % (str(self.inputNode), str(self.outputNode))

	def __repr__(self):
		return str(self)
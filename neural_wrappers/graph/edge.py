import torch.nn as nn
from functools import partial
from .node import MapNode, VectorNode

class Edge(nn.Module):
	def __init__(self, inputNode, outputNode, forwardFn=None, lossFn=None, dependencies=[]):
		super().__init__()
		self.inputNode = inputNode
		self.outputNode = outputNode
		self.model = self.inputNode.getModel(self.outputNode)
		self.metrics = self.model.getMetrics()
		self.dependencies = dependencies

		if forwardFn is None:
			forwardFn = Edge.defaultForward
		if lossFn is None:
			lossFn = Edge.defaultLossFn
		self.forward = partial(forwardFn, A=self.inputNode, B=self.outputNode, model=self.model, edgeID=str(self))
		self.lossFn = partial(lossFn, A=self.inputNode, B=self.outputNode, edgeID=str(self))

	# Default loss of this edge goes through all ground truths and all outputs of the output node and computes the
	#  loss between them. This can be updated for a more specific edge algorithm for loss computation.
	def defaultLossFn(t, A, B, edgeID):
		L = 0
		t = t[B.groundTruthKey]
		for y in B.outputs[edgeID]:
			L += B.lossFn(y, t)
		return L

	# Communication between input and output node.
	def defaultForward(inputs, A, B, model, edgeID):
		edgeInputs, inputNodeKeys = A.getInputs(inputs)
		B.outputs[edgeID] = []
		for x in edgeInputs:
			# Get the input from all possible inputs
			y = model.forward(x)
			B.outputs[edgeID].append(y)
		# print("[%s forward] Num messages: %d. In keys: %s. In Shape: %s. Out Shape: %s" % (edgeID, inputNodeKeys, \
			# len(B.outputs[edgeID]), edgeInputs[0].shape, B.outputs[edgeID][0].shape))

	def __str__(self):
		return "%s -> %s" % (str(self.inputNode), str(self.outputNode))

	def __repr__(self):
		return str(self)
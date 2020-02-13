import torch as tr
from functools import partial
from ..edge import Edge, defaultLossFn
from ..node import Node
from ...pytorch import trModuleWrapper

### Some custom edges ###
class ReduceNode(Edge):
	def __init__(self, inNode, forwardFn, useGT=True, *args, **kwargs):
		name = "ReduceNode (%s)" % (inNode.name)
		outNode = Node(name, inNode.groundTruthKey)
		self.useGT = useGT
		super().__init__(inNode, outNode, forwardFn=forwardFn, *args, **kwargs)

	def forward(self, x):
		# Simply collect all inputs and send it to the default forward function (that will call our callback)
		res = []
		for key in x:
			print(key, self.useGT)
			if key == "GT" and not self.useGT:
				continue
			res.append(x[key])
		res = tr.cat(res, dim=0)
		return super().forward(res)

	# Default model for this edge is just a sequential mapping between the A's encoder and B's decoder.
	#  Other edges may requires additional edge-specific parameters or some more complicated views to convert the
	#   output of A's encoder to the input of B's decoder.
	def setupModel(self):
		assert self.model is None
		self.model = trModuleWrapper(lambda x : x)
		self.addMetrics(self.inputNode.getMetrics())
		self.setCriterion(self.inputNode.getCriterion())
		if not self.lossFn:
			self.lossFn = defaultLossFn

class ReduceEdge(ReduceNode):
	def __init__(self, senderNodes, receiverNode, forwardFn, useGT=True, *args, **kwargs):
		self.senderNodes = senderNodes
		super().__init__(receiverNode, forwardFn, useGT, *args, **kwargs)

	def forward(self, x):
		newX = {}
		for key in x:
			if hasattr(key, "inputNode") and key.inputNode in self.senderNodes:
				newX[key] = x[key]
			if key == "GT" and self.useGT:
				newX[key] = x[key]
		return super().forward(newX)
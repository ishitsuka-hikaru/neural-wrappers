import torch as tr
from functools import partial
from ..edge import Edge, defaultLossFn
from ..node import Node
from ...pytorch import trModuleWrapper

class ReduceNode(Edge):
	def __init__(self, inputNode, forwardFn, *args, **kwargs):
		super().__init__(inputNode, inputNode, forwardFn=forwardFn, *args, **kwargs)

	def forward(self, x):
		self.inputs = x
		res = self.forwardFn(self, x)
		self.outputs = res
		self.inputNode.messages = {}
		self.inputNode.addMessage(self, res)
		return self.outputs

	def loss(self, y, t):
		return None

	def setupModel(self):
		assert self.model is None
		self.model = trModuleWrapper(lambda x : x)
		self.lossFn = lambda y, t : None

	def getMetrics(self):
		return {}

	def getDecoder(self):
		return trModuleWrapper(lambda x : x)

	def getEncoder(self):
		return trModuleWrapper(lambda x : x)

class ReduceEdge: pass
import torch as tr
from functools import partial
from ..edge import Edge, defaultLossFn
from ..node import Node
from ...pytorch import trModuleWrapper

class ReduceNode(Edge):
	def __init__(self, inputNode, forwardFn, name=None, useMetrics=False, useLoss=False, *args, **kwargs):
		if name is None:
			name = "ReduceNode %s" % (str(inputNode))
		super().__init__(inputNode, inputNode, forwardFn=forwardFn, name=name, *args, **kwargs)
		if not useMetrics:
			self.clearMetrics()
		if not useLoss:
			self.lossFn = lambda y, t : None

	def forward(self, x):
		self.inputs = x
		res = self.forwardFn(self, x)
		self.outputs = res
		# Clear node's messages and replace them with the reduced version.
		self.inputNode.messages = {}
		self.inputNode.addMessage(self, res)
		return self.outputs

	def getDecoder(self):
		return trModuleWrapper(lambda x : x)

	def getEncoder(self):
		return trModuleWrapper(lambda x : x)

class ReduceEdge: pass
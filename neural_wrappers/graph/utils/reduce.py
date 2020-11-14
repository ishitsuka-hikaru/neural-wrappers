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
		# Clear node's messages and replace them with the reduced version.
		self.inputNode.messages = {}
		self.inputNode.addMessage(self, res)
		return self.outputs

	def getDecoder(self):
		return trModuleWrapper(lambda x : x)

	def getEncoder(self):
		return trModuleWrapper(lambda x : x)

	def __str__(self):
		return "ReduceNode %s" % (self.strInputNode)

class ReduceEdge: pass
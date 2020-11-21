import torch as tr
from ..edge import Edge
from ..node import Node
from ...pytorch import trModuleWrapper
from typing import Any

# Simple wrapper that takes ALL inputs from the input node and put them in the outputNode's messages.
class ForwardMessagesEdge(Edge):
	def __init__(self, inputNode, outputNode, forwardGT=True, *args, **kwargs):
		self.forwardGT = forwardGT
		super().__init__(inputNode=inputNode, outputNode=outputNode, *args, **kwargs)
	
	def forward(self, x : dict) -> Any: #type: ignore[override]
		assert type(x) == dict
		# Redirect all messags as is.
		res = {}
		for k in x:
			if k == "GT" and self.forwardGT == False:
				continue
			self.outputNode.messages[k] = x[k]
			res[k] = x[k]
		# Also return the inputs for further use in the graph.
		return [res]

	def loss(self, y, t):
		return None
	
	def getDecoder(self):
		return trModuleWrapper(lambda x : x)
			
	def getEncoder(self):
		return trModuleWrapper(lambda x : x)

	def setupModel(self):
		assert self.model is None
		self.model = trModuleWrapper(lambda x : x)
		self.lossFn = lambda y, t : None

	def getMetrics(self):
		return {}

	def __str__(self):
		return "ForwardMessages %s -> %s" % (str(self.inputNode), str(self.outputNode))

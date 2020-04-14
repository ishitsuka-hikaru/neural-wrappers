import torch as tr
from ..edge import Edge
from ..node import Node

# Simple wrapper that takes ALL inputs from the input node and put them in the outputNode's messages.
class ForwardMessagesEdge(Edge):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def forward(self, x : dict) -> dict:
		assert type(x) == dict
		# Redirect all messags as is.
		for k in x:
			self.outputNode.messages[k] = x[k]
		# Also return the inputs for further use in the graph.
		return x

	def loss(self, y, t):
		return None
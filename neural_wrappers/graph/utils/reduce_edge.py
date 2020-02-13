import torch as tr
from functools import partial
from ..edge import Edge, defaultLossFn
from ..node import Node
from ...pytorch import trModuleWrapper

### Some custom edges ###
class ReduceEdge(Edge):
	class ReduceNode(Node):
		def __init__(self, inNode, groundTruthKey):
			self.inNode = inNode
			name = "ReduceEdge (Node: %s)" % (inNode.name.split(" ")[0])
			super().__init__(name=name, groundTruthKey=self.inNode.groundTruthKey)

	def forward(self, x):
		# Simply collect all inputs and send it to the default forward function (that will call our callback)
		res = []
		for key in x:
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

	def __init__(self, inNode, forwardFn, *args, **kwargs):
		benchmarkOutNode = ReduceEdge.ReduceNode(inNode, inNode.groundTruthKey)
		super().__init__(inNode, benchmarkOutNode, forwardFn=forwardFn, *args, **kwargs)
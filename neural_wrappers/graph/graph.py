import torch.nn as nn
from neural_wrappers.pytorch import NeuralNetworkPyTorch
from functools import partial

class Graph(NeuralNetworkPyTorch):
	def __init__(self, edges):
		super().__init__()
		self.edges = nn.ModuleList(edges)

		# Add metrics
		self.addMetrics(self.getEdgesMetrics())
		self.setCriterion(partial(Graph.lossFn, self=self))

	def lossFn(y, t, self):
		loss = 0
		for edge in self.edges:
			loss += edge.lossFn(t)
		return loss

	def forward(self, x):
		# TODO: Execution order. (synchronus vs asynchronus as well as topological sort at various levels.)
		# For now, the execution is synchronous and linear as defined by the list of edges
		outputs = {}
		for edge in self.edges:
			edgeOutput = edge.forward(x)
			outputs[edge] = edgeOutput
		# print("_____________________________")
		return outputs

	# TODO!
	def getEdgesMetrics(self):
		return {}
	# 	metrics = {}
	# 	for edge in self.edges:
	# 		for metric in edge.metrics:
	# 		# outputMetrics = edge.outputNode.metrics
	# 		# for metric in outputMetrics:
	# 			newName = "%s %s" % (edge, metric)
	# 			newF = partial(outputMetrics[metric], keyName=edge)
	# 			metrics[newName] = newF
	# 	return metrics

	def __str__(self):
		Str = "Graph:"
		for edge in self.edges:
			Str += "\n\t-%s" % (str(edge))
		return Str
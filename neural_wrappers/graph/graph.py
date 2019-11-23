import torch.nn as nn
from ..pytorch import NeuralNetworkPyTorch
from functools import partial

class Graph(NeuralNetworkPyTorch):
	def __init__(self, edges):
		super().__init__()
		self.edges = nn.ModuleList(edges)
		self.nodes = self.getNodes()

		# Add metrics
		self.addMetrics(self.getEdgesMetrics())
		self.setCriterion(partial(Graph.lossFn, self=self))

	def lossFn(y, t, self):
		loss = 0
		for edge in self.edges:
			loss += edge.lossFn(t)
		return loss

	def networkAlgorithm(self, trInputs, trLabels):
		trResults = {}
		# TODO: Execution order. (synchronus vs asynchronus as well as topological sort at various levels.)
		# For now, the execution is synchronous and linear as defined by the list of edges
		for edge in self.edges:
			# Set GT for both sides of the edge, if none is already available. Edge algorithm could've overwritten this
			#  node's GT OR can also clear the GT itself. However, it is graph's responsability to try to give the GT
			#  using the groundTruthKey if possible
			Graph.trySetNodeGT(trInputs, edge.inputNode)
			Graph.trySetNodeGT(trInputs, edge.outputNode)

			edgeOutput = edge.forward(trInputs)
			trResults[edge] = edgeOutput

		# print("_____________________________")
		trLoss = self.criterion(trResults, trLabels)

		# Clear GT for all nodes after going through the graph
		for node in self.nodes:
			node.setGroundTruth(None)
			node.clearNodeOutputs()
		return trResults, trLoss

	# TODO!
	def getEdgesMetrics(self):
		for edge in self.edges:
			print(edge)
		# exit()
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

	def getNodes(self):
		nodes = set()
		for edge in self.edges:
			nodes.add(edge.inputNode)
			nodes.add(edge.outputNode)
		return nodes

	# Try to set the GT of the node from the dict of inputs. We will store the value inputs[node.groundTruthKey] if
	#  the node has a groundTruthKey and its groundTruth is None (Not set by other edge beforehand)
	def trySetNodeGT(inputs, node):
		if not node.groundTruthKey in inputs:
			return
		if not node.getGroundTruth() is None:
			return
		node.setGroundTruth(inputs[node.groundTruthKey])

	def __str__(self):
		Str = "Graph:"
		for edge in self.edges:
			Str += "\n\t-%s" % (str(edge))
		return Str
import torch.nn as nn
import torch.optim as optim
from functools import partial
from copy import copy
from overrides import overrides

from .draw_graph import drawGraph
from .graph_serializer import GraphSerializer
from ..pytorch import NWModule, npGetData
from ..utilities import isType

# A Graph is a list of Edges. Each edge is a FeedForward network between two nodes.
class Graph(NWModule):
	def __init__(self, edges, hyperParameters={}):
		self.edges = edges
		self.nodes = self.getNodes()
		hyperParameters = self.getHyperParameters(hyperParameters)
		super().__init__(hyperParameters=hyperParameters)

		self.edges = nn.ModuleList(self.getEdges())
		self.edgeIDsToEdges = self.getStrMapping()
		self.edgeLoss = {}
		self.setCriterion(self.loss)

		self.serializer = GraphSerializer(self)

	@overrides
	def networkAlgorithm(self, trInputs, trLabels, isTraining, isOptimizing):
		trResults = {}
		# TODO: Execution order. (synchronus vs asynchronus as well as topological sort at various levels.)
		# For now, the execution is synchronous and linear as defined by the list of edges
		for edge in self.edges:
			edgeID = str(edge)
			trInputs, b = trInputs
			edgeInputs = edge.getInputs(trInputs)
			edgeOutput = edge.forward(edgeInputs)
			# Update the outputs of the whole graph as well
			trResults[edgeID] = edgeOutput

		trLoss = self.criterion(trResults, trLabels)
		self.updateOptimizer(trLoss, isTraining, isOptimizing)
		return trResults, trLoss

	def loss(self, y, t):
		loss = 0
		self.edgeLoss[None] = 0
		for edge in self.edges:
			edgeID = str(edge)
			edgeLoss = edge.loss(y, t)
			self.edgeLoss[edgeID] = npGetData(edgeLoss)
			# Cummulate it for the entire graph as well for statistics.
			self.edgeLoss[None] += self.edgeLoss[edgeID]
	
			if not edgeLoss is None:
				loss += edgeLoss
		return loss

	# Graphs and subgraphs use all the possible inputs.
	# TODO: Perhaps it'd be better to check what inputs the edges require beforehand, but that might be just too
	#  and redundant, since the forward of the subgraphs will call getInputs of each edge anyway.
	def getInputs(self, trInputs):
		return trInputs

	def getEdges(self):
		edges = []
		for edge in self.edges:
			edges.append(edge)
		return edges

	def getStrMapping(self):
		res = {}
		for edge in self.edges:
			edgeMapping = edge.getStrMapping()
			# This adds graphs too
			res[str(edge)] = edge
			if type(edgeMapping) == str:
				res[edgeMapping] = edge
			else:
				for k in edgeMapping:
					res[k] = edgeMapping[k]
		return res

	def getNodes(self):
		nodes = set()
		for edge in self.edges:
			# edge can be an actual Graph.
			for node in edge.getNodes():
				nodes.add(node)
		return nodes

	def draw(self, fileName, cleanup=True, view=False):
		drawGraph(self.nodes, self.edges, fileName, cleanup, view)

	def getHyperParameters(self, hyperParameters):
		# Set up hyperparameters for every node
		hyperParameters = {k : hyperParameters[k] for k in hyperParameters}
		for node in self.nodes:
			hyperParameters[node.name] = node.hyperParameters
		for edge in self.edges:
			hyperParameters[str(edge)] = edge.hyperParameters
		return hyperParameters

	def graphStr(self, depth=1):
		Str = "Graph:"
		pre = "\t" * depth
		for edge in self.edges:
			if type(edge) == Graph:
				edgeStr = edge.graphStr(depth + 1)
			else:
				edgeStr = str(edge)
			Str += "\n%s-%s" % (pre, edgeStr)
		return Str

	def getGroundTruth(self, x):
		return x

	# We also override some methods on the Network class so it works with edges as well.

	@overrides
	def setOptimizer(self, optimizer, **kwargs):
		assert isinstance(optimizer, )
		breakpoint()
		assert isType(optimizer, "type"), "TODO For more special cases: %s" % type(optimizer)
	
		# 	self.optimizer = optimizer
		# else:
		# 	params = []
		# 	for edge in self.edges:
		# 		edgeParams =  list(filter(lambda p : p.requires_grad, edge.parameters()))
		# 		params.append({"params" : edgeParams})
		# 	self.optimizer = optimizer(params, **kwargs)
		# self.optimizer.storedArgs = kwargs

	@overrides
	def getOptimizerStr(self):
		strList = super().getOptimizerStr()
		for edge in self.edges:
			strEdge = str(edge)
			if type(edge) == Graph:
				strEdge = "SubGraph"
			edgeStrList = edge.getOptimizerStr()
			strList.extend(edgeStrList)
		return strList

	@overrides
	def initializeEpochMetrics(self):
		res = super().initializeEpochMetrics()
		for edge in self.edges:
			res[str(edge)] = edge.initializeEpochMetrics()
		return res

	@overrides
	def reduceEpochMetrics(self, metricResults):
		results = {None : super().reduceEpochMetrics(metricResults)}
		for edge in self.edges:
			results[str(edge)] = edge.reduceEpochMetrics(metricResults[str(edge)])
		return results

	@overrides
	def callbacksOnIterationStart(self, isTraining, isOptimizing):
		super().callbacksOnIterationStart(isTraining, isOptimizing)
		for edge in self.edges:
			edge.callbacksOnIterationStart(isTraining, isOptimizing)

	@overrides
	def callbacksOnIterationEnd(self, data, labels, results, loss, iteration, numIterations, \
		metricResults, isTraining, isOptimizing):
		thisResults = super().callbacksOnIterationEnd(data, labels, results, loss, iteration, numIterations, \
				metricResults, isTraining, isOptimizing)

		for edge in self.edges:
			edgeResults = results[str(edge)]
			edgeLabels = edge.getGroundTruth(labels)
			edgeMetricResults = metricResults[str(edge)]
			edgeLoss = self.edgeLoss[str(edge)]
			thisResults[str(edge)] = edge.callbacksOnIterationEnd(data, edgeLabels, \
				edgeResults, edgeLoss, iteration, numIterations, edgeMetricResults, isTraining, isOptimizing)
		return thisResults

	@overrides
	def callbacksOnEpochStart(self, isTraining):
		super().callbacksOnEpochStart(isTraining)
		for edge in self.edges:
			edge.callbacksOnEpochStart(isTraining)

	@overrides
	def metricsSummary(self):
		summaryStr = super().metricsSummary()
		for edge in self.edges:
			strEdge = str(edge)
			if type(edge) == Graph:
				strEdge = "SubGraph"
			lines = edge.metricsSummary().split("\n")[0 : -1]
			if len(lines) > 0:
				summaryStr += "\t- %s:\n" % (strEdge)
				for line in lines:
					summaryStr += "\t%s\n" % (line)
		return summaryStr

	@overrides
	def callbacksSummary(self):
		summaryStr = super().callbacksSummary()
		for edge in self.edges:
			strEdge = str(edge)
			if type(edge) == Graph:
				strEdge = "SubGraph"
			lines = edge.callbacksSummary()
			if len(lines) == 0:
				continue
			summaryStr += "\n\t- %s:\n\t\t%s" % (strEdge, lines)
		return summaryStr

	@overrides
	def iterationEpilogue(self, isTraining, isOptimizing, trLabels):
		# Set the GT for each node based on the inputs available at this step. Edges may overwrite this when reaching
		#  a node via an edge, however it is the graph's responsability to set the default GTs. What happens during the
		#  optimization shouldn't be influenced by this default.
		# If the ground truth key is "*", then all items are provided to the node and it's expected that the node will
		#  manage the labels accordingly.
		for node in self.nodes:
			node.setGroundTruth(trLabels)
			node.messages = {}

	@overrides
	def epochPrologue(self, epochResults, numEpochs, isTraining):
		mainResults = {k : epochResults[k][None] for k in epochResults}
		super().epochPrologue(mainResults, numEpochs, isTraining)
		for edge in self.edges:
			edgeResults = {k : epochResults[k][str(edge)] for k in epochResults}
			edge.epochPrologue(edgeResults, numEpochs, isTraining)

	def __str__(self):
		return self.graphStr()
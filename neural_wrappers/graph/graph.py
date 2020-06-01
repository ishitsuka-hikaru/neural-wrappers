import torch.nn as nn
from ..pytorch import NeuralNetworkPyTorch, getNpData, getTrData, npTrCall, trNpCall
from ..utilities import MultiLinePrinter, getFormattedStr
from functools import partial
from copy import copy

from .draw_graph import drawGraph

class Graph(NeuralNetworkPyTorch):
	def __init__(self, edges, hyperParameters={}):
		self.edges =edges
		self.nodes = self.getNodes()
		hyperParameters = self.getHyperParameters(hyperParameters)
		super().__init__(hyperParameters=hyperParameters)

		self.edges = nn.ModuleList(self.getEdges())
		self.edgeIDsToEdges = self.getStrMapping()
		self.edgeLoss = {}
		self.linePrinter = MultiLinePrinter()
		self.setCriterion(self.loss)

	def loss(self, y, t):
		loss = 0
		for edge in self.edges:
			edgeID = str(edge)
			edgeLoss = edge.loss(y, t)
			self.edgeLoss[edgeID] = getNpData(edgeLoss)
	
			# If this edge has no loss, ignore it.
			if edgeLoss is None:
				continue
			# If this edge is not trainable, also ignore it (? To think if this is correct ?)
			# TODO: see how to fast check if edge is trainable (perhaps not an issue at all to add untrainable ones)

			# Otherwise, just add it to the loss of the entire graph
			loss += edgeLoss
		return loss

	# Graphs and subgraphs use all the possible inputs.
	# TODO: Perhaps it'd be better to check what inputs the edges require beforehand, but that might be just too
	#  and redundant, since the forward of the subgraphs will call getInputs of each edge anyway.
	def getInputs(self, trInputs):
		return trInputs

	def forward(self, trInputs):
		trResults = {}
		# TODO: Execution order. (synchronus vs asynchronus as well as topological sort at various levels.)
		# For now, the execution is synchronous and linear as defined by the list of edges
		for edge in self.edges:
			edgeID = str(edge)
			edgeInputs = edge.getInputs(trInputs)
			edgeOutput = edge.forward(edgeInputs)
			# Update the outputs of the whole graph as well
			trResults[edgeID] = edgeOutput
		return trResults

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

	def initializeEpochMetrics(self):
		res = super().initializeEpochMetrics()
		for edge in self.edges:
			res[str(edge)] = edge.initializeEpochMetrics()
		return res

	def reduceEpochMetrics(self, metricResults):
		results = super().reduceEpochMetrics(metricResults)
		for edge in self.edges:
			results[str(edge)] = edge.reduceEpochMetrics(metricResults[str(edge)])
		return results

	### Some updates to original NeuralNetworkPyTorch to work seamlessly with graphs (mostly printing)

	def getGroundTruth(self, x):
		return x

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

	def computeIterPrintMessage(self, i, stepsPerEpoch, metricResults, iterFinishTime):
		messages = super().computeIterPrintMessage(i, stepsPerEpoch, metricResults, iterFinishTime)

		for edge in self.edges:
			if type(edge) == Graph:
				strEdge = "SubGraph"
			else:
				strEdge = str(edge)
			edgeMetrics = metricResults[str(edge)]
			if len(edgeMetrics) == 0:
				continue
			edgeIterPrintMessage = edge.computeIterPrintMessage(i, stepsPerEpoch, \
				metricResults[str(edge)], iterFinishTime)[1 :]
			messages.append(strEdge)
			messages.extend(edgeIterPrintMessage)
		return messages

	# Computes the message that is printed to the stdout. This method is also called by SaveHistory callback.
	# @param[in] kwargs The arguments sent to any regular callback.
	# @return A string that contains the one-line message that is printed at each end of epoch.
	def computePrintMessage(self, trainMetrics, validationMetrics, numEpochs, duration):
		messages = super().computePrintMessage(trainMetrics, validationMetrics, numEpochs, duration)
		for edge in self.edges:
			if type(edge) == Graph:
				strEdge = "SubGraph"
			else:
				strEdge = str(edge)
			if len(edgeMetrics) == 0:
				continue
			edgePrintMessage = edge.computePrintMessage(trainMetrics[str(edge)], validationMetrics[str(edge)], \
				numEpochs, duration)[1 : ]
			messages.append(strEdge)
			messages.extend(edgePrintMessage)
		return messages

	def iterationEpilogue(self, isTraining, isOptimizing, trLabels):
		# Set the GT for each node based on the inputs available at this step. Edges may overwrite this when reaching
		#  a node via an edge, however it is the graph's responsability to set the default GTs. What happens during the
		#  optimization shouldn't be influenced by this default.
		# If the ground truth key is "*", then all items are provided to the node and it's expected that the node will
		#  manage the labels accordingly.
		for node in self.nodes:
			node.setGroundTruth(trLabels)
			node.messages = {}

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

	def __str__(self):
		return self.graphStr()
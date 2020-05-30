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

		# Add metrics
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
			# try:
			# 	edges.extend(edge.getEdges())
			# except Exception:
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

	### Some updates to original NeuralNetworkPyTorch to work seamlessly with graphs (mostly printing)

	def getGroundTruth(self, x):
		return x

	def callbacksOnIterationEnd(self, data, labels, results, loss, iteration, numIterations, \
		metricResults, isTraining, isOptimizing):
		a = super().callbacksOnIterationEnd(data, labels, results, loss, iteration, numIterations, \
				metricResults, isTraining, isOptimizing)
		print(a)

		for edge in self.edges:
			a = edge.callbacksOnIterationEnd(data, edge.getGroundTruth(labels), results[str(edge)], loss, iteration, numIterations, \
				metricResults, isTraining, isOptimizing)
			print(edge, a)
		exit()
		# iterResults = {}
		# print(self.topologicalKeys)
		# exit()
		# for key in self.topologicalKeys:
		# 	# Hack the args so we only use relevant results and labels. Make a list (of all edge outputs), but also
		# 	#  for regular metrics.
		# 	results, iterLoss = [results], loss

		# 	if type(key) == tuple:
		# 		edgeID = key[0]
		# 		edge = self.edgeIDsToEdges[edgeID]
		# 		B = edge.outputNode
		# 		labels = getNpData(B.getGroundTruth())
		# 		results = getNpData(edge.outputs)
		# 		# Some edges may have no loss (are pre-trained, for example)
		# 		iterLoss = None
		# 		if edgeID in self.edgeLoss:
		# 			iterLoss = self.edgeLoss[edgeID]

		# 	metricKwArgs = {"data" : data, "loss" : iterLoss, "iteration" : iteration, \
		# 		"numIterations" : numIterations, "iterResults" : iterResults, \
		# 		"metricResults" : metricResults, "isTraining" : isTraining, "isOptimizing" : isOptimizing
		# 	}

		# 	# Loop through all edge outputs and average results.
		# 	for result in results:
		# 		# iterResults is updated at each step in the order of topological sort
		# 		iterResults[key] = self.callbacks[key].onIterationEnd(result, labels, **metricKwArgs)
		# 		# Add it to running mean only if it's numeric
		# 		try:
		# 			metricResults[key].update(iterResults[key], 1)
		# 		except Exception:
		# 			continue

	def metricsSummary(self):
		metrics = self.getMetrics()
		edges = self.edges
		summaryStr = ""
		for metric in metrics:
			if not metric in self.edgeIDsToEdges:
				summaryStr += "\t- %s (%s)\n" % (metric, metrics[metric].getDirection())
			else:
				edge = self.edgeIDsToEdges[metric]
				summaryStr += edge.metricsSummary()
		return summaryStr

	def getMetrics(self):
		thisCallbacks = super().getMetrics()
		for edge in self.edges:
			thisCallbacks[str(edge)] = edge.getMetrics()
		return thisCallbacks

	def computeIterPrintMessage(self, i, stepsPerEpoch, metricResults, iterFinishTime):
		strMetricResults = {k : metricResults[k] for k in filter(lambda x : type(x) == str, metricResults.keys())}
		messages = super().computeIterPrintMessage(i, stepsPerEpoch, strMetricResults, iterFinishTime)

		edgesMessages = []
		for edge in self.edges:
			condition = lambda x : (x in self.iterPrintMessageKeys) and (x != "Loss")
			printableMetrics = sorted(filter(condition, edge.getMetrics()))
			if len(printableMetrics) == 0:
				continue
			message = "    - [%s] " % (edge)
			for metric in printableMetrics:
				# metric is a tuple of (edge, metricName) for graphs.
				formattedStr = getFormattedStr(metricResults[metric].get(), precision=3)
				metricName = metric[1]
				message += " %s: %s." % (metricName, formattedStr)
			edgesMessages.append(message)

		if len(edgesMessages) > 0:
			messages.append("  - Edge metrics:")
			messages.extend(edgesMessages)

		return messages

	# Computes the message that is printed to the stdout. This method is also called by SaveHistory callback.
	# @param[in] kwargs The arguments sent to any regular callback.
	# @return A string that contains the one-line message that is printed at each end of epoch.
	def computePrintMessage(self, trainMetrics, validationMetrics, numEpochs, duration):
		# Keep the non-edge metrics for the default printer.
		strTrainMetrics = {k : trainMetrics[k] for k in filter(lambda x : type(x) == str, trainMetrics.keys())}
		strValMetrics = None if not validationMetrics else \
			{k : validationMetrics[k] for k in filter(lambda x : type(x) == str, validationMetrics.keys())}
		messages = super().computePrintMessage(strTrainMetrics, strValMetrics, numEpochs, duration)

		messages.append("  - Edge metrics:")
		for edge in self.edges:
			trainMessage, validationMessage = "      - [Train]", "      - [Validation]"
			printableMetrics = filter(lambda x : x in self.iterPrintMessageKeys and (x != "Loss"), edge.getMetrics())
			printableMetrics = sorted(printableMetrics)
			if len(printableMetrics) == 0:
				continue
			for metric in printableMetrics:
				trainMessage += " %s: %s." % (metric[1], getFormattedStr(trainMetrics[metric], precision=3))
				if not validationMetrics is None:
					validationMessage += " %s: %s." % (metric[1], getFormattedStr(validationMetrics[metric], \
						precision=3))
			messages.append("    - [%s] " % (edge))
			messages.append(trainMessage)
			if not validationMetrics is None:
				messages.append(validationMessage)

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

	def __str__(self):
		Str = "Graph:"
		for edge in self.edges:
			Str += "\n\t-%s" % (str(edge))
		return Str
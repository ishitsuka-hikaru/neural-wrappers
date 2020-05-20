import torch.nn as nn
from ..pytorch import NeuralNetworkPyTorch, getNpData, getTrData
from ..utilities import MultiLinePrinter, getFormattedStr
from functools import partial
from copy import copy

from .draw_graph import drawGraph

class Graph(NeuralNetworkPyTorch):
	def __init__(self, edges, hyperParameters={}):
		self.nodes = Graph.getNodes(edges)
		hyperParameters = self.getHyperParameters(hyperParameters, edges)
		super().__init__(hyperParameters=hyperParameters)

		self.edges = nn.ModuleList(edges)
		self.edgeIDsToEdges = {str(edge) : edge for edge in self.edges}
		self.edgeLoss = {}
		self.linePrinter = MultiLinePrinter()

		# Add metrics
		self.addMetrics(self.getEdgesMetrics())
		self.setCriterion(partial(Graph.lossFn, self=self))

	@staticmethod
	def lossFn(y, t, self):
		loss = 0
		for edge in self.edges:
			edgeID = str(edge)
			edgeLoss = edge.loss(y, t)
			if not edgeLoss is None:
				self.edgeLoss[edgeID] = getNpData(edgeLoss)
				loss += edgeLoss
		return loss

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

	def getEdgesMetrics(self):
		metrics = {}
		for edge in self.edges:
			edgeMetrics = edge.getMetrics()
			for metric in edgeMetrics:
				if metric == "Loss":
					continue
				metrics[metric] = edgeMetrics[metric]
		return metrics

	@staticmethod
	def getNodes(edges):
		nodes = set()
		for edge in edges:
			nodes.add(edge.inputNode)
			nodes.add(edge.outputNode)
		return nodes

	### Some updates to original NeuralNetworkPyTorch to work seamlessly with graphs (mostly printing)

	def callbacksOnIterationEnd(self, data, labels, results, loss, iteration, numIterations, \
		metricResults, isTraining, isOptimizing):
		iterResults = {}
		for key in self.topologicalKeys:
			# Hack the args so we only use relevant results and labels. Make a list (of all edge outputs), but also
			#  for regular metrics.
			results, iterLoss = [results], loss

			if type(key) == tuple:
				edgeID = key[0]
				edge = self.edgeIDsToEdges[edgeID]
				B = edge.outputNode
				labels = getNpData(B.getGroundTruth())
				results = getNpData(edge.outputs)
				# Some edges may have no loss (are pre-trained, for example)
				iterLoss = None
				if edgeID in self.edgeLoss:
					iterLoss = self.edgeLoss[edgeID]

			metricKwArgs = {"data" : data, "loss" : iterLoss, "iteration" : iteration, \
				"numIterations" : numIterations, "iterResults" : iterResults, \
				"metricResults" : metricResults, "isTraining" : isTraining, "isOptimizing" : isOptimizing
			}

			# Loop through all edge outputs and average results.
			for result in results:
				# iterResults is updated at each step in the order of topological sort
				iterResults[key] = self.callbacks[key].onIterationEnd(result, labels, **metricKwArgs)
				# Add it to running mean only if it's numeric
				try:
					metricResults[key].update(iterResults[key], 1)
				except Exception:
					continue

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

	def getHyperParameters(self, hyperParameters, edges):
		# Set up hyperparameters for every node
		hyperParameters = {k : hyperParameters[k] for k in hyperParameters}
		for node in self.nodes:
			hyperParameters[node.name] = node.hyperParameters
		for edge in edges:
			hyperParameters[str(edge)] = edge.hyperParameters
		return hyperParameters

	def __str__(self):
		Str = "Graph:"
		for edge in self.edges:
			Str += "\n\t-%s" % (str(edge))
		return Str
import torch.nn as nn
from ..pytorch import NeuralNetworkPyTorch, getNpData
from ..utilities import MultiLinePrinter
from functools import partial
from copy import copy

from .draw_graph import drawGraph

class Graph(NeuralNetworkPyTorch):
	def __init__(self, edges, **kwargs):
		self.nodes = Graph.getNodes(edges)
		# Set up hyperparameters for every node
		for node in self.nodes:
			kwargs[node.name] = node.hyperParameters
		super().__init__(hyperParameters=kwargs)

		self.edges = nn.ModuleList(edges)
		self.edgeIDsToEdges = {str(edge) : edge for edge in self.edges}
		self.edgeLoss = {}
		self.linePrinter = MultiLinePrinter()

		# Add metrics
		self.addMetrics(self.getEdgesMetrics())
		self.setCriterion(partial(Graph.lossFn, self=self))

	def lossFn(y, t, self):
		loss = 0
		for edge in self.edges:
			edgeLoss = edge.loss(y, t)
			self.edgeLoss[edge] = getNpData(edgeLoss)
			loss += edgeLoss
		return loss

	def forward(self, trInputs):
		trResults = {}
		# TODO: Execution order. (synchronus vs asynchronus as well as topological sort at various levels.)
		# For now, the execution is synchronous and linear as defined by the list of edges
		for edge in self.edges:
			# We kind of hacked the metrics of all edges using this class. Perhaps a more modular approach would be to
			#  call run_one_epoch here for each edge.
			edgeID = str(edge)
			edgeOutput = edge.forward(trInputs)
			trResults[edgeID] = edgeOutput
		return trResults

	def getEdgesMetrics(self):
		metrics = {}
		for edge in self.edges:
			for metric in edge.metrics:
				# Store the actual edge object as part of the metric key so we can retrive its nodes.
				newName = (str(edge), metric)
				metrics[newName] = edge.metrics[metric]
		return metrics

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
		for i, key in enumerate(self.topologicalKeys):
			# Hack the args so we only use relevant results and labels. Make a list (of all edge outputs), but also
			#  for regular metrics.
			inputResults, inputLabels, iterLoss = [results], labels, loss

			if type(key) == tuple:
				edgeID = key[0]
				edge = self.edgeIDsToEdges[edgeID]
				B = edge.outputNode
				inputLabels = getNpData(B.getGroundTruth())
				iterLoss = self.edgeLoss[edge]
				if not edgeID in B.outputs:
					continue
				inputResults = getNpData(B.outputs[edgeID])

			metricKwArgs = {"data" : data, "loss" : iterLoss, "iteration" : iteration, \
				"numIterations" : numIterations, "iterResults" : iterResults, \
				"metricResults" : metricResults, "isTraining" : isTraining, "isOptimizing" : isOptimizing
			}

			# Loop through all edge outputs and average results.
			for inputResult in inputResults:
				inputResult = getNpData(inputResult)
				inputLabels = getNpData(inputLabels)
				# iterResults is updated at each step in the order of topological sort
				iterResults[key] = self.callbacks[key].onIterationEnd(inputResult, inputLabels, **metricKwArgs)
				# Add it to running mean only if it's numeric
				try:
					metricResults[key].update(iterResults[key], 1)
				except Exception:
					continue

	def computeIterPrintMessage(self, i, stepsPerEpoch, metricResults, iterFinishTime):
		strMetricResults = {k : metricResults[k] for k in filter(lambda x : type(x) == str, metricResults.keys())}
		messages = super().computeIterPrintMessage(i, stepsPerEpoch, strMetricResults, iterFinishTime)
		for edge in self.edges:
			message = "  - [%s] " % (edge)
			edgeID = str(edge)
			for metric in edge.metrics:
				key = (edgeID, metric)
				if not key in self.iterPrintMessageKeys:
					continue
				message += "%s: %2.3f. " % (metric, metricResults[key].get())
			messages.append(message)
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

		for edge in self.edges:
			message = "  - %s. [Train] " % (edge)
			edgeID = str(edge)
			for metric in edge.metrics:
				key = (edgeID, metric)
				if not key in self.iterPrintMessageKeys:
					continue
				message += "%s: %2.3f. " % (metric, trainMetrics[key])
			if not validationMetrics is None:
				message += " | [Validation] "
				for metric in edge.metrics:
					key = (edgeID, metric)
					if not key in self.iterPrintMessageKeys:
						continue
					message += "%s: %2.3f. " % (metric, validationMetrics[key])
			messages.append(message)
		return messages

	def iterationEpilogue(self, isTraining, isOptimizing, trLabels):
		# Set the GT for each node based on the inputs available at this step. Edges may overwrite this when reaching
		#  a node via an edge, however it is the graph's responsability to set the default GTs. What happens during the
		#  optimization shouldn't be influenced by this default.
		for node in self.nodes:
			if not node.groundTruthKey in trLabels:
				continue
			node.setGroundTruth(trLabels[node.groundTruthKey].detach())

	def iterationPrologue(self, inputs, labels, results, loss, iteration, \
		stepsPerEpoch, metricResults, isTraining, isOptimizing, printMessage, startTime):
		super().iterationPrologue(inputs, labels, results, loss, iteration, stepsPerEpoch, metricResults, \
			isTraining, isOptimizing, printMessage, startTime)

		# Super important step. We need to clean the GTs of the previous step, so trySetNodeGT actually updates it.
		# We can only do it here, because run_one_epoch in NeuralNetworkPyTorch calls this as the last thing at each
		#  iteration.
		for node in self.nodes:
			node.setGroundTruth(None)
			# Important that this is here, otherwise, we keep old items from the graph at the next iteration causing
			#  .backward() to fail (asks for retain_graph). Solution for TimeEdges would be to save outputs with
			#  detach().
			node.outputs = {}
			node.inputs = {}

	def draw(self, fileName, cleanup=True):
		nodes = [x.name for x in self.nodes]
		edges = [(x.inputNode.name, x.outputNode.name) for x in self.edges]
		drawGraph(nodes, edges, fileName, cleanup)

	def __str__(self):
		Str = "Graph:"
		for edge in self.edges:
			Str += "\n\t-%s" % (str(edge))
		return Str
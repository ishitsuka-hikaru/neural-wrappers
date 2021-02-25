import torch as tr
import torch.nn as nn
from datetime import datetime
from overrides import overrides

from .utils import getFormattedStr
from .draw_graph import drawGraph
from .graph_serializer import GraphSerializer
from ..callbacks import CallbackName
from ..metrics import Metric
from ..pytorch import NWModule, npGetData
from ..utilities import Debug

def getNodesFromEdges(edges):
	nodes = set()
	for edge in edges:
		# edge can be an actual Graph.
		for node in edge.getNodes():
			nodes.add(node)
	return nodes

class Graph(NWModule):
	def __init__(self, edges, hyperParameters={}):
		hyperParameters = self.getHyperParameters(hyperParameters, edges)
		super().__init__(hyperParameters=hyperParameters)

		self.edges = nn.ModuleList(edges)
		self.nodes = getNodesFromEdges(edges)
		self.edgeLoss = {}
		self.setCriterion(self.loss)
		self.serializer = GraphSerializer(self)
		self.iterPrintMessageKeys = [CallbackName("Loss")]

	def loss(self, y, t):
		loss = 0
		for edge in self.edges:
			edgeID = str(edge)
			edgeLoss = edge.loss(y, t)
			self.edgeLoss[edgeID] = npGetData(edgeLoss)
	
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

	def setNodesGroundTruth(self, trInputs):
		for node in self.nodes:
			node.messages = {}
			if node.groundTruthKey in trInputs:
				node.setGroundTruth(trInputs)

	@overrides
	def networkAlgorithm(self, trInputs, trLabels, isTraining, isOptimizing):
		assert not self.criterion is None, "Set criterion before training or testing"

		# TODO: Execution order. (synchronus vs asynchronus as well as topological sort at various levels.)
		# For now, the execution is synchronous and linear as defined by the list of edges
		trResults = self.forward(trInputs)
		trLoss = self.criterion(trResults, trLabels)
		self.updateOptimizer(trLoss, isTraining, isOptimizing)
		return trResults, trLoss

	def forward(self, x):
		trResults = {}
		for edge in self.edges:
			edgeID = str(edge)
			edgeInputs = edge.getInputs(x)
			edgeOutput = edge.forward(edgeInputs)
			# Update the outputs of the whole graph as well
			trResults[edgeID] = edgeOutput
		return trResults

	def getEdges(self):
		edges = []
		for edge in self.edges:
			edges.append(edge)
		return edges

	def getEdge(self, edgeName:str):
		found = False
		for edge in self.edges:
			if edge.name == edgeName:
				found = True
				break
		assert found == True, "Couldn't find edge %s" % edgeName
		return edge

	def getNodes(self):
		return getNodesFromEdges(self.edges)

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

	def iterationPrologue(self, inputs, labels, results, loss, iteration, \
		stepsPerEpoch, metricResults, isTraining, isOptimizing, startTime):
		# metrics and callbacks are merged. Each callback/metric can have one or more "parents" which
		#  forms an ayclical graph. They must be called in such an order that all the parents are satisfied before
		#  all children (topological sort).
		# Iteration callbacks are called here. These include metrics or random callbacks such as plotting results
		#  in testing mode.
		self.callbacksOnIterationEnd(inputs, labels, results=results, \
			loss=loss, iteration=iteration, numIterations=stepsPerEpoch, metricResults=metricResults, \
			isTraining=isTraining, isOptimizing=isOptimizing)

		# Print the message, after the metrics are updated.
		iterFinishTime = (datetime.now() - startTime)
		if Debug.getLogLevel() >= 2:
			message = self.computeIterPrintMessage(iteration, stepsPerEpoch, metricResults, iterFinishTime)
			Debug.print(message)

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

	def metricsStr(metrics):
		message = ""
		metricKeys = filter(lambda x : isinstance(x, CallbackName), metrics.keys())
		Keys = sorted(list(set(metricKeys)))
		for key in Keys:
			formattedStr = getFormattedStr(metrics[key], precision=3)
			message += " %s: %s." % (key, formattedStr)
		return message

	@overrides
	def getMetric(self, metricName) -> Metric:
		if isinstance(metricName, CallbackName):
			metricName = metricName.name
		if isinstance(metricName, tuple) and len(metricName) == 1:
			metricName = metricName[0]

		if isinstance(metricName, tuple):
			edge = self.getEdge(metricName[0])
			innerName = metricName[1 :]
			if len(innerName) == 1:
				innerName = innerName[0]
			return edge.getMetric(innerName)
		else:
			return super().getMetric(metricName)

	def trainValMetricsStr(trainMetrics, validationMetrics, depth):
		def padding(depth):
			return "  " * depth
		
		messages = []
		if len(trainMetrics) > 0:
			messages.append("%s- Metrics:" % padding(depth + 1))
			trainMessage = "%s- [Train] %s" % (padding(depth + 2), Graph.metricsStr(trainMetrics))
			messages.append(trainMessage)
			if not validationMetrics is None:
				validationMessage = "%s- [Validation] %s" % (padding(depth + 2), Graph.metricsStr(validationMetrics))
				messages.append(validationMessage)
		return messages

	def computeIterPrintMessage(self, i, stepsPerEpoch, metricResults, iterFinishTime, depth=0):
		def padding(depth):
			return "  " * depth

		messages = []
		if depth == 0:
			# iterFinishTime / (i + 1) is the current estimate per iteration. That value times stepsPerEpoch is
			#  the current estimation per epoch. That value minus current time is the current estimation for
			#  time remaining for this epoch. It can also go negative near end of epoch, so use abs.
			ETA = abs(iterFinishTime / (i + 1) * stepsPerEpoch - iterFinishTime)

			message = "Epoch: %d. Iteration: %d/%d. ETA: %s" % (self.currentEpoch, i + 1, stepsPerEpoch, ETA)
			messages.append(message)
			if self.optimizer:
				messages.append("  - Optimizer: %s" % self.getOptimizerStr())

		if len(metricResults) > 0:
			messages.append("%s- Metrics: %s" % (padding(depth + 1), Graph.metricsStr(metricResults)))

		for edge in self.edges:
			edgeMetrics = metricResults[str(edge)]
			if len(edgeMetrics) == 0:
				continue

			if isinstance(edge, Graph):
				messages.append("%s- [SubGraph]" % padding(depth + 1))
				subGraphMessage = edge.computeIterPrintMessage(None, None, edgeMetrics, None, depth + 1)
				messages.extend(subGraphMessage)
			else:
				messages.append("%s- [%s]" % (padding(depth + 1), str(edge)))
				messages.append("%s- Metrics: %s" % (padding(depth + 2), Graph.metricsStr(edgeMetrics)))

		return messages

	def newtorkComputePrintMessage(self, trainMetrics, validationMetrics, numEpochs, duration):
		messages = []
		done = self.currentEpoch / numEpochs * 100
		message = "Epoch %d/%d. Done: %2.2f%%. Took: %s." % (self.currentEpoch, numEpochs, done, duration)
		messages.append(message)

		if self.optimizer:
			messages.append("  - Optimizer: %s" % self.getOptimizerStr())

		if len(trainMetrics) == 0:
			return messages

		messages.append("  - Metrics:")
		# trainMetrics = dict(filter(lambda x, y : isinstance(x, CallbackName), trainMetrics.items()))
		trainMetrics = {k : trainMetrics[k] \
			for k in filter(lambda x : isinstance(x, CallbackName), trainMetrics)}
		printableMetrics = filter(lambda x : x in self.iterPrintMessageKeys, sorted(trainMetrics))
		trainMessage, validationMessage = "    - [Train]", "    - [Validation]"
		for metric in printableMetrics:
			formattedStr = getFormattedStr(trainMetrics[metric], precision=3)
			trainMessage += " %s: %s." % (metric, formattedStr)
			if not validationMetrics is None:
				formattedStr = getFormattedStr(validationMetrics[metric], precision=3)
				validationMessage += " %s: %s." % (metric, formattedStr)
		messages.append(trainMessage)
		if not validationMetrics is None:
			messages.append(validationMessage)
		return messages

	# Computes the message that is printed to the stdout. This method is also called by SaveHistory callback.
	# @param[in] kwargs The arguments sent to any regular callback.
	# @return A string that contains the one-line message that is printed at each end of epoch.
	def computePrintMessage(self, trainMetrics, validationMetrics, numEpochs, duration, depth=0):
		def padding(depth):
			return "  " * depth

		messages = []
		if depth == 0:
			done = self.currentEpoch / numEpochs * 100
			message = "Epoch %d/%d. Done: %2.2f%%. Took: %s." % (self.currentEpoch, numEpochs, done, duration)
			messages.append(message)

			if self.optimizer:
				messages.append("  - Optimizer: %s" % self.getOptimizerStr())

		if len(trainMetrics) > 0:
			messages.extend(Graph.trainValMetricsStr(trainMetrics, validationMetrics, depth))

		for edge in self.edges:
			edgeTrainMetrics = trainMetrics[str(edge)]
			edgeValMetrics = validationMetrics[str(edge)] if not validationMetrics is None else None
			if len(edgeTrainMetrics) == 0:
				continue

			if isinstance(edge, Graph):
				messages.append("%s- [SubGraph]" % padding(depth + 1))
				subGraphMessage = edge.computePrintMessage(edgeTrainMetrics, validationMessage, None, None, depth + 1)
				messages.extend(subGraphMessage)
			else:
				messages.append("%s- [%s]" % (padding(depth + 1), str(edge)))
				messages.extend(Graph.trainValMetricsStr(edgeTrainMetrics, edgeValMetrics, depth + 1))

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
		for edge in edges:
			hyperParameters[str(edge)] = edge.hyperParameters
		for node in getNodesFromEdges(edges):
			hyperParameters[node.name] = node.hyperParameters
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
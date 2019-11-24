import torch.nn as nn
from ..pytorch import NeuralNetworkPyTorch, getNpData
from ..utilities import MultiLinePrinter
from functools import partial

class Graph(NeuralNetworkPyTorch):
	def __init__(self, edges):
		super().__init__()
		self.edges = nn.ModuleList(edges)
		self.nodes = self.getNodes()
		self.edgeLoss = {}
		self.linePrinter = MultiLinePrinter()

		# Add metrics
		self.addMetrics(self.getEdgesMetrics())
		self.setCriterion(partial(Graph.lossFn, self=self))

	def lossFn(y, t, self):
		loss = 0
		for edge in self.edges:
			edgeLoss = edge.lossFn(t)
			self.edgeLoss[edge] = getNpData(edgeLoss)
			loss += edgeLoss
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

			# We kind of hacked the metrics of all edges using this class. Perhaps a more modular approach would be to
			#  call run_one_epoch here for each edge.
			edgeOutput = edge.forward(trInputs)

			trResults[edge] = edgeOutput

		# print("_____________________________")
		trLoss = self.criterion(trResults, trLabels)
		return trResults, trLoss

	def getEdgesMetrics(self):
		metrics = {}
		for edge in self.edges:
			# metrics[edge] = edge.metrics
			for metric in edge.metrics:
				# newName = "%s %s" % (edge, metric)
				newName = (edge, metric)
				metrics[newName] = edge.metrics[metric]
		return metrics

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

	### Some updates to original NeuralNetworkPyTorch to work seamlessly with graphs (mostly printing)

	def callbacksOnIterationEnd(self, data, labels, results, loss, iteration, numIterations, \
		metricResults, isTraining, isOptimizing):
		iterResults = {}
		for i, key in enumerate(self.topologicalKeys):
			# Hack the args so we only use relevant results and labels. Make a list (of all edge outputs), but also
			#  for regular metrics.
			inputResults, inputLabels, iterLoss = [results], labels, loss
			if type(key) == tuple:
				edge = key[0]
				edgeID = str(edge)
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
		messages = []
		message = "Iteration: %d/%d." % (i + 1, stepsPerEpoch)
		if self.optimizer:
			message += "LR: %2.5f." % (self.optimizer.state_dict()["param_groups"][0]["lr"])
		# iterFinishTime / (i + 1) is the current estimate per iteration. That value times stepsPerEpoch is
		#  the current estimation per epoch. That value minus current time is the current estimation for
		#  time remaining for this epoch. It can also go negative near end of epoch, so use abs.
		ETA = abs(iterFinishTime / (i + 1) * stepsPerEpoch - iterFinishTime)
		message += " ETA: %s" % (ETA)
		messages.append(message)

		for edge in self.edges:
			message = "  - [%s] " % (edge)
			for metric in edge.metrics:
				key = (edge, metric)
				if not key in self.iterPrintMessageKeys:
					continue
				message += "%s: %2.3f. " % (metric, metricResults[key].get())
			messages.append(message)
		return messages

	# Computes the message that is printed to the stdout. This method is also called by SaveHistory callback.
	# @param[in] kwargs The arguments sent to any regular callback.
	# @return A string that contains the one-line message that is printed at each end of epoch.
	def computePrintMessage(self, trainMetrics, validationMetrics, numEpochs, duration):
		messages = []
		done = self.currentEpoch / numEpochs * 100
		message = "Epoch %d/%d. Done: %2.2f%%." % (self.currentEpoch, numEpochs, done)
		if self.optimizer:
			message += " LR: %2.5f." % (self.optimizer.state_dict()["param_groups"][0]["lr"])

		message += " Took: %s." % (duration)
		messages.append(message)

		for edge in self.edges:
			message = "  - [%s] " % (edge)
			for metric in edge.metrics:
				key = (edge, metric)
				if not key in self.iterPrintMessageKeys:
					continue
				message += "%s: %2.3f. " % (metric, trainMetrics[key])
			if not validationMetrics is None:
				for metric in edge.metrics:
					key = (edge, metric)
					if not key in self.iterPrintMessageKeys:
						continue
				message += "Val %s: %2.3f. " % (metric, validationMetrics[key])
			messages.append(message)
		return messages

	def __str__(self):
		Str = "Graph:"
		for edge in self.edges:
			Str += "\n\t-%s" % (str(edge))
		return Str
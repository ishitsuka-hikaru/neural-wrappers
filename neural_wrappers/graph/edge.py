import torch as tr
import torch.nn as nn
from functools import partial
from .node import MapNode, VectorNode
from ..pytorch import FeedForwardNetwork, trModuleWrapper, trGetData, npGetData, trDetachData
from .graph_serializer import GraphSerializer
from ..callbacks import CallbackName
from overrides import overrides

# Default loss of this edge goes through all ground truths and all outputs of the output node and computes the
#  loss between them. This can be updated for a more specific edge algorithm for loss computation.
def defaultLossFn(y, t, obj):
	B = obj.outputNode
	L = 0
	t = obj.getGroundTruth(t)
	for y in obj.outputs:
		res = obj.criterion(y, t)
		if not res is None:
			L += obj.criterion(y, t)
	return L

# @param[in] inputNode Instance of the input node of this edge
# @param[in] outputNode Instance of the output node of this edge
# @param[in] edgeType The type of edge. Available options are: node-node, node-edge, edge-node, edge-edge. In the node
#  cases, we'll use the node specific encoder/decoder for this edge, while in the edge cases, the edge specific
#  encoder/decoder. It's up to the node to implement those, however the edge may have to adapt the results in a custom
#  forward function.
# @param[in] forwardFn Custom forward function. If not set, we'll use forwardUseAll which passes forward all available
#  inputs at the inputNode
# @param[in] lossFn Custom loss function. If not set, we'll use the default loss function which uses all outputs at the
#  output node and call the output node's loss function for each of them.
# @param[in] dependencies A list of edge dependenices. This is used for topological sort during each iteration.
# @param[in] blockGradients If set to true, each output of this edge will be owned by the outputNode, rather than
#  maintaing a history of its origin. This is used s.t. long graphs don't have to backpropagate to the source of each
#  input.
class Edge(FeedForwardNetwork):
	def __init__(self, inputNode, outputNode, name=None, useMetrics=True, useLoss=True, edgeType="edge-edge", \
		forwardFn=None, lossFn=None, dependencies=[], blockGradients=False, hyperParameters={}):
		hyperParameters = self.getHyperParameters(hyperParameters, edgeType, blockGradients)
		self.iterPrintMessageKeys = [CallbackName("Loss")]
		self.strInputNode = str(inputNode)
		self.strOutputNode = str(outputNode)
		if name is None:
			name = "%s -> %s" % (self.strInputNode, self.strOutputNode)
		self.name = name
		super().__init__(hyperParameters=hyperParameters)

		assert edgeType in ("node-node", "node-edge", "edge-node", "edge-edge")
		self.inputNode = inputNode
		self.outputNode = outputNode
		self.edgeType = edgeType

		# Model stuff
		self.edgeID = str(self)
		self.model = None
		self.forwardFn = forwardFn
		self.lossFn = lossFn
		self.useLoss = useLoss
		self.useMetrics = useMetrics
		self.setupModel()

		self.inputs = []
		self.outputs = []

		self.dependencies = dependencies
		self.setBlockGradients(blockGradients)
		self.serializer = GraphSerializer(self)

	def getInputs(self, x):
		self.fullGT = x
		# print("[Edge::getInputs]", type(self.inputNode), type(self.inputNode).mro(), self.inputNode.getInputs(x))
		inputs = self.inputNode.getInputs(x)
		if self.blockGradients:
			inputs = trDetachData(inputs)
		return inputs

	def forward(self, x):
		self.inputs = x
		res = self.forwardFn(self, x)
		self.outputs = res
		self.outputNode.addMessage(self, res)
		return self.outputs

	def loss(self, y, t):
		return self.lossFn(y, t)

	def getGroundTruth(self, x):
		return self.outputNode.getGroundTruthInput(x)

	# Creates the encoder for this edge. If the edge is node-node or node-edge, then use the node-spepcific encoder.
	def getEncoder(self):
		assert self.model is None
		A, B = self.inputNode, self.outputNode
		# If node-* edge, use the edge-specificic encoder
		if self.edgeType in ("node-node", "node-edge"):
			# If it wasn't instanciated yet, create it.
			if not A.nodeEncoder:
				A.nodeEncoder = A.getEncoder(None)
			return A.nodeEncoder
		else:
		# If edge-* edge, then instantiate a new encoder for the output node type
			return A.getEncoder(B.getType())

	# Creates the encoder for this edge. If the edge is node-node or node-edge, then use the node-spepcific encoder.
	def getDecoder(self):
		assert self.model is None
		A, B = self.inputNode, self.outputNode
		# If node-* edge, use the edge-specificic decoder
		if self.edgeType in ("node-node", "edge-node"):
			# If it wasn't instanciated yet, create it.
			if not B.nodeDecoder:
				B.nodeDecoder = B.getDecoder(None)
			return B.nodeDecoder
		else:
		# If *-node edge, then instantiate a new decoder for the output node type
			return B.getDecoder(A.getType())

	def getModel(self):
		if not self.model:
			self.setupModel()
		return self.model

	def getNodes(self):
		return [self.inputNode, self.outputNode]

	def computePrintMessage(self, trainMetrics, validationMetrics, numEpochs, duration):
		from .utils import getFormattedStr
		messages = []
		done = self.currentEpoch / numEpochs * 100
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

	@overrides
	def addMetric(self, metricName, metric):
		super().addMetric(metricName, metric)
		self.iterPrintMessageKeys.append(metricName)

	# Default model for this edge is just a sequential mapping between the A's encoder and B's decoder.
	#  Other edges may requires additional edge-specific parameters or some more complicated views to convert the
	#   output of A's encoder to the input of B's decoder.
	def setupModel(self):
		assert self.model is None
		self.model = trModuleWrapper(nn.Sequential(self.getEncoder(), self.getDecoder()))

		# Set the forward/loss functions for this edge as well.
		if not self.forwardFn:
			from .utils import forwardUseAll
			self.forwardFn = forwardUseAll

		if not self.lossFn:
			self.lossFn = partial(defaultLossFn, obj=self)

		if not self.useLoss:
			self.lossFn = lambda y, t : 0

		if self.useMetrics:
			self.addMetrics(self.outputNode.getMetrics())
		self.setCriterion(self.outputNode.getCriterion())

	def setBlockGradients(self, value):
		self.blockGradients = value

	def getHyperParameters(self, hyperParameters, edgeType, blockGradients):
		# Without this shallow copy we risk of having other references to hyperparameters.
		hyperParameters = {k : hyperParameters[k] for k in hyperParameters.keys()}
		hyperParameters["edgeType"] = edgeType
		hyperParameters["blockGradients"] = blockGradients
		return hyperParameters

	def callbacksOnIterationEnd(self, data, labels, results, loss, iteration, numIterations, \
		metricResults, isTraining, isOptimizing):
		results = list(results.values()) if isinstance(results, dict) else results
		for i in range(len(results)):
			metricResults = super().callbacksOnIterationEnd(data, labels, results[i], loss, iteration, \
				numIterations, metricResults, isTraining, isOptimizing)
		return metricResults

	def __str__(self):
		return self.name

	def __repr__(self):
		return str(self)

import torch.nn as nn
import numpy as np
from functools import partial
from collections import OrderedDict
from overrides import overrides
from typing import Union, Callable
from types import LambdaType

from .node import MapNode, VectorNode
from ..pytorch import FeedForwardNetwork, trModuleWrapper
from ..pytorch.network_serializer import NetworkSerializer
from ..callbacks import CallbackName
from ..metrics import Metric, MetricWrapper

# Default loss of this edge goes through all ground truths and all outputs of the output node and computes the
#  loss between them. This can be updated for a more specific edge algorithm for loss computation.
def defaultLossFn(y, t, obj):
	B = obj.outputNode
	L = 0
	t = B.getGroundTruth()
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
	def __init__(self, inputNode, outputNode, name=None, edgeType="edge-edge", forwardFn=None, \
		lossFn=None, dependencies=[], blockGradients=False, hyperParameters={}):
		hyperParameters = self.getHyperParameters(hyperParameters, edgeType, blockGradients)
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
		self.setupModel()

		self.inputs = []
		self.outputs = []

		self.dependencies = dependencies
		self.setBlockGradients(blockGradients)

	def getInputs(self, x):
		# print("[Edge::getInputs]", type(self.inputNode), type(self.inputNode).mro(), self.inputNode.getInputs(x))
		inputs = self.inputNode.getInputs(x)
		if self.blockGradients:
			inputs = {k : inputs[k].detach() for k in inputs}
		return inputs

	def forward(self, x):
		self.inputs = x
		res = self.forwardFn(self, x)
		self.outputs = res
		self.outputNode.addMessage(self, res)
		return self.outputs

	def loss(self, y, t):
		return self.lossFn(y, t)

	def getStrMapping(self):
		return str(self)

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

		self.addMetrics(self.outputNode.getNodeMetrics())
		self.setCriterion(self.outputNode.getNodeCriterion())

	def setBlockGradients(self, value):
		self.blockGradients = value

	def getHyperParameters(self, hyperParameters, edgeType, blockGradients):
		# Without this shallow copy we risk of having other references to hyperparameters.
		hyperParameters = {k : hyperParameters[k] for k in hyperParameters.keys()}
		hyperParameters["edgeType"] = edgeType
		hyperParameters["blockGradients"] = blockGradients
		return hyperParameters

	@overrides
	def callbacksOnIterationEnd(self, data, labels, results, loss, iteration, numIterations, \
		metricResults, isTraining, isOptimizing):
		for i in range(len(results)):
			metricResults = super().callbacksOnIterationEnd(data, labels, results[i], loss, iteration, \
				numIterations, metricResults, isTraining, isOptimizing)
		return metricResults

	# TODO: Remove this and fix loadModel for partial edges.
	def loadPretrainedEdge(self, path):
		thisInputNode = self.inputNode.name.split("(")[0][0 : -1]
		thisOutputNode = self.outputNode.name.split("(")[0][0 : -1]

		print("Attempting to load pretrained edge %s from %s" % (self, path))
		pklFile = NetworkSerializer.readPkl(path)
		# Do a sanity check that this loaded model is a single_link containing desired edge
		# Some parsing to find the relevant edge of the pkl file
		relevantKeys = list(filter(lambda x : x.find("->") != -1, pklFile["model_state"].keys()))
		relevantKeys = list(map(lambda x : x.split("->"), relevantKeys))
		relevantKeys = list(map(lambda x : (x[0].split(" ")[0], x[1][1 : ].split(" ")[0]), relevantKeys))[0]
		if relevantKeys[0] != thisInputNode:
			print("Warning! Input node is different. Expected: %s. Got: %s." % (relevantKeys[0], thisInputNode))
		if relevantKeys[1] != thisOutputNode:
			print("Warning! Output node is different. Expected: %s. Got: %s." % (relevantKeys[1], thisOutputNode))
		self.serializer.doLoadWeights(pklFile)

	# We also override some methods on the Network class so it works with edges as well.
	@overrides
	def addMetric(self, metricName:Union[str, CallbackName], metric:Union[Callable, Metric]):
		if isinstance(metricName, str): #type: ignore
			metricName = CallbackName(metricName) #type: ignore
		metricName = CallbackName((str(self), *metricName.name)) #type: ignore
		super().addMetric(metricName, metric)

	@overrides
	def clearCallbacks(self):
		metric = MetricWrapper(lambda y, t, **k : k["loss"])
		metricName = CallbackName((str(self), "Loss"))
		metric.setName(metricName)
		self.callbacks = OrderedDict({metricName : metric})
		self.iterPrintMessageKeys = [metricName]
		self.topologicalSort = np.array([0], dtype=np.uint8)
		self.topologicalKeys = np.array([metricName], dtype=str)
		self.topologicalSortDirty = False

	def __str__(self):
		return self.name

	def __repr__(self):
		return str(self)
import torch.nn as nn
from functools import partial
from .node import MapNode, VectorNode
from ..pytorch import NeuralNetworkPyTorch, trModuleWrapper, getTrData
from ..pytorch.network_serializer import NetworkSerializer

# Default loss of this edge goes through all ground truths and all outputs of the output node and computes the
#  loss between them. This can be updated for a more specific edge algorithm for loss computation.
def defaultLossFn(self, y, t):
	B = self.outputNode
	L = 0
	t = B.getGroundTruth()
	for y in self.outputs:
		L += self.criterion(y, t)
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
# @param[in] dependencies A list of edge dependenices. This is used for topological sorot during each iteration.
# @param[in] blockGradients If set to true, each output of this edge will be owned by the outputNode, rather than
#  maintaing a history of its origin. This is used s.t. long graphs don't have to backpropagate to the source of each
#  input.
class Edge(NeuralNetworkPyTorch):
	def __init__(self, inputNode, outputNode, edgeType="edge-edge", forwardFn=None, \
		lossFn=None, dependencies=[], blockGradients=False, hyperParameters={}):
		hyperParameters = self.getHyperParameters(hyperParameters, edgeType, blockGradients)
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
		ret = self.lossFn(self, y, t)
		return ret

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

	# Default model for this edge is just a sequential mapping between the A's encoder and B's decoder.
	#  Other edges may requires additional edge-specific parameters or some more complicated views to convert the
	#   output of A's encoder to the input of B's decoder.
	def setupModel(self):
		assert self.model is None
		self.model = trModuleWrapper(nn.Sequential(self.getEncoder(), self.getDecoder()))
		metrics = self.outputNode.getMetrics()
		criterion = self.outputNode.getCriterion()

		# Update metrics' name to be a tuple of (edge, originalName)
		newMetrics = {}
		for key in metrics:
			# Very important to use str(self), not self here, otherwise the metric's name is not pickleable (as it'd
			#  try to store the edge itself, which isn't pickleable)
			newName = (str(self), key)
			newMetrics[newName] = metrics[key]

		# Set the forward/loss functions for this edge as well.
		if not self.forwardFn:
			from .utils import forwardUseAll
			self.forwardFn = forwardUseAll
		if not self.lossFn:
			self.lossFn = defaultLossFn
		self.addMetrics(newMetrics)
		self.setCriterion(criterion)

	def setBlockGradients(self, value):
		self.blockGradients = value

	def getHyperParameters(self, hyperParameters, edgeType, blockGradients):
		# Without this shallow copy we risk of having other references to hyperparameters.
		hyperParameters = {k : hyperParameters[k] for k in hyperParameters.keys()}
		hyperParameters["edgeType"] = edgeType
		hyperParameters["blockGradients"] = blockGradients
		return hyperParameters

	def loadPretrainedEdge(self, path, trainable=True):
		thisInputNode = self.inputNode.name.split("(")[0][0 : -1]
		thisOutputNode = self.outputNode.name.split("(")[0][0 : -1]

		print("Attempting to load pretrained edge %s from %s (Trainable: %s)" % (self, path, trainable))
		pklFile = NetworkSerializer.readPkl(path)
		# Do a sanity check that this loaded model is a single_link containing desired edge
		# Some parsing to find the relevant edge of the pkl file
		relevantKeys = list(filter(lambda x : x.find("->") != -1, pklFile["model_state"].keys()))
		relevantKeys = list(map(lambda x : x.split("->"), relevantKeys))
		assert len(relevantKeys) == len(list(filter(lambda x : len(x) == 2, relevantKeys)))
		relevantKeys = list(map(lambda x : (x[0].split(" ")[0], x[1][1 : ].split(" ")[0]), relevantKeys))
		check2 = list(filter(lambda x : x[0] == thisInputNode and x[1] == thisOutputNode, relevantKeys))
		assert len(check2) == 1, "More than 1 %s->%s edges were found: %s" % (thisInputNode, thisOutputNode, check2)
		self.serializer.doLoadWeights(pklFile)
		self.setTrainableWeights(trainable)

	def __str__(self):
		return "%s -> %s" % (str(self.inputNode), str(self.outputNode))

	def __repr__(self):
		return str(self)
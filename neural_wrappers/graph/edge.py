import torch.nn as nn
from functools import partial
from .node import MapNode, VectorNode
from ..pytorch import NeuralNetworkPyTorch
from .utils import forwardUseAll

# Default loss of this edge goes through all ground truths and all outputs of the output node and computes the
#  loss between them. This can be updated for a more specific edge algorithm for loss computation.
def defaultLossFn(self, y, t):
	A, B, edgeID = self.inputNode, self.outputNode, self.edgeID
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
		self.setupModel(forwardFn, lossFn)
		self.inputs = []
		self.outputs = []

		self.dependencies = dependencies
		self.setBlockGradients(blockGradients)

	def forward(self, x):
		return self.forwardFn(self, x)

	def loss(self, y, t):
		return self.lossFn(self, y, t)

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
			return A.getEncoder(type(B))

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
			return B.getDecoder(type(A))

	def getModel(self):
		model = nn.Sequential(
			self.getEncoder(),
			self.getDecoder()
		)
		# self.addMetrics(model[-1].getMetrics())
		metrics = model[-1].getMetrics()
		if "Loss" in metrics:
			del metrics["Loss"]

		self.addMetrics(metrics)
		self.setCriterion(model[-1].criterion)
		return model

	# Default model for this edge is just a sequential mapping between the A's encoder and B's decoder.
	#  Other edges may requires additional edge-specific parameters or some more complicated views to convert the
	#   output of A's encoder to the input of B's decoder.
	def setupModel(self, forwardFn, lossFn):
		assert self.model is None
		self.model = self.getModel()

		# Set the forward/loss functions for this edge as well.
		if not forwardFn:
			forwardFn = forwardUseAll
		if not lossFn:
			lossFn = defaultLossFn
		self.forwardFn = forwardFn
		self.lossFn = lossFn

	def setBlockGradients(self, value):
		self.blockGradients = value

	def getHyperParameters(self, hyperParameters, edgeType, blockGradients):
		# Without this shallow copy we risk of having other references to hyperparameters.
		hyperParameters = {k : hyperParameters[k] for k in hyperParameters.keys()}
		hyperParameters["edgeType"] = edgeType
		hyperParameters["blockGradients"] = blockGradients
		return hyperParameters

	def __str__(self):
		return "%s -> %s" % (str(self.inputNode), str(self.outputNode))

	def __repr__(self):
		return str(self)
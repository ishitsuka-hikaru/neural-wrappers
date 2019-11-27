import numpy as np
from copy import copy

def pickTypeFromMRO(Type, switchType):
	Type = type(Type) if type(Type) != type else Type
	typeMRO = Type.mro()
	for Type in typeMRO:
		if Type in switchType:
			return switchType[Type]
	assert False, "%s not in %s" % (typeMRO, switchType)

class Node:
	# A dictionary that gives a unique tag to all nodes by appending an increasing number to name.
	lastNodeID = 0
	unicityNodesDict = {}

	def __init__(self, name, groundTruthKey, backPropIntermediateResults=True):
		assert not name is "GT", "GT is a reserved keyword"
		self.name = Node.getUniqueName(name)
		self.groundTruthKey = groundTruthKey

		# This parameter gives us the option to backprop any input back to its original source (or to the first node
		#  that has backPropIntermediateResults=False). If it's set to True, all results that appear in 
		self.backPropIntermediateResults = backPropIntermediateResults

		# Each node must implement an encoder and a decoder. The first one will transform the given input to some
		#  node specific representation, while the second one wiill decode incoming representations.
		# These dictionaries contain the models for all outgoing (encoders) and incoming (decoders) edges of this node.
		self.encoders = {}
		self.decoders = {}

		# This must be set at every iteration before calling getInputs and lossFn in Edge.
		self.groundTruth = self.setGroundTruth(None)
		self.outputs = {}

	# Adds an encoder to the encoders dictionary. If edgeType if node-node or node-edge, reuse the node-specific
	#  encoder. If it is not yet defined, instatiate it first using getEncoder(None).
	def addEncoder(self, outputNodeType, edgeID, edgeType):
		assert edgeType in ("node-node", "node-edge", "edge-node", "edge-edge"), "Got %s" % (edgeType)
		assert edgeID != "node"
		assert not edgeID in self.encoders, "An encoder is alredy defined for this edge."

		if edgeType in ("node-node", "node-edge"):
			if not "node" in self.encoders:
				self.encoders["node"] = self.getEncoder(None)
			model = self.encoders["node"]
		else:
			model = self.getEncoder(outputNodeType)
		self.encoders[edgeID] = model

	# Adds a decoder to the decoders dictionary. If edgeType if edge-edge or node-edge, reuse the node-specific
	#  decoder. If it is not yet defined, instatiate it first using getDecoder(None).
	def addDecoder(self, inputNodeType, edgeID, edgeType):
		assert edgeType in ("node-node", "node-edge", "edge-node", "edge-edge")
		assert edgeID != "node"
		assert not edgeID in self.decoders, "A decoder is alredy defined for this edge."

		if edgeType in ("node-node", "dege-node"):
			if not "node" in self.decoders:
				self.decoders["node"] = self.getDecoder(None)
			model = self.decoders["node"]
		else:
			model = self.getDecoder(inputNodeType)
		self.decoders[edgeID] = model

	def getEncoder(self, outputNodeType=None):
		raise Exception("Must be implemented by each node!")

	def getDecoder(self, inputNodeType=None):
		raise Exception("Must be implemented by each node!")

	# This node's inputs based on whatever GT data we receive (inputs dict + self.groundTruthKey) as well as whatever
	#  intermediate messages we recieved. This is programmable for every node. By default, we return all GTs and all
	#  received messages as possible inputs to the node's forward function
	def getInputs(self):
		nodeInputs, nodeKeys = [], []
		if not self.groundTruth is None:
			nodeInputs.append(self.groundTruth)
			nodeKeys.append("GT")

		for key in self.outputs.keys():
			edgeOutputs = self.outputs[key]
			# Very important. Some nodes may not want to backpropagate to the original link where this output was
			#  generated, as this could increase time/memory very much, especially for very long graphs. Thus, each
			#  node has the ability to cut the input and use it as it was simply generated from here. Optimization is
			#  thus done only with current node's weights (not self.outputs[key]'s inputNode or even its ancestors)
			if self.backPropIntermediateResults:
				edgeOutputs = [x.detach() for x in edgeOutputs]
			nodeInputs.extend(edgeOutputs)
			nodeKeys.extend([key] * len(edgeOutputs))
		return nodeInputs, nodeKeys

	def setGroundTruth(self, groundTruth):
		# Ground truth is always detached from the graph, so we don't optimize both sides of the graph, if the GT of
		#  one particular node was generated from other side.
		self.groundTruth = groundTruth
		if not self.groundTruth is None:
			self.groundTruth.detach_()

	def getGroundTruth(self):
		return self.groundTruth

	def clearNodeOutputs(self):
		self.outputs = {}

	def getUniqueName(name):
		name = "%s (ID: %d)" % (name, Node.lastNodeID)
		Node.lastNodeID += 1
		return name

	def __str__(self):
		return self.name

class VectorNode(Node): pass
class MapNode(Node): pass
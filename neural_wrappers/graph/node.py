import numpy as np
from copy import copy

class Node:
	# A dictionary that gives a unique tag to all nodes by appending an increasing number to name.
	lastNodeID = 0
	unicityNodesDict = {}

	def __init__(self, name, groundTruthKey, modelTypes, lossFn, metrics, modelKwArgs):
		assert not name is "GT", "GT is a reserved keyword"
		self.name = Node.getUniqueName(name)
		self.groundTruthKey = groundTruthKey

		# Model types. Each node must update this dictionary s.t. edges can implement the right type of models
		# Key is the other node and this node is the outputNode.
		#  Example: This node is a VectorNode, we tackle the MapNode neighbours: MapNode ----> THIS
		#  modelTypes = {MapNode: ModelMapToNumber}
		# We can make more specific implementations, such as a specific model for RGB nodes pointing to this node.
		self.modelTypes = modelTypes
		self.metrics = metrics
		self.lossFn = lossFn
		self.modelKwArgs = modelKwArgs

		# This must be set at every iteration before calling getInputs and lossFn in Edge.
		self.groundTruth = self.setGroundTruth(None)
		self.outputs = {}

	def getModel(self, inputNode):
		inputNodeAncestors = type(inputNode).mro()
		found = False
		for Type in inputNodeAncestors:
			if Type in self.modelTypes:
				found = True
				break
		if not found:
			assert False, "Couldn't find a model for type: %s out of the given models: %s" % (inputNodeAncestors, \
				self.modelTypes)
		model = self.modelTypes[Type](**inputNode.modelKwArgs)
		model.addMetrics(self.metrics)
		return model

	# This node's inputs based on whatever GT data we receive (inputs dict + self.groundTruthKey) as well as whatever
	#  intermediate messages we recieved. This is programmable for every node. By default, we return all GTs and all
	#  received messages as possible inputs to the node's forward function
	def getInputs(self, inputs):
		nodeInputs = [inputs[self.groundTruthKey]]
		nodeKeys = ["GT"]
		for key in self.outputs.keys():
			nodeInputs.extend(self.outputs[key])
			nodeKeys.extend([key] * len(self.outputs[key]))
		return nodeInputs, nodeKeys

	def setGroundTruth(self, groundTruth):
		self.groundTruth = groundTruth

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
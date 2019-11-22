import numpy as np
from copy import copy

class Node:
	# A dictionary that gives a unique tag to all nodes by appending an increasing number to name.
	lastNodeID = 0
	unicityNodesDict = {}

	def __init__(self, name, groundTruthKey, lossFn, metrics, numDims):
		assert not name is "GT", "GT is a reserved keyword"
		self.name = Node.getUniqueName(name)
		self.groundTruthKey = groundTruthKey
		# This must be set at every iteration before calling getInputs and lossFn in Edge.
		self.groundTruth = None
		self.lossFn = lossFn
		self.outputs = {}

		# Probably will be removed as they are not necessary for a base node class
		self.metrics = metrics
		self.numDims = numDims

	# This node's inputs based on whatever GT data we receive (inputs dict + self.groundTruthKey) as well as whatever
	#  intermediate messages we recieved. This is programmable for every node. By default, we return all GTs and all
	#  received messages as possible inputs to the node's forward function
	def getInputs(self, inputs):
		# TODO: Add possibility to not use GT OR to use a GT from outputs
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
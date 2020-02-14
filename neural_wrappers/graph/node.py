from ..pytorch import getTrData

class Node:
	# A dictionary that gives a unique tag to all nodes by appending an increasing number to name.
	lastNodeID = 0

	def __init__(self, name, groundTruthKey, hyperParameters={}):
		assert not name is "GT", "GT is a reserved keyword"
		self.name = Node.getUniqueName(name)
		self.groundTruthKey = groundTruthKey

		# Set up hyperparameters for this node (used for saving/loading identical node)
		self.hyperParameters = self.getHyperParameters(hyperParameters)
		self.setGroundTruth(None)
		# Messages are the items received at this node via all its incoming edges.
		self.messages = {}

		# Node-specific encoder and decoder instances. By default they are not instancicated.
		self.nodeEncoder = None
		self.nodeDecoder = None

	# This function is called for getEncoder/getDecoder. By default we'll return the normal type of this function.
	#  However, we are free to overwrite what type a node offers to be seen as. A concrete example is a
	#  ConcatenateNode, which might be more useful to be seen as a MapNode (if it concatenates >=2 MapNodes)
	def getType(self):
		return type(self)

	def getEncoder(self, outputNodeType=None):
		raise Exception("Must be implemented by each node!")

	def getDecoder(self, inputNodeType=None):
		raise Exception("Must be implemented by each node!")

	def getMetrics(self):
		raise Exception("Must be implemented by each node!")

	def getCriterion(self):
		raise Exception("Must be implemented by each node!")

	def getInputs(self, x):
		inputs = self.getMessages()
		if not self.groundTruth is None:
			inputs["GT"] = self.getGroundTruthInput(x).unsqueeze(0)
		return inputs

	def getMessages(self):
		return {k : getTrData(self.messages[k]) for k in self.messages}

	def addMessage(self, edgeID, message):
		self.messages[edgeID] = message

	def setGroundTruth(self, groundTruth):
		# Ground truth is always detached from the graph, so we don't optimize both sides of the graph, if the GT of
		#  one particular node was generated from other side.
		self.groundTruth = groundTruth
		if type(self.groundTruth) in (dict, ):
			self.groundTruth = {k : self.groundTruth[k].detach() for k in self.groundTruth}
		elif not self.groundTruth is None:
			self.groundTruth = self.groundTruth.detach()

	def getGroundTruth(self):
		return self.groundTruth

	def getGroundTruthInput(self, inputs):
		assert not self.groundTruthKey is None
		if type(self.groundTruthKey) is str:
			return inputs[self.groundTruthKey]
		elif type(self.groundTruthKey) in (list, tuple):
			return [inputs[key] for key in self.groundTruthKey]
		assert False

	def clearNodeOutputs(self):
		self.outputs = {}

	@staticmethod
	def getUniqueName(name):
		name = "%s (ID: %d)" % (name, Node.lastNodeID)
		Node.lastNodeID += 1
		return name

	def getHyperParameters(self, hyperParameters):
		# This is some weird bug. If i leave the same hyperparameters coming (here I make a shallow copy),
		#  making two instances of the same class results in having same hyperparameters.
		hyperParameters = {k : hyperParameters[k] for k in hyperParameters.keys()}
		hyperParameters["name"] = self.name
		hyperParameters["groundTruthKey"] = self.groundTruthKey
		return hyperParameters

	def __str__(self):
		return self.name

	def __repr__(self):
		return self.name.split(" ")[0]

class VectorNode(Node): pass
class MapNode(Node): pass
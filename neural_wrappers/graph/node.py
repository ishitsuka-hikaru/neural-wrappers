class Node:
	# A dictionary that gives a unique tag to all nodes by appending an increasing number to name.
	lastNodeID = 0

	def __init__(self, name, groundTruthKey, hyperParameters={}):
		assert not name is "GT", "GT is a reserved keyword"
		self.name = Node.getUniqueName(name)
		self.groundTruthKey = groundTruthKey

		# Set up hyperparameters for this node (used for saving/loading identical node)
		self.hyperParameters = self.getHyperParameters(hyperParameters)
		self.groundTruth = self.setGroundTruth(None)
		# Messages are the items received at this node via all its incoming edges.
		self.messages = {}

		# Node-specific encoder and decoder instances. By default they are not instancicated.
		self.nodeEncoder = None
		self.nodeDecoder = None

	def getEncoder(self, outputNodeType=None):
		raise Exception("Must be implemented by each node!")

	def getDecoder(self, inputNodeType=None):
		raise Exception("Must be implemented by each node!")

	def getMetrics(self):
		raise Exception("Must be implemented by each node!")

	# This node's inputs based on whatever GT data we receive (inputs dict + self.groundTruthKey) as well as whatever
	#  intermediate messages we recieved. This is programmable for every node. By default, we return all GTs and all
	#  received messages as possible inputs to the node's forward function
	def getInputs(self, blockGradients=False):
		items, edgeKeys = [], []
		# Add GT first, if it exist
		if not self.groundTruth is None:
			items.append(self.groundTruth)
			edgeKeys.append("GT")

		# All the messages are received from incoming edges to this node.
		for key in self.messages.keys():
			edgeMessages = self.messages[key]
			items.extend(edgeMessages)
			edgeKeys.extend([key] * len(edgeMessages))

		# If the edge that required this inputs wishes to prune the history of the inputs, detach them.
		# For debugging: Add a print here before and after detach for all items' grad_fn and requires_grad
		if blockGradients:
			items = [x.detach() for x in items]
		return items, edgeKeys

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

	def clearNodeOutputs(self):
		self.outputs = {}

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
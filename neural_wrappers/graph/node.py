from __future__ import annotations
import torch as tr
from typing import Optional, Dict, Type, Union, Any
from ..pytorch import trGetData, trDetachData, NWModule
# from .utils import _getGroundTruth

# TODO return type
def _getGroundTruth(labels:dict, groundTruthKey:str): #type: ignore
    # Combination of two functions. To be refactored :)
    if groundTruthKey is None:
        return None
    elif groundTruthKey == "*":
        return labels
    elif isinstance(groundTruthKey, str):
        return labels[groundTruthKey]
    elif isinstance(groundTruthKey, (list, tuple)):
        return {k : labels[k] for k in groundTruthKey}
    raise Exception("Key %s required from GT data not in labels %s" % (groundTruthKey, list(labels.keys())))

class Node:
	# A dictionary that gives a unique tag to all nodes by appending an increasing number to name.
	lastNodeID = 0

	def __init__(self, name : str, groundTruthKey : str, nodeEncoder : Optional[NWModule]=None, \
		nodeDecoder : Optional[NWModule]=None, hyperParameters : dict={}):
		assert name != "GT", "GT is a reserved keyword"
		self.name = Node.getUniqueName(name)
		self.groundTruthKey = groundTruthKey

		# Set up hyperparameters for this node (used for saving/loading identical node)
		self.hyperParameters = self.getHyperParameters(hyperParameters)
		self.groundTruth:Any = None
		# Messages are the items received at this node via all its incoming edges.
		self.messages : Dict[str, tr.Tensor] = {}

		# Node-specific encoder and decoder instances. By default they are not instancicated.
		self.nodeEncoder = nodeEncoder
		self.nodeDecoder = nodeDecoder

	# This function is called for getEncoder/getDecoder. By default we'll return the normal type of this function.
	#  However, we are free to overwrite what type a node offers to be seen as. A concrete example is a
	#  ConcatenateNode, which might be more useful to be seen as a MapNode (if it concatenates >=2 MapNodes)
	def getType(self) -> Type[Node]:
		return type(self)

	def getEncoder(self, outputNodeType : Optional[Node]=None):
		if not self.nodeEncoder is None:
			return self.nodeEncoder
		raise Exception("Must be implemented by each node!")

	def getDecoder(self, inputNodeType : Optional[Node]=None) -> Optional[NWModule]:
		if not self.getDecoder is None:
			return self.nodeDecoder
		raise Exception("Must be implemented by each node!")

	# TODO type
	def getMetrics(self) -> dict:
		raise Exception("Must be implemented by each node!")

	# TODO: Return callable
	def getCriterion(self):
		raise Exception("Must be implemented by each node!")

	def getInputs(self, x:tr.Tensor) -> Dict[str, tr.Tensor]:
		inputs = self.getMessages()
		GT:Optional[tr.Tensor] = self.groundTruth
		if not GT is None:
			res = self.getGroundTruthInput(x)
			if isinstance(res, (tuple, list)):
				res = [x[None] for x in res]
			elif isinstance(res, tr.Tensor):
				res = res.unsqueeze(0)
			inputs["GT"] = res
		return inputs

	def getMessages(self) -> Dict[str, tr.Tensor]:
		return {k : trGetData(self.messages[k]) for k in self.messages}

	def addMessage(self, edgeID:str, message:tr.Tensor) -> None:
		from .edge import Edge
		if isinstance(edgeID, Edge):
			edgeID:str = str(edgeID)
		self.messages[edgeID] = message

	# TODO: labels type
	def setGroundTruth(self, labels:Any):
		# labels = self.getNodeLabelOnly(labels) #type: ignore
		labels = _getGroundTruth(labels, self.groundTruthKey)
		# Ground truth is always detached from the graph, so we don't optimize both sides of the graph, if the GT of
		#  one particular node was generated from other side.
		labels = trDetachData(labels)
		self.groundTruth = labels

	def getGroundTruth(self) -> tr.Tensor:
		return self.groundTruth

	def getGroundTruthInput(self, inputs):
		assert not self.groundTruthKey is None
		if type(self.groundTruthKey) is str:
			return inputs[self.groundTruthKey]
		elif type(self.groundTruthKey) in (list, tuple):
			return [inputs[k] for k in self.groundTruthKey]
		assert False

	@staticmethod
	def getUniqueName(name : str) -> str:
		name = "%s (ID: %d)" % (name, Node.lastNodeID)
		Node.lastNodeID += 1
		return name

	def getHyperParameters(self, hyperParameters : dict) -> dict:
		# This is some weird bug. If i leave the same hyperparameters coming (here I make a shallow copy),
		#  making two instances of the same class results in having same hyperparameters.
		hyperParameters = {k : hyperParameters[k] for k in hyperParameters.keys()}
		hyperParameters["name"] = self.name
		hyperParameters["groundTruthKey"] = self.groundTruthKey
		return hyperParameters

	def __str__(self) -> str:
		return self.name

	def __repr__(self) -> str:
		return self.name.split(" ")[0]

class VectorNode(Node): pass
class MapNode(Node): pass
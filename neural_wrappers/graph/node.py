from __future__ import annotations
import torch as tr
import torch.nn as nn
from typing import Optional, Dict, Type, Union, Callable, Any
from abc import ABC, abstractmethod
from overrides import overrides
from ..pytorch import trGetData, trDetachData, NWModule
from ..metrics import Metric

GTType = Optional[Union[Dict[Any, Any], tr.Tensor]]

class Node(ABC, nn.Module):
	# A dictionary that gives a unique tag to all nodes by appending an increasing number to name.
	lastNodeID = 0

	def __init__(self, name:str, groundTruthKey:str, hyperParameters:dict={}):
		super().__init__()
		assert name != "GT", "GT is a reserved keyword"
		self.name = Node.getUniqueName(name)
		self.groundTruthKey = groundTruthKey

		# Set up hyperparameters for this node (used for saving/loading identical node)
		self.hyperParameters = self.getHyperParameters(hyperParameters)
		self.groundTruth:GTType = None
		# Messages are the items received at this node via all its incoming edges.
		self.messages:Dict[str, tr.Tensor] = {}

	# This function is called for getEncoder/getDecoder. By default we'll return the normal type of this function.
	#  However, we are free to overwrite what type a node offers to be seen as. A concrete example is a
	#  ConcatenateNode, which might be more useful to be seen as a MapNode (if it concatenates >=2 MapNodes)
	def getType(self) -> Type[Node]:
		return type(self)

	@abstractmethod
	def getEncoder(self, outputNodeType : Optional[Node]=None) -> NWModule:
		pass

	@abstractmethod
	def getDecoder(self, inputNodeType : Optional[Node]=None) -> NWModule:
		pass

	@abstractmethod
	def getNodeMetrics(self) -> Dict[str, Metric]:
		pass

	@abstractmethod
	def getNodeCriterion(self) -> Callable[[tr.Tensor, tr.Tensor, dict], tr.Tensor]:
		pass

	def getInputs(self, x : tr.Tensor) -> Dict[str, tr.Tensor]:
		inputs = self.getMessages()
		GT:Optional[Union[Dict[Any, tr.Tensor], tr.Tensor]] = self.groundTruth
		if not GT is None:
			inputs["GT"] = self.getGroundTruthInput(x).unsqueeze(0)
		return inputs

	def getMessages(self) -> Dict[str, tr.Tensor]:
		return {k : trGetData(self.messages[k]) for k in self.messages}

	def addMessage(self, edgeID : str, message : tr.Tensor) -> None:
		self.messages[edgeID] = message

	# TODO return type
	def getNodeLabelOnly(self, labels : dict): #type: ignore
		# Combination of two functions. To be refactored :)
		if self.groundTruthKey is None:
			return None
		elif self.groundTruthKey == "*":
			return labels
		elif (type(self.groundTruthKey) is str) and (self.groundTruthKey != "*"):
			return labels[self.groundTruthKey]
		elif type(self.groundTruthKey) in (list, tuple):
			return {k : labels[k] for k in self.groundTruthKey}
		raise Exception("Key %s required from GT data not in labels %s" % (self.groundTruthKey, list(labels.keys())))

	# TODO: labels type
	def setGroundTruth(self, labels:GTType):
		labels = self.getNodeLabelOnly(labels) #type: ignore
		# Ground truth is always detached from the graph, so we don't optimize both sides of the graph, if the GT of
		#  one particular node was generated from other side.
		labels = trDetachData(labels)
		self.groundTruth = labels

	def getGroundTruth(self) -> GTType:
		return self.groundTruth

	def getGroundTruthInput(self, inputs):
		assert not self.groundTruthKey is None
		if type(self.groundTruthKey) is str:
			return inputs[self.groundTruthKey]
		elif type(self.groundTruthKey) in (list, tuple):
			return [inputs[key] for key in self.groundTruthKey]
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
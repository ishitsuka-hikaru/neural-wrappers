import torch as tr
from functools import partial
from ..edge import Edge, defaultLossFn
from ..node import Node
from ...pytorch import trModuleWrapper

from itertools import product

# ReduceNode and ReduceEdge implementations.
class ConcatenateNode(Node):
	def __init__(self, nodes, concatenateFn, name=None, combineGTWithInputs=True, *args, **kwargs):
		if not name:
			name = "ConcatenateNode (%s)" % (", ".join(map(lambda x : x.name, nodes)))
		self.nodes = nodes
		groundTruthKeys = ConcatenateNode.gatherGroundTruthKeys(nodes)
		self.concatenateFn = concatenateFn
		self.combineGTWithInputs = combineGTWithInputs
		super().__init__(name, groundTruthKey=groundTruthKeys, *args, **kwargs)

	def getDecoder(self, inputNodeType=None):
		raise Exception("Concatenate Node mustn't be used as target node in any edge. ")

	def getMetrics(self):
		raise Exception("Concatenate Node mustn't be used as target node in any edge. ")

	def getCriterion(self):
		raise Exception("Concatenate Node mustn't be used as target node in any edge. ")

	@staticmethod
	def gatherGroundTruthKeys(nodes):
		groundTruthKeys = [node.groundTruthKey for node in nodes]
		if len(groundTruthKeys) > 1 and None in groundTruthKeys:
			groundTruthKeys = filter(lambda x : not x is None, groundTruthKeys)
		return groundTruthKeys

	def setGroundTruth(self, groundTruth):
		for node in self.nodes:
			node.setGroundTruth(groundTruth)

	def getGroundTruth(self):
		return self.concatenateFn({node : node.getGroundTruth() for node in self.nodes})

	def getGroundTruthInput(self, inputs):
		return self.concatenateFn({node : node.getGroundTruthInput() for node in self.nodes})

	# We will check the intersection of the MRO of all nodes and see if there is something better in the subtypes of
	#  all the nodes' MRO. If we fail to find anything, we'll just return ConcatenateNode.
	def getType(self):
		resultTypes = set(type(self.nodes[0]).mro())
		for node in self.nodes:
			nodeTypes = set(type(node).mro())
			resultTypes = resultTypes.intersection(nodeTypes)
		# This can result in something like:
		#  {<class 'object'>, <class 'neural_wrappers.graph.node.MapNode'>, <class 'neural_wrappers.graph.node.Node'>}
		# We need the most specific one (MapNode -> Node -> object) here.
		checked = {k : True for k in resultTypes}
		for Type in resultTypes:
			for subType in Type.mro():
				if Type == subType:
					continue
				checked[subType] = False

		Sum = sum(checked.values())
		# If most common ancestor IS Node/Object, then we give up and result ConcatenateNode
		if Sum == 0:
			return type(self)
		# However, If we got a more specific common ancestor, we return that (i.e. MapNode for RGB + Depth)
		elif Sum == 1:
			for Type in checked:
				if checked[Type] == True:
					return Type
		else:
			raise Exception("This should never be more than 1. Possible multiple inheritance.")

	# Actual example:
	# ConcatenateNode (RGB (ID: 0), Depth (ID: 1), Halftone (ID: 5)) (ID: 6) <class 'dict'>
	# RGB (ID: 0)
	#	- Wireframe (ID: 3) -> RGB (ID: 0) torch.Size([1, 3, 256, 256, 3])
	#	- CameraNormal (ID: 4) -> RGB (ID: 0) torch.Size([1, 3, 256, 256, 3])
	#	- GT torch.Size([1, 3, 256, 256, 3])
	# Depth (ID: 1)
	#	- GT torch.Size([1, 3, 256, 256, 1])
	# Halftone (ID: 5)
	#	- Wireframe (ID: 3) -> Halftone (ID: 5) torch.Size([1, 3, 256, 256, 3])
	#	- GT torch.Size([1, 3, 256, 256, 3])
	# So, we got 3 nodes. RGB, for example has 3 In-Edges and each in-edge has 1 message.
	# We need to build the cartesian product for each In-Edge of all nodes. So, the first input that gets sent to
	#  the concatenate function is [RGB (from Wireframe) + Depth (GT) + Halftone (from Wireframe)].
	# All the results are put under the "Concatenation" In-Edge.
	@staticmethod
	def cartesianProductOfInputs(inputs, concatenateFn):
		Keys = {node : list(inputs[node].keys()) for node in inputs}
		numKeys = {node : range(len(Keys[node])) for node in inputs}
		cartesianProduct = product(*numKeys.values())
		newInputs = []
		for item in cartesianProduct:
			thisInputs = {}
			# Might have to recheck this.
			for i, node in enumerate(Keys):
				Key = Keys[node][item[i]]
				thisNodeInput = inputs[node][Key]
				thisInputs[node] = thisNodeInput
			concatFunctionResult = concatenateFn(thisInputs)
			newInputs.append(concatFunctionResult)
		return tr.cat(newInputs, dim=0)

	def getInputs(self, x):
		inputs = {node : node.getInputs(x) for node in self.nodes}
		cartesianOfAll = {self : ConcatenateNode.cartesianProductOfInputs(inputs, self.concatenateFn)}

		Keys = {node : list(inputs[node].keys()) for node in inputs}
		# Check if all nodes have GT and if so, add the GT as well to the list of inputs. This also needs to be done
		#  via the cartesian product procedure, since, in theory, the GT can also have multiple mesages...
		allHaveGT = len(list(filter(lambda x : x.index("GT") != -1, Keys.values())))
		if allHaveGT == len(self.nodes):
			inputsGT = {k : {"GT" : inputs[k]["GT"]} for k in self.nodes}
			cartesianOfAll["GT"] = ConcatenateNode.cartesianProductOfInputs(inputsGT, self.concatenateFn)

		return cartesianOfAll
# Various utility functions regarding the concepts graph implementation.
import torch as tr
from ..edge import Edge
from ..node import Node
### Forward functions ###

# @brief Use all the possible inputs (GT or precomputed) for forward in edge.
# @param[in] self The edge object (which can access the model, inputNode, outputNode, edgeID etc.)
# @param[in] x The input to the input node
# @return The outputs (which are also stored in self.outputs). This is to preserve PyTorch's interface, while also
#  storing the intermediate results.
def forwardUseAll(self : Edge, x : tr.Tensor) -> tr.Tensor:
	outputs = []
	for key in x:
		for message in x[key]:
			y = self.model.forward(message)
			outputs.append(y)
	return tr.stack(outputs)

# @brief Use the GT as input to input node and nothing else.
# @param[in] self The edge object (which can access the model, inputNode, outputNode, edgeID etc.)
# @param[in] x The input to the input node
# @return The outputs (which are also stored in self.outputs). This is to preserve PyTorch's interface, while also
#  storing the intermediate results.
def forwardUseGT(self : Edge, x : tr.Tensor) -> tr.Tensor:
	y = self.model.forward(x["GT"][0]).unsqueeze(0)
	return y

# Use all incoming values as inputs, except GT.
# @brief Use the GT as input to input node and nothing else
# @param[in] self The edge object (which can access the model, inputNode, outputNode, edgeID etc.)
# @param[in] x The input to the input node
# @return The outputs (which are also stored in self.outputs). This is to preserve PyTorch's interface, while also
#  storing the intermediate results.
def forwardUseIntermediateResult(self : Edge, x : tr.Tensor) -> tr.Tensor:
	outputs = []
	for key in x:
		if key == "GT":
			continue
		for message in x[key]:
			y = self.model.forward(message)
			outputs.append(y)
	return tr.stack(outputs)

# @brief Transforms the GT of a node into a running mean of computed GT under "computed", as well as storing
#  original one in "GT".
def updateRunningMeanNodeGT(node : Node, result : tr.Tensor) -> None:
	GT = node.getGroundTruth()
	if not type(GT) is dict:
		GT = {"GT" : GT}
		GT["computed"] = result
		node.setGroundTruth(GT)
		node.count = 1
	else:
		# Running mean with GT
		GT["computed"] = (GT["computed"] * node.count + result) / (node.count + 1)
		node.setGroundTruth(GT)
		node.count += 1

def forwardUseAllStoreAvgGT(self : Edge, x : tr.Tensor) -> tr.Tensor:
	# Single link only to output node
	res = forwardUseAll(self, x)
	for i in range(len(res)):
		updateRunningMeanNodeGT(self.outputNode, res[i])
	return res

def forwardUseGTStoreAvgGT(self : Edge, x : tr.Tensor) -> tr.Tensor:
	res = forwardUseGT(self, x)
	for i in range(len(res)):
		updateRunningMeanNodeGT(self.outputNode, res[i])
	return res

def forwardUseIntermediateResultStoreAvgGT(self : Edge, x : tr.Tensor) -> tr.Tensor:
	res = forwardUseIntermediateResult(self, x)
	for i in range(len(res)):
		updateRunningMeanNodeGT(self.outputNode, res[i])
	return res
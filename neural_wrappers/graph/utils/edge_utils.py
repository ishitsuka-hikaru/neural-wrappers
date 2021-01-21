# Various utility functions regarding the concepts graph implementation.
import torch as tr
from ..edge import Edge
from ..node import Node
from typing import Dict, Sequence
### Forward functions ###

fwdFuncInput = Dict[str, Sequence[tr.Tensor]]

# @brief Use all the possible inputs (GT or precomputed) for forward in edge.
# @param[in] self The edge object (which can access the model, inputNode, outputNode, edgeID etc.)
# @param[in] x The input to the input node
# @return The outputs (which are also stored in self.outputs). This is to preserve PyTorch's interface, while also
#  storing the intermediate results.
def forwardUseAll(self:Edge, x:fwdFuncInput) -> tr.Tensor:
	outputs = []
	for key in x:
		nMessages = len(x[key])
		for i in range(nMessages):
			message = x[key][i]
			y = self.model.forward(message)
			outputs.append(y)
	return tr.stack(outputs)

# @brief Use the GT as input to input node and nothing else.
# @param[in] self The edge object (which can access the model, inputNode, outputNode, edgeID etc.)
# @param[in] x The input to the input node
# @return The outputs (which are also stored in self.outputs). This is to preserve PyTorch's interface, while also
#  storing the intermediate results.
def forwardUseGT(self:Edge, x:fwdFuncInput) -> tr.Tensor:
	y = self.model.forward(x["GT"][0]).unsqueeze(0)
	return y

# Use all incoming values as inputs, except GT.
# @brief Use the GT as input to input node and nothing else
# @param[in] self The edge object (which can access the model, inputNode, outputNode, edgeID etc.)
# @param[in] x The input to the input node
# @return The outputs (which are also stored in self.outputs). This is to preserve PyTorch's interface, while also
#  storing the intermediate results.
def forwardUseIntermediateResult(self:Edge, x:fwdFuncInput) -> tr.Tensor:
	outputs = []
	for key in x:
		if key == "GT":
			continue
		nMessages = len(x[key])
		for i in range(nMessages):
			message = x[key][i]
			y = self.model.forward(message)
			outputs.append(y)
	return tr.stack(outputs)

# @brief Transforms the GT of a node into a running mean of computed GT under "computed", as well as storing
#  original one in "GT".
def updateRunningMeanNodeGT(node:Node, result:tr.Tensor) -> None:
	GT = node.getGroundTruth()
	if not type(GT) is dict:
		newGT = {"GT":GT, "computed":result}
		node.setGroundTruth(newGT)
		node.count = 1 #type: ignore
	else:
		# Running mean with GT
		GT["computed"] = (GT["computed"] * node.count + result) / (node.count + 1) #type: ignore
		node.setGroundTruth(GT)
		node.count += 1 #type: ignore

	# Single link only to output node
def forwardUseAllStoreAvgGT(self:Edge, x: fwdFuncInput) -> tr.Tensor:
	res = forwardUseAll(self, x)
	for i in range(res.shape[0]):
		updateRunningMeanNodeGT(self.outputNode, res[i])
	return res

def forwardUseGTStoreAvgGT(self:Edge, x:fwdFuncInput) -> tr.Tensor:
	res = forwardUseGT(self, x)
	for i in range(res.shape[0]):
		updateRunningMeanNodeGT(self.outputNode, res[i])
	return res

def forwardUseIntermediateResultStoreAvgGT(self:Edge, x:fwdFuncInput) -> tr.Tensor:
	res = forwardUseIntermediateResult(self, x)
	for i in range(res.shape[0]):
		updateRunningMeanNodeGT(self.outputNode, res[i])
	return res
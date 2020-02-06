import torch as tr
# Various utility functions regarding the concepts graph implementation.

# @brief Use all the possible inputs (GT or precomputed) for forward in edge.
# @param[in] self The edge object (which can access the model, inputNode, outputNode, edgeID etc.)
# @param[in] x The input to the input node
# @return The outputs (which are also stored in self.outputs). This is to preserve PyTorch's interface, while also
#  storing the intermediate results.
def forwardUseAll(self, x):
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
def forwardUseGT(self, x):
	y = self.model.forward(x["GT"][0]).unsqueeze(0)
	return y

# Use all incoming values as inputs, except GT.
# @brief Use the GT as input to input node and nothing else
# @param[in] self The edge object (which can access the model, inputNode, outputNode, edgeID etc.)
# @param[in] x The input to the input node
# @return The outputs (which are also stored in self.outputs). This is to preserve PyTorch's interface, while also
#  storing the intermediate results.
def forwardUseIntermediateResult(self, x):
	outputs = []
	for key in x:
		if key == "GT":
			continue
		for message in x[key]:
			y = self.model.forward(message)
			outputs.append(y)
	return tr.stack(outputs)

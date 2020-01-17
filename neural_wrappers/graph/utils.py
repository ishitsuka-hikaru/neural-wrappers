# Various utility functions regarding the concepts graph implementation.

# @brief Use all the possible inputs (GT or precomputed) for forward in edge.
# @param[in] self The edge object (which can access the model, inputNode, outputNode, edgeID etc.)
# @param[in] x The input to the input node
# @return The outputs (which are also stored in self.outputs). This is to preserve PyTorch's interface, while also
#  storing the intermediate results.
def forwardUseAll(self, x):
	A, model = self.inputNode, self.model
	edgeInputs, _ = A.getInputs(blockGradients=self.blockGradients)
	self.inputs = []
	self.outputs = []

	for x in edgeInputs:
		self.inputs.append(x)
		y = model.forward(x)
		self.outputs.append(y)
	return self.outputs

# @brief Use the GT as input to input node and nothing else.
# @param[in] self The edge object (which can access the model, inputNode, outputNode, edgeID etc.)
# @param[in] x The input to the input node
# @return The outputs (which are also stored in self.outputs). This is to preserve PyTorch's interface, while also
#  storing the intermediate results.
def forwardUseGT(self, x):
	A, model = self.inputNode, self.model
	edgeInputs, inputNodeKeys = A.getInputs(blockGradients=self.blockGradients)
	self.inputs = []
	self.outputs = []

	x = edgeInputs[inputNodeKeys.index("GT")]
	y = model.forward(x)

	self.inputs = [x]
	self.outputs = [y]
	return self.outputs

# Use all incoming values as inputs, except GT.
# @brief Use the GT as input to input node and nothing else
# @param[in] self The edge object (which can access the model, inputNode, outputNode, edgeID etc.)
# @param[in] x The input to the input node
# @return The outputs (which are also stored in self.outputs). This is to preserve PyTorch's interface, while also
#  storing the intermediate results.
def forwardUseIntermediateResult(self, x):
	A, model = self.inputNode, self.model
	edgeInputs, inputNodeKeys = A.getInputs(blockGradients=self.blockGradients)
	self.inputs = []
	self.outputs = []

	for name, x in zip(inputNodeKeys, edgeInputs):
		if name == "GT":
			continue
		self.inputs.append(x)
		y = model.forward(x)
		self.outputs.append(y)
	return self.outputs
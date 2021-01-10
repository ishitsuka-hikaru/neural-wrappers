import numpy as np
from functools import partial
from overrides import overrides
from typing import Any, Dict, Iterator, Callable, Optional, Tuple

import torch as tr
import torch.nn as nn
import torch.optim as optim

from neural_wrappers.pytorch import device, FeedForwardNetwork
from neural_wrappers.graph import Graph, Edge, Node
from neural_wrappers.utilities import pickTypeFromMRO
from neural_wrappers.pytorch.layers import IdentityLayer
from neural_wrappers.metrics import Metric
from neural_wrappers.readers import BatchedDatasetReader, StaticBatchedDatasetReader

class Reader(BatchedDatasetReader):
	def __init__(self, dataStuff:Dict[Node, int]):
		self.N = 100
		self.dataset = {
			"data" : {a.groundTruthKey : np.random.randn(self.N, dataStuff[a]).astype(np.float32) for a in dataStuff}
		}
		#  dataBuckets : Dict[str, List[str]], dimGetter : Dict[str, DimGetterCallable], \
		# dimTransform : Dict[str, Dict[str, Callable]]
		dimGetterFn = lambda dataset, index, dim : dataset["data"][dim][index.start : index.stop]
		super().__init__(
			dataBuckets = {"data" : ["A", "B", "C", "D", "E"]}, \
			dimGetter = {"A" : partial(dimGetterFn, dim="A"), "B" : partial(dimGetterFn, dim="B"), \
				"C" : partial(dimGetterFn, dim="C"), "D" : partial(dimGetterFn, dim="D"), \
				"E" : partial(dimGetterFn, dim="E")}, \
			dimTransform = {}
		)

	@overrides
	def getDataset(self) -> Any:
		return self.dataset

	# @brief Returns the number of items in a given top level name
	# @param[in] topLevel The top-level dimension that is iterated over (example: train, validation, test, etc.)
	# @return The number of items in a given top level name
	@overrides
	def __len__(self) -> int:
		return self.N

	@overrides
	def __getitem__(self, index):
		item, B = super().__getitem__(index)
		return (item["data"], item["data"]), B

class Model(FeedForwardNetwork):
	def __init__(self, inDims, outDims):
		super().__init__()
		self.fc = nn.Linear(inDims, outDims)

	def forward(self, x):
		return self.fc(x)

class MyNode(Node):
	def __init__(self, nDims, name, gtKey):
		self.nDims = nDims
		super().__init__(name, gtKey)

	@overrides
	def getEncoder(self, outputNodeType : Optional[Node]=None) -> FeedForwardNetwork:
		modelTypes = {
			A : partial(Model, outDims=5),
			B : partial(Model, outDims=7),
			C : partial(Model, outDims=10),
			D : partial(Model, outDims=6),
			E : partial(Model, outDims=3),
		}
		return pickTypeFromMRO(outputNodeType, modelTypes)(inDims=self.nDims).to(device)

	@overrides
	def getDecoder(self, inputNodeType : Optional[Node]=None) -> IdentityLayer:
		return IdentityLayer().to(device)

	def getMetrics(self) -> Dict[str, Metric]:
		return {}

	def getCriterion(self) -> Callable[[tr.Tensor, tr.Tensor], tr.Tensor]:
		return lambda y, t : ((y - t)**2).mean()	

class A(MyNode):
	def __init__(self):
		super().__init__(nDims=5, name="A", gtKey="A")

class B(MyNode):
	def __init__(self):
		super().__init__(nDims=7, name="B", gtKey="B")

class C(MyNode):
	def __init__(self):
		super().__init__(nDims=10, name="C", gtKey="C")

class D(MyNode):
	def __init__(self):
		super().__init__(nDims=6, name="D", gtKey="D")

class E(MyNode):
	def __init__(self):
		super().__init__(nDims=3, name="E", gtKey="E")

class TestGraphStable:
	def test_get_inputs_1(self):
		MB = 13
		dataStuff = {A : 5, B : 7, C : 10, D : 6, E : 3}
		nodes = {A : A(), B : B(), C : C(), D : D(), E : E()}
		edges = [(A, C), (B, C), (C, E), (D, E)]
		graph = Graph([Edge(nodes[a], nodes[b]) for (a, b) in edges]).to(device)
		data = {nodes[a].groundTruthKey : tr.randn(MB, dataStuff[a]).to(device) for a in dataStuff}
		graph.iterationEpilogue(False, False, data)

		expectedOutputsShapes = {
			(A, C) : (1, 13, 10),
			(B, C) : (1, 13, 10),
			(C, E) : (3, 13, 3),
			(D, E) : (1, 13, 3)
		}
		for edge in graph.edges:
			edgeInputs = edge.getInputs(data)
			res = edge.forward(edgeInputs)
			Key = (type(edge.inputNode), type(edge.outputNode))
			assert res.shape == expectedOutputsShapes[Key]
		
		expectedInputsShapes = {
			A : [],
			B : [],
			C : [(1, 13, 10), (1, 13, 10)],
			D : [],
			E : [(1, 13, 3), (3, 13, 3)]
		}
		for node in graph.nodes:
			result = sorted(list(map(lambda x : tuple(x.shape), node.messages.values())))
			expected = sorted(expectedInputsShapes[type(node)])
			assert result == expected

	def test_train_1(self):
		nodes = {A : A(), B : B(), C : C(), D : D(), E : E()}
		reader = StaticBatchedDatasetReader(
			Reader(dataStuff={nodes[A] : 5, nodes[B] : 7, nodes[C] : 10, nodes[D] : 6, nodes[E] : 3}),
			batchSize=11)
		edges = [(A, C), (B, C), (C, E), (D, E)]
		graph = Graph([Edge(nodes[a], nodes[b]) for (a, b) in edges]).to(device)
		graph.setOptimizer(optim.SGD, lr=0.01)
		print(graph.summary())

		generator = reader.iterateForever()
		numSteps = len(generator)
		graph.train_generator(generator, numSteps, numEpochs=5)

if __name__ == "__main__":
	# TestGraph().test_get_inputs_1()
	TestGraphStable().test_train_1()
	pass
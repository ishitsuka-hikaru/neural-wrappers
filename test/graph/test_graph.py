import numpy as np
import torch.nn as nn
import torch as tr
from functools import partial

from neural_wrappers.pytorch import device, NeuralNetworkPyTorch
from neural_wrappers.graph import Graph, Edge, Node
from neural_wrappers.utilities import pickTypeFromMRO
from neural_wrappers.models import IdentityLayer

class Model(NeuralNetworkPyTorch):
	def __init__(self, inDims, outDims):
		super().__init__()
		self.fc = nn.Linear(inDims, outDims)

	def forward(self, x):
		return self.fc(x)

class MyNode(Node):
	def __init__(self, nDims, name, gtKey):
		self.nDims = nDims
		super().__init__(name, gtKey)

	def getEncoder(self, outputNodeType=None):
		modelTypes = {
			A : partial(Model, outDims=5),
			B : partial(Model, outDims=7),
			C : partial(Model, outDims=10),
			D : partial(Model, outDims=6),
			E : partial(Model, outDims=3),
		}
		return pickTypeFromMRO(outputNodeType, modelTypes)(inDims=self.nDims).to(device)

	def getDecoder(self, inputNodeType=None):
		return IdentityLayer().to(device)

	def getMetrics(self):
		return {}

	def getCriterion(self):
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

class TestGraph:
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

if __name__ == "__main__":
	TestGraph().test_get_inputs_1()
	pass
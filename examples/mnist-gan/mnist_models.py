import torch as tr
import torch.nn.functional as F
import torch.nn as nn
from neural_wrappers.pytorch import NeuralNetworkPyTorch

class GeneratorLinear(NeuralNetworkPyTorch):
	def __init__(self, inputSize, outputSize):
		super().__init__()
		assert len(outputSize) == 3
		self.inputSize = inputSize
		self.outputSize = outputSize

		self.fc1 = nn.Linear(self.inputSize, 128)
		self.fc2 = nn.Linear(128, 256)
		self.bn2 = nn.BatchNorm1d(256)
		self.fc3 = nn.Linear(256, 512)
		self.bn3 = nn.BatchNorm1d(512)
		self.fc4 = nn.Linear(512, 1024)
		self.bn4 = nn.BatchNorm1d(1024)
		self.fc5 = nn.Linear(1024, outputSize[0] * outputSize[1] * outputSize[2])

	def forward(self, x):
		y1 = F.leaky_relu(self.fc1(x))
		y2 = F.leaky_relu(self.bn2(self.fc2(y1)), negative_slope=0.2)
		y3 = F.leaky_relu(self.bn3(self.fc3(y2)), negative_slope=0.2)
		y4 = F.leaky_relu(self.bn4(self.fc4(y3)), negative_slope=0.2)
		y5 = tr.tanh(self.fc5(y4))
		y5 = y5.view(-1, *self.outputSize)
		return y5

class DiscriminatorLinear(NeuralNetworkPyTorch):
	def __init__(self, inputSize):
		super().__init__()
		assert len(inputSize) == 3
		self.inputSize = inputSize
		self.fc1 = nn.Linear(inputSize[0] * inputSize[1] * inputSize[2], 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 1)

	def forward(self, x):
		x = x.view(-1, self.inputSize[0] * self.inputSize[1] * self.inputSize[2])
		y1 = F.leaky_relu(self.fc1(x), negative_slope=0.2)
		y2 = F.leaky_relu(self.fc2(y1), negative_slope=0.2)
		y3 = tr.sigmoid(self.fc3(y2))
		y3 = y3.view(y3.shape[0])
		return y3

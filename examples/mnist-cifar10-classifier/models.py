from neural_wrappers.pytorch import NeuralNetworkPyTorch
import torch.nn as nn
from neural_wrappers.models import MobileNetV2Cifar10 as MobileNetV2

class ModelFC(NeuralNetworkPyTorch):
	def __init__(self, inputShape=(28, 28, 1), outputShape=10):
		super().__init__()

		self.outputShape = outputShape
		self.inputShape = inputShape
		self.inputShapeProd = inputShape[0] * inputShape[1] * inputShape[2]

		self.fc1 = nn.Linear(self.inputShapeProd, 100)
		self.fc2 = nn.Linear(100, 100)
		self.fc3 = nn.Linear(100, self.outputShape)

	def forward(self, x):
		x = x.view(-1, self.inputShapeProd)
		y1 = self.fc1(x)
		y2 = self.fc2(y1)
		y3 = self.fc3(y2)
		y4 = nn.functional.softmax(y3, dim=1)
		return y4

class ModelConv(NeuralNetworkPyTorch):
	def __init__(self, inputShape=(28, 28, 1), outputShape=10):
		super().__init__()

		if len(inputShape) == 2:
			inputShape = (*inputShape, 1)

		self.outputShape = outputShape
		self.inputShape = inputShape

		self.conv1 = nn.Conv2d(in_channels=inputShape[2], out_channels=100, kernel_size=3, stride=1)
		self.conv2 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, stride=1)
		self.fc1 = nn.Linear((inputShape[0] - 4) * (inputShape[1] - 4) * 100, 100)
		self.fc2 = nn.Linear(100, outputShape)

	def forward(self, x):
		x = x.view(-1, self.inputShape[2], self.inputShape[0], self.inputShape[1])
		y1 = self.conv1(x)
		y2 = self.conv2(y1)
		y2 = y2.view(-1, (self.inputShape[0] - 4) * (self.inputShape[1] - 4) * 100)
		y3 = self.fc1(y2)
		y4 = self.fc2(y3)
		y5 = nn.functional.softmax(y4, dim=1)
		return y5
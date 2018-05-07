import torch as tr
import torch.nn.functional as F
import torch.nn as nn
from neural_wrappers.models import MobileNetV2Cifar10
from neural_wrappers.pytorch import NeuralNetworkPyTorch

class GeneratorLinear(NeuralNetworkPyTorch):
	def __init__(self, inputSize, outputSize):
		super().__init__()
		assert len(outputSize) == 3
		self.inputSize = inputSize
		self.outputSize = outputSize

		self.fc1 = nn.Linear(100, 128)
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
		y5 = F.tanh(self.fc5(y4))
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
		y3 = F.sigmoid(self.fc3(y2))
		y3 = y3.view(y3.shape[0])
		return y3

class DiscriminatorMobileNetV2(MobileNetV2Cifar10):
	def __init__(self):
		super().__init__(num_classes=1)

	def forward(self, x):
		x = tr.transpose(tr.transpose(x, 1, 3), 2, 3)
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layers(out)
		out = F.relu(self.bn2(self.conv2(out)))
		# NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
		out = F.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		out = F.sigmoid(out)
		out = out.view(out.shape[0])
		return out
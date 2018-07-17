from neural_wrappers.pytorch import NeuralNetworkPyTorch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ModelFC(NeuralNetworkPyTorch):
	# (28, 28, 1) => (10, 1)
	def __init__(self, inputShape, outputNumClasses):
		super().__init__()

		self.inputShapeProd = int(np.prod(np.array(inputShape)))
		self.fc1 = nn.Linear(self.inputShapeProd, 100)
		self.fc2 = nn.Linear(100, 100)
		self.fc3 = nn.Linear(100, outputNumClasses)

	def forward(self, x):
		x = x.view(-1, self.inputShapeProd)
		y1 = F.relu(self.fc1(x))
		y2 = F.relu(self.fc2(y1))
		y3 = self.fc3(y2)
		return y3

class ModelConv(NeuralNetworkPyTorch):
	def __init__(self, inputShape, outputNumClasses):
		super().__init__()

		if len(inputShape) == 2:
			inputShape = (*inputShape, 1)

		self.inputShape = inputShape
		self.fc1InputShape = (inputShape[0] - 4) * (inputShape[1] - 4) * 10

		self.conv1 = nn.Conv2d(in_channels=inputShape[2], out_channels=50, kernel_size=3, stride=1)
		self.conv2 = nn.Conv2d(in_channels=50, out_channels=10, kernel_size=3, stride=1)
		self.fc1 = nn.Linear(self.fc1InputShape, 100)
		self.fc2 = nn.Linear(100, outputNumClasses)

	def forward(self, x):
		x = x.view(-1, self.inputShape[2], self.inputShape[0], self.inputShape[1])
		y1 = F.relu(self.conv1(x))
		y2 = F.relu(self.conv2(y1))
		y2 = y2.view(-1, self.fc1InputShape)
		y3 = F.relu(self.fc1(y2))
		y4 = self.fc2(y3)
		return y4
from neural_wrappers.pytorch import FeedForwardNetwork
import torch.nn as nn
import numpy as np

class ModelFC(FeedForwardNetwork):
	def __init__(self, inputShape=(28, 28), outputShape=10):
		super().__init__()

		self.outputShape = outputShape
		self.inputShape = inputShape
		self.inputShapeProd = int(np.prod(inputShape))

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
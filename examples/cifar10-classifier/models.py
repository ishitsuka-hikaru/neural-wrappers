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
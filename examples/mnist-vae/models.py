import torch as tr
import torch.nn as nn
import torch.nn.functional as F

from neural_wrappers.pytorch import FeedForwardNetwork

class Encoder(FeedForwardNetwork):
	def __init__(self, noiseSize):
		super().__init__()
		self.noiseSize = noiseSize
		self.fc1 = nn.Linear(28 * 28, 100)
		self.fc2 = nn.Linear(100, 100)
		self.mean_fc = nn.Linear(100, noiseSize)
		self.mean_std = nn.Linear(100, noiseSize)

	def forward(self, x):
		x = x.view(-1, 28 * 28)
		y1 = F.relu(self.fc1(x))
		y2 = F.relu(self.fc2(y1))
		y_mean = self.mean_fc(y2)
		y_std = self.mean_std(y2)
		return y_mean, y_std

class Decoder(FeedForwardNetwork):
	def __init__(self, noiseSize):
		super().__init__()
		self.noiseSize = noiseSize
		self.fc1 = nn.Linear(noiseSize, 300)
		self.fc2 = nn.Linear(300, 28 * 28)

	def forward(self, z_samples):
		y1 = F.relu(self.fc1(z_samples))
		y2 = self.fc2(y1)
		y_decoder = tr.sigmoid(y2)
		return y_decoder